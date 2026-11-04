"""Second neural baseline: Gaussian-smoothed labels + style conditioning.

Keeps the 1D U-Net but improves the learning signal (notes become soft bumps
instead of one-frame spikes), conditions the bottleneck on the key-pattern
style, and evaluates note-level F1 after peak decoding.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if (ROOT / "vendor_torch").exists():
    sys.path.insert(0, str(ROOT / "vendor_torch"))

from .neural_chart import (DATA_DIR, HOP_SEC, MODEL_PATH, N_MELS,  # noqa: E402
                           build_dataset)

MODEL_V2 = ROOT / "assets" / "chartnet_v2.pt"
CONFIG_V2 = ROOT / "assets" / "chartnet_v2.json"
STYLE_IDS = {"stream": 0, "speed": 1, "jack": 2, "tech": 3}


def smooth_labels(labels: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    out = np.zeros_like(labels, dtype=np.float32)
    for lane in range(labels.shape[0]):
        out[lane] = np.convolve(labels[lane].astype(np.float32), kernel, mode="same")
    return np.clip(out, 0.0, 1.0)


def _model():
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(cin, cout, 5, padding=2),
                nn.GroupNorm(8, cout), nn.SiLU(),
                nn.Conv1d(cout, cout, 5, padding=2),
                nn.GroupNorm(8, cout), nn.SiLU(),
            )

        def forward(self, x):
            return self.net(x)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.style = nn.Embedding(4, 32)
            self.style_proj = nn.Linear(32, 128)
            self.e1 = Block(N_MELS, 32)
            self.e2 = Block(32, 64)
            self.e3 = Block(64, 96)
            self.e4 = Block(96, 128)
            self.pool = nn.MaxPool1d(2)
            self.u3 = nn.ConvTranspose1d(128, 96, 2, stride=2)
            self.d3 = Block(96 + 96, 96)
            self.u2 = nn.ConvTranspose1d(96, 64, 2, stride=2)
            self.d2 = Block(64 + 64, 64)
            self.u1 = nn.ConvTranspose1d(64, 32, 2, stride=2)
            self.d1 = Block(32 + 32, 32)
            self.head = nn.Conv1d(32, 4, 1)

        def forward(self, x, style):
            import torch.nn.functional as F
            original = x.shape[-1]
            pad = (-original) % 8
            if pad:
                x = F.pad(x, (0, pad))
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            e4 = self.e4(self.pool(e3))
            condition = self.style_proj(self.style(style))[:, :, None]
            e4 = e4 + condition
            d3 = self.d3(torch.cat([self.u3(e4), e3], dim=1))
            d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
            return self.head(d1)[..., :original]

    return Net()


def decode_notes(prob: np.ndarray, threshold: float = 0.35, gap: int = 3):
    notes = {}
    for lane in range(4):
        line = prob[lane]
        candidates = np.where(line > threshold)[0]
        if candidates.size == 0:
            continue
        order = candidates[np.argsort(line[candidates])[::-1]]
        taken = []
        for index in order:
            if all(abs(index - other) >= gap for other in taken):
                taken.append(int(index))
        for index in taken:
            notes[(int(index), lane)] = float(line[index])
    return notes


def note_f1(prob: np.ndarray, target: np.ndarray, threshold=0.35, gap=3):
    predicted = decode_notes(prob, threshold, gap)
    truth = {(int(frame), lane) for lane in range(4)
             for frame in np.where(target[lane] > 0.5)[0]}
    hits = 0
    for frame, lane in predicted:
        if any(abs(frame - other) <= gap and lane == other_lane
               for other, other_lane in truth):
            hits += 1
    precision = hits / max(1, len(predicted))
    recall = hits / max(1, len(truth))
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    return f1, precision, recall


def train(epochs: int = 100, crop: int = 1024, batch: int = 8,
          sigma: float = 2.0) -> None:
    import torch
    records = build_dataset()
    if not records:
        raise SystemExit("empty dataset")
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    cache = {}

    def load(name):
        if name in cache:
            return cache[name]
        data = np.load(DATA_DIR / f"{name}.npz")
        mel = data["mel"]
        labels = smooth_labels(data["labels"], sigma)
        cache[name] = (mel, labels)
        return mel, labels

    audios = sorted({record["audio"] for record in records})
    rng.shuffle(audios)
    val_audios = set(audios[: max(1, len(audios) // 5)])
    val_names = [r["file"] for r in records if r["audio"] in val_audios]
    train_names = [r["file"] for r in records if r["audio"] not in val_audios]
    style_of = {r["file"]: STYLE_IDS.get(r["style"], 0) for r in records}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _model().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4.0, device=device))

    def make_batch(names, length):
        mels, labels, styles = [], [], []
        for name in names:
            mel, lab = load(name)
            if mel.shape[1] <= length:
                pad = length - mel.shape[1] + 1
                mel = np.pad(mel, ((0, 0), (0, pad)))
                lab = np.pad(lab, ((0, 0), (0, pad)))
            start = int(rng.integers(0, mel.shape[1] - length))
            mels.append(torch.from_numpy(mel[:, start:start + length]).float())
            labels.append(torch.from_numpy(lab[:, start:start + length]).float())
            styles.append(style_of[name])
        mels = torch.stack(mels).to(device)
        labels = torch.stack(labels).to(device)
        if rng.random() < 0.5:
            labels = torch.flip(labels, dims=[1])
        return mels, labels, torch.tensor(styles, device=device)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(train_names)
        total = 0.0
        steps = 0
        for start in range(0, len(train_names), batch):
            names = train_names[start:start + batch]
            x, y, style = make_batch(names, crop)
            optim.zero_grad()
            logits = model(x, style)
            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()
            total += float(loss.detach())
            steps += 1
        model.eval()
        frame_tp = frame_pred = frame_pos = note_total = note_pred = note_pos = 0
        f1s = []
        with torch.no_grad():
            for name in val_names[:10]:
                mel, lab = load(name)
                x = torch.from_numpy(mel).float().unsqueeze(0).to(device)
                style = torch.tensor([style_of[name]], device=device)
                prob = torch.sigmoid(model(x, style))[0].cpu().numpy()
                target = (lab > 0.5).astype(np.float32)
                guess = (prob > 0.35).astype(np.float32)
                frame_tp += float((guess * target).sum())
                frame_pred += float(guess.sum())
                frame_pos += float(target.sum())
                f1, _p, _r = note_f1(prob, lab)
                f1s.append(f1)
        precision = frame_tp / max(1.0, frame_pred)
        recall = frame_tp / max(1.0, frame_pos)
        frame_f1 = 2 * precision * recall / max(1e-6, precision + recall)
        note = float(np.mean(f1s)) if f1s else 0.0
        history.append({"epoch": epoch, "loss": total / max(1, steps),
                        "frame_f1": frame_f1, "note_f1": note})
        print(f"epoch {epoch:03d} loss={total/max(1,steps):.4f} "
              f"frameF1={frame_f1:.3f} noteF1={note:.3f}", flush=True)
        if epoch % 20 == 0:
            torch.save(model.state_dict(), MODEL_V2)
    torch.save(model.state_dict(), MODEL_V2)
    CONFIG_V2.write_text(json.dumps({"history": history, "sigma": sigma,
                                     "val": sorted(val_names)}, indent=1),
                         encoding="utf-8")
    print("saved", MODEL_V2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    train(epochs=args.epochs)
