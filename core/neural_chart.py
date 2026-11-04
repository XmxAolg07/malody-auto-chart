"""Frame-level neural chart model (MuG-Diffusion-inspired, CPU/GPU friendly).

Input : 64-band log-mel spectrogram, 10 ms frames.
Output: 4-lane note probability per frame.

This is a compact convolutional U-Net rather than a diffusion model: it learns
the audio-to-piano-roll mapping directly, then decodes peaks to notes.  It is a
research/experiment path; the rule-based generator remains the default.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if (ROOT / "vendor_torch").exists():
    sys.path.insert(0, str(ROOT / "vendor_torch"))
CORPUS = ROOT / "...AAA实例谱子存放处！"
if not (CORPUS / "ex stream").exists():
    CORPUS = ROOT
DATA_DIR = ROOT / ".screen" / "neural"
MODEL_PATH = ROOT / "assets" / "chartnet.pt"
CONFIG_PATH = ROOT / "assets" / "chartnet.json"
FOLDERS = {
    "stream": ["ex stream/0", "reg stream/0"],
    "speed": ["ex speed/0", "reg speed/0"],
    "jack": ["ex jack/0", "reg jack/0"],
    "tech": ["ex tech/0", "reg tech/0"],
}
HOP_SEC = 0.01
N_MELS = 64
N_FFT = 1024


def _mel_bank(sr: int, n_fft: int, n_bands: int):
    from .mel import _mel_filterbank
    return _mel_filterbank(sr, n_fft, n_bands)


def log_mel(samples, sr: int) -> np.ndarray:
    import numpy as np
    from .mel import _mel_filterbank
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    hop = max(1, int(sr * HOP_SEC))
    if samples.size < N_FFT * 2:
        return np.zeros((N_MELS, 1), dtype=np.float32)
    count = 1 + (samples.size - N_FFT) // hop
    frames = np.lib.stride_tricks.as_strided(
        samples, shape=(count, N_FFT),
        strides=(samples.strides[0] * hop, samples.strides[0]), writeable=False)
    window = np.hanning(N_FFT).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)
    bank = _mel_filterbank(sr, N_FFT, N_MELS)
    mel = np.log1p(spectrum @ bank.T).T
    mel = (mel - mel.mean(axis=1, keepdims=True)) / (mel.std(axis=1, keepdims=True) + 1e-6)
    return mel.astype(np.float32)


def chart_rows(chart):
    times = []
    for item in chart.get("time") or []:
        beat = item["beat"]
        times.append((beat[0] + beat[1] / beat[2], float(item["bpm"])))
    times.sort()
    segments = []
    seconds = 0.0
    for index, (beat, bpm) in enumerate(times):
        if index:
            prev_beat, prev_seconds, prev_bpm = segments[-1]
            seconds = prev_seconds + (beat - prev_beat) * 60.0 / prev_bpm
        segments.append((beat, seconds, bpm))
    rows = {}
    for note in chart.get("note") or []:
        if isinstance(note, dict):
            beat_list, lane = note.get("beat"), note.get("column")
        else:
            beat_list, lane = note[0], note[1]
        if beat_list is None or lane is None:
            continue
        beat = beat_list[0] + beat_list[1] / beat_list[2]
        segment = segments[0]
        for candidate in segments:
            if candidate[0] <= beat + 1e-9:
                segment = candidate
            else:
                break
        sec = segment[1] + (beat - segment[0]) * 60.0 / segment[2]
        key = round(sec, 3)
        rows[key] = rows.get(key, 0) | (1 << int(lane))
    return sorted(rows.items()), times


def build_dataset(force: bool = False) -> list[dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = DATA_DIR / "index.json"
    if meta_path.exists() and not force:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    from .audio import load_audio
    records = []
    for style, folders in FOLDERS.items():
        index = 0
        for folder in folders:
            base = CORPUS / folder
            for path in sorted(base.glob("*.mc")):
                data = json.loads(path.read_text(encoding="utf-8"))
                background = (data.get("meta") or {}).get("background") or ""
                audio = base / (Path(background).stem + ".ogg")
                if not audio.exists():
                    continue
                samples, sr = load_audio(str(audio))
                mel = log_mel(samples, sr)
                rows, times = chart_rows(data)
                labels = np.zeros((4, mel.shape[1]), dtype=np.uint8)
                for sec, mask in rows:
                    frame = int(round(sec / HOP_SEC))
                    if 0 <= frame < labels.shape[1]:
                        for lane in range(4):
                            if mask & (1 << lane):
                                labels[lane, frame] = 1
                name = f"{style}_{index:02d}"
                np.savez_compressed(DATA_DIR / f"{name}.npz", mel=mel,
                                    labels=labels, name=path.stem)
                bpm = times[0][1] if times else 180.0
                records.append({"file": name, "style": style, "bpm": bpm,
                                "audio": str(audio), "chart": str(path),
                                "frames": int(mel.shape[1])})
                print(style, index, path.stem[:40], mel.shape, flush=True)
                index += 1
    meta_path.write_text(json.dumps(records, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    return records


class ChartNet:  # placeholder for type checkers; real class defined lazily
    pass


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

        def forward(self, x):
            import torch.nn.functional as F
            original = x.shape[-1]
            pad = (-original) % 8
            if pad:
                x = F.pad(x, (0, pad))
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            e4 = self.e4(self.pool(e3))
            d3 = self.d3(torch.cat([self.u3(e4), e3], dim=1))
            d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
            return self.head(d1)[..., :original]

    return Net()


def train(epochs: int = 20, crop: int = 1024, batch: int = 8) -> None:
    import torch
    records = build_dataset()
    if not records:
        raise SystemExit("empty dataset")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    audios = sorted({record["audio"] for record in records})
    rng.shuffle(audios)
    val_audios = set(audios[: max(1, len(audios) // 5)])
    val_names = {record["file"] for record in records if record["audio"] in val_audios}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _model().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(6.0, device=device))

    def load(name):
        data = np.load(DATA_DIR / f"{name}.npz")
        return data["mel"], data["labels"]

    def make_batch(names, length):
        mels, labels = [], []
        for name in names:
            mel, lab = load(name)
            if mel.shape[1] <= length:
                pad = length - mel.shape[1] + 1
                mel = np.pad(mel, ((0, 0), (0, pad)))
                lab = np.pad(lab, ((0, 0), (0, pad)))
            start = int(rng.integers(0, mel.shape[1] - length))
            mels.append(torch.from_numpy(mel[:, start:start + length]).float())
            labels.append(torch.from_numpy(lab[:, start:start + length]).float())
        mels = torch.stack(mels).to(device)
        labels = torch.stack(labels).to(device)
        if rng.random() < 0.5:
            labels = torch.flip(labels, dims=[1])
        return mels, labels

    train_names = [r["file"] for r in records if r["file"] not in val_names]
    val_names_list = [r["file"] for r in records if r["file"] in val_names]
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(train_names)
        total_loss = 0.0
        steps = 0
        for start in range(0, len(train_names), batch):
            names = train_names[start:start + batch]
            x, y = make_batch(names, crop)
            optim.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()
            total_loss += float(loss.detach())
            steps += 1
        model.eval()
        val_loss = 0.0
        probs_all, labels_all = [], []
        with torch.no_grad():
            for name in val_names_list[:12]:
                mel, lab = load(name)
                x = torch.from_numpy(mel).float().unsqueeze(0).to(device)
                y = torch.from_numpy(lab).float().unsqueeze(0).to(device)
                logits = model(x)
                val_loss += float(loss_fn(logits, y))
                probs_all.append(torch.sigmoid(logits).cpu())
                labels_all.append(y.cpu())
        best = (0.0, 0.35, 0.0, 0.0)
        for threshold in np.arange(0.20, 0.76, 0.05):
            tp = pred_count = pos_count = 0.0
            for prob, label in zip(probs_all, labels_all):
                guess = (prob > threshold).float()
                tp += float((guess * label).sum())
                pred_count += float(guess.sum())
                pos_count += float(label.sum())
            precision = tp / max(1.0, pred_count)
            recall = tp / max(1.0, pos_count)
            f1 = 2 * precision * recall / max(1e-6, precision + recall)
            if f1 > best[0]:
                best = (f1, float(threshold), precision, recall)
        f1, best_threshold, precision, recall = best
        history.append({"epoch": epoch, "train_loss": total_loss / max(1, steps),
                        "val_loss": val_loss / max(1, len(val_names_list[:12])),
                        "precision": precision, "recall": recall, "f1": f1, "threshold": best_threshold})
        print(f"epoch {epoch:02d} train={total_loss/max(1,steps):.4f} "
              f"val={val_loss/max(1,len(val_names_list[:12])):.4f} "
              f"P={precision:.3f} R={recall:.3f} F1={f1:.3f} thr={best_threshold:.2f}", flush=True)
    torch.save(model.state_dict(), MODEL_PATH)
    CONFIG_PATH.write_text(json.dumps({"history": history, "n_mels": N_MELS,
                                       "hop_sec": HOP_SEC,
                                       "val": sorted(val_names)}, indent=1),
                           encoding="utf-8")
    print("saved", MODEL_PATH)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "train"], default="train", nargs="?")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if args.command == "build":
        build_dataset(force=True)
    else:
        train(epochs=args.epochs)
