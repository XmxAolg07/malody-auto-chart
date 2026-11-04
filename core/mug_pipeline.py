"""Small-scale MuG-style conditional diffusion chart generator.

Design mirrors Mug-Diffusion's pipeline:
  * 64-band log-mel spectrogram (10 ms frames) as global audio condition
  * 4-lane piano roll (note on/off represented by continuous values) as data
  * DDPM/DDIM denoiser (1D U-Net) conditioned on audio + attributes
  * attributes: style one-hot, difficulty (note density), LN ratio (0 here)
  * classifier-free guidance, seed, step count for diversity
  * gridify + mini-jack cleanup after decoding

It is deliberately small: 76 training songs is far below what a diffusion model
needs, so this is an architectural experiment, not a replacement.
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

DATA_DIR = ROOT / ".screen" / "neural"
MODEL_PATH = ROOT / "assets" / "mug_pipeline.pt"
CONFIG_PATH = ROOT / "assets" / "mug_pipeline.json"
STYLE_IDS = {"stream": 0, "speed": 1, "jack": 2, "tech": 3}
ATTR_DIM = 6  # style one-hot(4) + difficulty + ln ratio
TIMESTEPS = 1000


def _torch():
    import torch
    return torch


def load_index():
    return json.loads((DATA_DIR / "index.json").read_text(encoding="utf-8"))


def load_song(name):
    data = np.load(DATA_DIR / f"{name}.npz")
    return data["mel"].astype(np.float32), data["labels"].astype(np.float32)


def attributes(record, labels):
    onehot = [0.0] * 4
    onehot[STYLE_IDS.get(record["style"], 0)] = 1.0
    density = float(labels.sum()) / max(1.0, labels.size)
    difficulty = min(1.0, density * 25.0)
    return np.array(onehot + [difficulty, 0.0], dtype=np.float32)


def _model():
    torch = _torch()
    import torch.nn as nn

    class FiLM(nn.Module):
        def __init__(self, dim, channels):
            super().__init__()
            self.scale = nn.Linear(dim, channels)
            self.shift = nn.Linear(dim, channels)

        def forward(self, h, cond):
            return h * (1 + self.scale(cond))[:, :, None] + self.shift(cond)[:, :, None]

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
            self.time = nn.Embedding(TIMESTEPS, 128)
            self.attr = nn.Linear(ATTR_DIM, 128)
            self.inp = nn.Conv1d(4 + 64, 64, 5, padding=2)
            self.e1 = Block(64, 64)
            self.e2 = Block(64, 128)
            self.e3 = Block(128, 192)
            self.mid = Block(192, 192)
            self.pool = nn.MaxPool1d(2)
            self.u3 = nn.ConvTranspose1d(192, 128, 2, stride=2)
            self.d3 = Block(128 + 192, 128)
            self.u2 = nn.ConvTranspose1d(128, 64, 2, stride=2)
            self.d2 = Block(64 + 128, 64)
            self.u1 = nn.ConvTranspose1d(64, 64, 2, stride=2)
            self.d1 = Block(64 + 64, 64)
            self.film1 = FiLM(128, 64)
            self.film2 = FiLM(128, 128)
            self.film3 = FiLM(128, 192)
            self.head = nn.Conv1d(64, 4, 1)

        def forward(self, x, t, mel, attr):
            import torch.nn.functional as F
            original = x.shape[-1]
            pad = (-original) % 8
            if pad:
                x = F.pad(x, (0, pad))
                mel = F.pad(mel, (0, pad))
            cond = self.time(t) + self.attr(attr)
            h = self.inp(torch.cat([x, mel], dim=1))
            e1 = self.film1(self.e1(h), cond)
            e2 = self.film2(self.e2(self.pool(e1)), cond)
            e3 = self.film3(self.e3(self.pool(e2)), cond)
            mid = self.mid(self.pool(e3))
            d3 = self.d3(torch.cat([self.u3(mid), e3], dim=1))
            d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
            return self.head(d1)[..., :original]

    return Net()


def _schedule(device):
    torch = _torch()
    betas = torch.linspace(1e-4, 0.02, TIMESTEPS, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar


def train(epochs: int = 120, crop: int = 512, batch: int = 8, lr: float = 2e-4):
    torch = _torch()
    records = load_index()
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    cache = {r["file"]: load_song(r["file"]) for r in records}
    audios = sorted({r["audio"] for r in records})
    rng.shuffle(audios)
    val_audios = set(audios[: max(1, len(audios) // 5)])
    train_records = [r for r in records if r["audio"] not in val_audios]
    val_records = [r for r in records if r["audio"] in val_audios]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _model().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    betas, alphas, alpha_bar = _schedule(device)

    def make_batch(items, length):
        xs, mels, attrs = [], [], []
        for record in items:
            mel, labels = cache[record["file"]]
            if mel.shape[1] <= length:
                pad = length - mel.shape[1] + 1
                mel = np.pad(mel, ((0, 0), (0, pad)))
                labels = np.pad(labels, ((0, 0), (0, pad)))
            start = int(rng.integers(0, mel.shape[1] - length))
            xs.append(torch.from_numpy(labels[:, start:start + length]).float())
            mels.append(torch.from_numpy(mel[:, start:start + length]).float())
            attrs.append(attributes(record, labels))
        x = torch.stack(xs).to(device) * 2.0 - 1.0
        mel = torch.stack(mels).to(device)
        attr = torch.from_numpy(np.stack(attrs)).to(device)
        if rng.random() < 0.5:
            x = torch.flip(x, dims=[1])
        if rng.random() < 0.1:
            mel = torch.zeros_like(mel)
            attr = torch.zeros_like(attr)
        return x, mel, attr

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(train_records)
        total = 0.0
        steps = 0
        for start in range(0, len(train_records), batch):
            x0, mel, attr = make_batch(train_records[start:start + batch], crop)
            noise = torch.randn_like(x0)
            t = torch.randint(0, TIMESTEPS, (x0.shape[0],), device=device)
            ab = alpha_bar[t][:, None, None]
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
            pred = model(xt, t, mel, attr)
            loss = torch.nn.functional.mse_loss(pred, noise)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += float(loss.detach())
            steps += 1
        history.append({"epoch": epoch, "loss": total / max(1, steps)})
        print(f"epoch {epoch:03d} loss={total/max(1,steps):.4f}", flush=True)
        if epoch % 20 == 0:
            torch.save(model.state_dict(), MODEL_PATH)
    torch.save(model.state_dict(), MODEL_PATH)
    CONFIG_PATH.write_text(json.dumps({"history": history,
                                       "val": [r["file"] for r in val_records],
                                       "attrs": ["style0", "style1", "style2", "style3",
                                                 "difficulty", "ln"]}, indent=1),
                           encoding="utf-8")
    print("saved", MODEL_PATH)


def sample_for(record, steps: int = 50, cfg: float = 1.0, seed: int = 0,
               threshold: float = 0.5):
    """Generate a binary piano roll for one record (used by evaluation)."""
    torch = _torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    mel, labels = load_song(record["file"])
    mel_t = torch.from_numpy(mel).float().unsqueeze(0).to(device)
    attr = torch.from_numpy(attributes(record, labels)).float().unsqueeze(0).to(device)
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(1, 4, mel.shape[1], device=device, generator=g)
    betas, alphas, alpha_bar = _schedule(device)
    step_list = np.linspace(TIMESTEPS - 1, 0, steps).astype(int)
    with torch.no_grad():
        for i, t in enumerate(step_list):
            tt = torch.tensor([t], device=device)
            eps = model(x, tt, mel_t, attr)
            if cfg != 1.0:
                eps_u = model(x, tt, torch.zeros_like(mel_t), torch.zeros_like(attr))
                eps = eps_u + cfg * (eps - eps_u)
            ab = alpha_bar[t]
            x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
            x0 = x0.clamp(-1, 1)
            if i + 1 < len(step_list):
                t_next = step_list[i + 1]
                ab_next = alpha_bar[t_next]
                x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
            else:
                x = x0
    prob = (x[0].cpu().numpy() + 1.0) / 2.0
    return (prob > threshold).astype(np.float32), prob


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    args = parser.parse_args()
    train(epochs=args.epochs)
