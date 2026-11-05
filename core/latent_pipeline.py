"""Mug-style two-stage training: KL autoencoder + latent diffusion.

Stage 1: 1D KL autoencoder compresses the 4-lane piano roll (10 ms) into a
32-channel latent at 1/8 resolution.
Stage 2: diffusion on the latent, x0-prediction with L1 loss, conditioned on
downsampled log-mel + style/difficulty attributes; EMA weights for sampling.

This follows the training recipe in Mug-Diffusion's config:
  first_stage KL autoencoder -> latent diffusion, x0 parameterization, L1,
  linear beta schedule, condition dropout for CFG, mirror augmentation.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if (ROOT / "vendor_torch").exists():
    sys.path.insert(0, str(ROOT / "vendor_torch"))

DATA_DIR = ROOT / ".screen" / "neural"
AE_PATH = ROOT / "assets" / "latent_ae.pt"
DIFF_PATH = ROOT / "assets" / "latent_diffusion.pt"
CONFIG_PATH = ROOT / "assets" / "latent_pipeline.json"
STYLE_IDS = {"stream": 0, "speed": 1, "jack": 2, "tech": 3, "community": 4}
ATTR_DIM = 7
TIMESTEPS = 1000
LATENT_CHANNELS = 32
DOWN = 8


def _torch():
    import torch
    return torch


def load_index():
    combined = DATA_DIR / "index_combined.json"
    path = combined if combined.exists() else DATA_DIR / "index.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_song(name):
    data = np.load(DATA_DIR / f"{name}.npz")
    return data["mel"].astype(np.float32), data["labels"].astype(np.float32)


def attrs(record, labels):
    onehot = [0.0] * len(STYLE_IDS)
    onehot[STYLE_IDS.get(record["style"], 0)] = 1.0
    density = float(labels.sum()) / max(1.0, labels.size)
    return np.array(onehot + [min(1.0, density * 25.0), 0.0], dtype=np.float32)


class AE:
    def __init__(self):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = nn.Sequential(
                    nn.Conv1d(4, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.SiLU(),
                    nn.Conv1d(32, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
                    nn.Conv1d(64, 96, 4, 2, 1), nn.GroupNorm(8, 96), nn.SiLU(),
                )
                self.mean = nn.Conv1d(96, LATENT_CHANNELS, 1)
                self.logvar = nn.Conv1d(96, LATENT_CHANNELS, 1)
                self.dec = nn.Sequential(
                    nn.ConvTranspose1d(LATENT_CHANNELS, 96, 4, 2, 1),
                    nn.GroupNorm(8, 96), nn.SiLU(),
                    nn.ConvTranspose1d(96, 64, 4, 2, 1),
                    nn.GroupNorm(8, 64), nn.SiLU(),
                    nn.ConvTranspose1d(64, 32, 4, 2, 1),
                    nn.GroupNorm(8, 32), nn.SiLU(),
                    nn.Conv1d(32, 4, 1),
                )

            def encode(self, x):
                h = self.enc(x)
                return self.mean(h), self.logvar(h)

            def forward(self, x):
                torch = _torch()
                mean, logvar = self.encode(x)
                if self.training:
                    std = torch.exp(0.5 * logvar)
                    z = mean + std * torch.randn_like(std)
                else:
                    z = mean
                return self.dec(z), mean, logvar

        self.net = Net()


def train_autoencoder(epochs: int = 20, crop: int = 1024, batch: int = 8,
                      lr: float = 2e-4):
    torch = _torch()
    records = load_index()
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae = AE().net.to(device)
    optim = torch.optim.AdamW(ae.parameters(), lr=lr, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0, device=device))
    cache = {r["file"]: load_song(r["file"]) for r in records}
    for epoch in range(1, epochs + 1):
        ae.train()
        order = [r["file"] for r in records]
        rng.shuffle(order)
        total = 0.0
        steps = 0
        for start in range(0, len(order), batch):
            xs = []
            for name in order[start:start + batch]:
                mel, labels = cache[name]
                length = min(crop, labels.shape[1])
                index = int(rng.integers(0, max(1, labels.shape[1] - length)))
                patch = labels[:, index:index + length]
                if patch.shape[1] < crop:
                    patch = np.pad(patch, ((0, 0), (0, crop - patch.shape[1])))
                xs.append(torch.from_numpy(patch).float())
            x = torch.stack(xs).to(device)
            recon, mean, logvar = ae(x)
            recon_loss = bce(recon, x)
            kl = -0.5 * (1 + logvar - mean ** 2 - logvar.exp()).mean()
            loss = recon_loss + 1e-5 * kl
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += float(loss.detach())
            steps += 1
        print(f"AE epoch {epoch:02d} loss={total/max(1,steps):.4f}", flush=True)
    torch.save(ae.state_dict(), AE_PATH)
    print("saved", AE_PATH)


def _unet():
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(cin, cout, 5, padding=2), nn.GroupNorm(8, cout), nn.SiLU(),
                nn.Conv1d(cout, cout, 5, padding=2), nn.GroupNorm(8, cout), nn.SiLU(),
            )

        def forward(self, x):
            return self.net(x)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.time = nn.Embedding(TIMESTEPS, 128)
            self.attr = nn.Linear(ATTR_DIM, 128)
            self.inp = nn.Conv1d(LATENT_CHANNELS + 64, 64, 5, padding=2)
            self.cond_proj = nn.Linear(128, 64)
            self.e1 = Block(64, 64); self.e2 = Block(64, 128)
            self.mid = Block(128, 128)
            self.pool = nn.MaxPool1d(2)
            self.u2 = nn.ConvTranspose1d(128, 64, 2, stride=2)
            self.d2 = Block(64 + 128, 64)
            self.u1 = nn.ConvTranspose1d(64, 64, 2, stride=2)
            self.d1 = Block(64 + 64, 64)
            self.head = nn.Conv1d(64, LATENT_CHANNELS, 1)

        def forward(self, x, t, mel, attr):
            import torch
            import torch.nn.functional as F
            original = x.shape[-1]
            pad = (-original) % 4
            if pad:
                x = F.pad(x, (0, pad)); mel = F.pad(mel, (0, pad))
            cond = self.time(t) + self.attr(attr)
            h = self.inp(torch.cat([x, mel], dim=1)) + self.cond_proj(cond)[:, :, None]
            e1 = self.e1(h)
            e2 = self.e2(self.pool(e1))
            mid = self.mid(self.pool(e2))
            d2 = self.d2(torch.cat([self.u2(mid), e2], dim=1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
            return self.head(d1)[..., :original]

    return Net()


def _schedule(device):
    torch = _torch()
    betas = torch.linspace(1e-4, 0.02, TIMESTEPS, device=device)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    return betas, alpha_bar


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    def update(self, model):
        with _torch().no_grad():
            for target, source in zip(self.shadow.parameters(), model.parameters()):
                target.mul_(self.decay).add_(source, alpha=1 - self.decay)


def train_diffusion(epochs: int = 80, crop: int = 128, batch: int = 8,
                    lr: float = 5e-5):
    torch = _torch()
    records = load_index()
    rng = np.random.default_rng(1)
    torch.manual_seed(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae = AE().net.to(device)
    ae.load_state_dict(torch.load(AE_PATH, map_location=device, weights_only=True))
    ae.eval()
    model = _unet().to(device)
    ema = EMA(model)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    betas, alpha_bar = _schedule(device)
    cache = {r["file"]: load_song(r["file"]) for r in records}

    def make_batch(items):
        xs, mels, attrs_list = [], [], []
        for record in items:
            mel, labels = cache[record["file"]]
            length = min(crop * DOWN, labels.shape[1])
            index = int(rng.integers(0, max(1, labels.shape[1] - length)))
            patch = labels[:, index:index + length]
            if patch.shape[1] < crop * DOWN:
                patch = np.pad(patch, ((0, 0), (0, crop * DOWN - patch.shape[1])))
            mel_patch = mel[:, index:index + length]
            if mel_patch.shape[1] < crop * DOWN:
                mel_patch = np.pad(mel_patch, ((0, 0), (0, crop * DOWN - mel_patch.shape[1])))
            with torch.no_grad():
                latent = ae.encode(torch.from_numpy(patch).float().unsqueeze(0).to(device))[0]
            xs.append(latent[0].cpu())
            mels.append(torch.from_numpy(mel_patch).float())
            attrs_list.append(attrs(record, labels))
        x = torch.stack(xs).to(device) * 2.0
        mel = torch.stack(mels).to(device)
        attr = torch.from_numpy(np.stack(attrs_list)).to(device)
        if rng.random() < 0.3:
            x = torch.flip(x, dims=[1])
        if rng.random() < 0.5:
            attr = torch.zeros_like(attr)
        return x, mel, attr

    for epoch in range(1, epochs + 1):
        model.train()
        order = [r["file"] for r in records]
        rng.shuffle(order)
        total = 0.0
        steps = 0
        for start in range(0, len(order), batch):
            x0, mel, attr = make_batch([{"file": n, "style": next(r["style"] for r in records if r["file"] == n)} for n in order[start:start + batch]])
            # downsample mel to latent rate
            import torch.nn.functional as F
            mel = F.avg_pool1d(mel, DOWN)
            noise = torch.randn_like(x0)
            t = torch.randint(0, TIMESTEPS, (x0.shape[0],), device=device)
            ab = alpha_bar[t][:, None, None]
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
            pred = model(xt, t, mel, attr)
            loss = torch.nn.functional.l1_loss(pred, x0)
            optim.zero_grad()
            loss.backward()
            optim.step()
            ema.update(model)
            total += float(loss.detach())
            steps += 1
        print(f"DIFF epoch {epoch:03d} loss={total/max(1,steps):.4f}", flush=True)
        if epoch % 20 == 0:
            torch.save({"model": model.state_dict(), "ema": ema.shadow.state_dict()}, DIFF_PATH)
    torch.save({"model": model.state_dict(), "ema": ema.shadow.state_dict()}, DIFF_PATH)
    CONFIG_PATH.write_text(json.dumps({"records": len(records),
                                       "attrs": [f"style{i}" for i in range(len(STYLE_IDS))] + ["difficulty", "ln"]},
                                      indent=1), encoding="utf-8")
    print("saved", DIFF_PATH)




def sample_record(record, steps: int = 50, cfg: float = 1.0, seed: int = 0):
    torch = _torch()
    import torch.nn.functional as F
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae = AE().net.to(device)
    ae.load_state_dict(torch.load(AE_PATH, map_location=device, weights_only=True))
    ae.eval()
    model = _unet().to(device)
    payload = torch.load(DIFF_PATH, map_location=device, weights_only=True)
    model.load_state_dict(payload.get("ema", payload["model"]))
    model.eval()
    mel, labels = load_song(record["file"])
    mel_t = torch.from_numpy(mel).float().unsqueeze(0).to(device)
    mel_ds = F.avg_pool1d(mel_t, DOWN)
    attr = torch.from_numpy(attrs(record, labels)).float().unsqueeze(0).to(device)
    length = int(mel.shape[1] // DOWN)
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(1, LATENT_CHANNELS, length, device=device, generator=g)
    betas, alpha_bar = _schedule(device)
    step_list = np.linspace(TIMESTEPS - 1, 0, steps).astype(int)
    with torch.no_grad():
        for i, t in enumerate(step_list):
            tt = torch.tensor([t], device=device)
            x0 = model(x, tt, mel_ds, attr)
            if cfg != 1.0:
                x0_u = model(x, tt, torch.zeros_like(mel_ds), torch.zeros_like(attr))
                x0 = x0_u + cfg * (x0 - x0_u)
            ab = alpha_bar[t]
            if i + 1 < len(step_list):
                t_next = step_list[i + 1]
                ab_next = alpha_bar[t_next]
                x = (ab_next.sqrt() * x0 + ((1 - ab_next) / (1 - ab)).sqrt() * (x - ab.sqrt() * x0))
            else:
                x = x0
        recon = ae.dec(x)
    prob = torch.sigmoid(recon)[0].cpu().numpy()
    return prob


def decode_latent(prob, bpm, step=0.25, threshold=0.5, gap=3):
    rows = {}
    for lane in range(4):
        line = prob[lane]
        candidates = np.where(line > threshold)[0]
        if candidates.size == 0:
            continue
        order = candidates[np.argsort(line[candidates])[::-1]]
        chosen = []
        for index in order:
            if all(abs(int(index) - other) >= gap for other in chosen):
                chosen.append(int(index))
        for frame in chosen:
            beat = round((frame * 0.01 * bpm / 60.0) / step) * step
            rows[beat] = rows.get(beat, 0) | (1 << lane)
    return rows
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["ae", "diff"], default="ae", nargs="?")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if args.command == "ae":
        train_autoencoder(epochs=args.epochs)
    else:
        train_diffusion(epochs=args.epochs)
