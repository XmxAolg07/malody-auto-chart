"""MuG-style v2 framework: STFT/Mel + S4-like encoder -> piano-roll VAE -> latent diffusion.

This is a rewrite of the neural path, kept separate from the rule generator:

  audio : 64-band log-mel (10 ms) -> conv downsampling + bidirectional diagonal
          state-space (S4-like) blocks -> 256-d context at latent rate
  chart : 16-channel piano roll (note / LN start / hold / end x 4 lanes)
  stage1: KL autoencoder 16 -> 32 latent channels, 8x time compression
  stage2: latent diffusion, x0 prediction + L1, EMA, CFG dropout, mirror
  post  : gridify / auto-snap / remove intractable mini jacks

Training data: `.screen/neural` (391 songs).  Taps only in the current cache,
so LN channels are zero-filled; the architecture supports them.
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
AE_PATH = ROOT / "assets" / "mug_v2_ae.pt"
DIFF_PATH = ROOT / "assets" / "mug_v2_diff.pt"
CONFIG_PATH = ROOT / "assets" / "mug_v2.json"
STYLE_IDS = {"stream": 0, "speed": 1, "jack": 2, "tech": 3, "community": 4}
ATTR_DIM = 7
TIMESTEPS = 1000
LATENT = 32
DOWN = 8
AUDIO = 64


def _torch():
    import torch
    return torch


def load_index():
    path = DATA_DIR / "index_combined.json"
    if not path.exists():
        path = DATA_DIR / "index.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_song(name):
    data = np.load(DATA_DIR / f"{name}.npz")
    mel = data["mel"].astype(np.float32)
    labels = data["labels"].astype(np.float32)
    return mel, labels


def attrs(record, labels):
    onehot = [0.0] * len(STYLE_IDS)
    onehot[STYLE_IDS.get(record["style"], 0)] = 1.0
    density = float(labels.sum()) / max(1.0, labels.size)
    return np.array(onehot + [min(1.0, density * 25.0), 0.0], dtype=np.float32)


# ---------------------------------------------------------------- audio S4
class S4Encoder:
    def __init__(self):
        import torch
        import torch.nn as nn

        class SSM(nn.Module):
            def __init__(self, channels, state=16):
                super().__init__()
                self.log_a = nn.Parameter(torch.linspace(-2.0, -0.2, state))
                self.b = nn.Linear(channels, state)
                self.c = nn.Linear(state, channels)
                self.d = nn.Conv1d(channels, channels, 1)

            def forward(self, x):
                # x: (B, C, T); diagonal recurrence, bidirectional handled outside
                b, c, t = x.shape
                a = torch.exp(self.log_a)[None, :, None]
                state = torch.zeros(b, self.log_a.numel(), t, device=x.device)
                h = torch.zeros(b, self.log_a.numel(), device=x.device)
                out = []
                for index in range(t):
                    h = a[:, :, 0] * h + self.b(x[:, :, index])
                    state[:, :, index] = h
                    out.append(self.c(h))
                y = torch.stack(out, dim=2) + self.d(x)
                return y

        class Block(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.norm = nn.GroupNorm(8, channels)
                self.ssm = SSM(channels)
                self.proj = nn.Conv1d(channels, channels, 1)

            def forward(self, x):
                y = self.ssm(self.norm(x))
                y = y + torch.flip(self.ssm(torch.flip(self.norm(x), dims=[2])), dims=[2])
                return x + self.proj(y)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = nn.Conv1d(AUDIO, 128, 5, padding=2)
                self.down1 = nn.Conv1d(128, 128, 4, 2, 1)
                self.block1 = Block(128)
                self.down2 = nn.Conv1d(128, 192, 4, 2, 1)
                self.block2 = Block(192)
                self.down3 = nn.Conv1d(192, 256, 4, 2, 1)
                self.block3 = Block(256)
                self.out = nn.Conv1d(256, 128, 1)

            def forward(self, mel):
                h = torch.nn.functional.silu(self.inp(mel))
                h = self.block1(torch.nn.functional.silu(self.down1(h)))
                h = self.block2(torch.nn.functional.silu(self.down2(h)))
                h = self.block3(torch.nn.functional.silu(self.down3(h)))
                return self.out(h)

        self.net = Net()


# ---------------------------------------------------------------- piano VAE
class PianoVAE:
    def __init__(self):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = nn.Sequential(
                    nn.Conv1d(16, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
                    nn.Conv1d(64, 96, 4, 2, 1), nn.GroupNorm(8, 96), nn.SiLU(),
                    nn.Conv1d(96, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.SiLU(),
                )
                self.mean = nn.Conv1d(128, LATENT, 1)
                self.logvar = nn.Conv1d(128, LATENT, 1)
                self.dec = nn.Sequential(
                    nn.ConvTranspose1d(LATENT, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.SiLU(),
                    nn.ConvTranspose1d(128, 96, 4, 2, 1), nn.GroupNorm(8, 96), nn.SiLU(),
                    nn.ConvTranspose1d(96, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
                    nn.Conv1d(64, 16, 1),
                )

            def encode(self, x):
                h = self.enc(x)
                return self.mean(h), self.logvar(h)

            def forward(self, x):
                torch = _torch()
                mean, logvar = self.encode(x)
                z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean) if self.training else mean
                return self.dec(z), mean, logvar

        self.net = Net()


# ---------------------------------------------------------------- diffusion
def _diffusion():
    import torch
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
            self.audio_proj = nn.Conv1d(128, 64, 1)
            self.inp = nn.Conv1d(LATENT + 64, 64, 5, padding=2)
            self.cond_proj = nn.Linear(128, 64)
            self.e1 = Block(64, 64); self.e2 = Block(64, 128)
            self.mid = Block(128, 128)
            self.pool = nn.MaxPool1d(2)
            self.u2 = nn.ConvTranspose1d(128, 64, 2, stride=2)
            self.d2 = Block(64 + 128, 64)
            self.u1 = nn.ConvTranspose1d(64, 64, 2, stride=2)
            self.d1 = Block(64 + 64, 64)
            self.head = nn.Conv1d(64, LATENT, 1)

        def forward(self, x, t, audio, attr):
            import torch.nn.functional as F
            original = x.shape[-1]
            pad = (-original) % 4
            if pad:
                x = F.pad(x, (0, pad))
                audio = F.pad(audio, (0, pad))
            cond = self.time(t) + self.attr(attr)
            h = self.inp(torch.cat([x, self.audio_proj(audio)], dim=1))
            h = h + self.cond_proj(cond)[:, :, None]
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
    return betas, torch.cumprod(1.0 - betas, dim=0)


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


def _roll(labels):
    # 4 taps -> 16 channels (note / LN start / hold / end), LN channels zero
    return np.concatenate([labels, np.zeros_like(labels), np.zeros_like(labels),
                           np.zeros_like(labels)], axis=0)


def train_vae(epochs=10, crop=1024, batch=8, lr=2e-4):
    torch = _torch()
    records = load_index()
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = PianoVAE().net.to(device)
    optim = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0, device=device))
    cache = {r["file"]: load_song(r["file"]) for r in records}
    for epoch in range(1, epochs + 1):
        vae.train()
        order = [r["file"] for r in records]
        rng.shuffle(order)
        total, steps = 0.0, 0
        for start in range(0, len(order), batch):
            xs = []
            for name in order[start:start + batch]:
                _mel, labels = cache[name]
                length = min(crop, labels.shape[1])
                index = int(rng.integers(0, max(1, labels.shape[1] - length)))
                patch = _roll(labels[:, index:index + length])
                if patch.shape[1] < crop:
                    patch = np.pad(patch, ((0, 0), (0, crop - patch.shape[1])))
                xs.append(torch.from_numpy(patch).float())
            x = torch.stack(xs).to(device)
            recon, mean, logvar = vae(x)
            loss = bce(recon, x) + 1e-5 * (-0.5 * (1 + logvar - mean ** 2 - logvar.exp()).mean())
            optim.zero_grad(); loss.backward(); optim.step()
            total += float(loss.detach()); steps += 1
        print(f"VAE epoch {epoch:02d} loss={total/max(1,steps):.4f}", flush=True)
    torch.save(vae.state_dict(), AE_PATH)
    print("saved", AE_PATH)


def train_diffusion(epochs=40, crop=128, batch=8, lr=5e-5):
    torch = _torch()
    records = load_index()
    rng = np.random.default_rng(1)
    torch.manual_seed(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio = S4Encoder().net.to(device)
    vae = PianoVAE().net.to(device)
    vae.load_state_dict(torch.load(AE_PATH, map_location=device, weights_only=True))
    vae.eval()
    model = _diffusion().to(device)
    ema = EMA(model)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    _betas, alpha_bar = _schedule(device)
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
                latent = vae.encode(torch.from_numpy(_roll(patch)).float().unsqueeze(0).to(device))[0]
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
        total, steps = 0.0, 0
        for start in range(0, len(order), batch):
            names = order[start:start + batch]
            by_name = {r["file"]: r for r in records}
            items = [by_name[name] for name in names]
            x0, mel, attr = make_batch(items)
            with torch.no_grad():
                audio_feat = audio(mel)
            noise = torch.randn_like(x0)
            t = torch.randint(0, TIMESTEPS, (x0.shape[0],), device=device)
            ab = alpha_bar[t][:, None, None]
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
            pred = model(xt, t, audio_feat, attr)
            loss = torch.nn.functional.l1_loss(pred, x0)
            optim.zero_grad(); loss.backward(); optim.step(); ema.update(model)
            total += float(loss.detach()); steps += 1
        print(f"DIFF epoch {epoch:03d} loss={total/max(1,steps):.4f}", flush=True)
        if epoch % 10 == 0:
            torch.save({"model": model.state_dict(), "ema": ema.shadow.state_dict()}, DIFF_PATH)
    torch.save({"model": model.state_dict(), "ema": ema.shadow.state_dict()}, DIFF_PATH)
    CONFIG_PATH.write_text(json.dumps({"records": len(records)}, indent=1), encoding="utf-8")
    print("saved", DIFF_PATH)


# ---------------------------------------------------------------- postprocess
def gridify(rows, bpm, step=0.25):
    return {round(beat / step) * step: mask for beat, mask in rows.items()}


def remove_mini_jacks(rows, bpm, min_ms=80.0):
    """Drop later notes of same-lane repeats closer than min_ms."""
    limit = min_ms / 1000.0 * bpm / 60.0
    out = {}
    lane_last = {}
    for beat in sorted(rows):
        mask = rows[beat]
        kept = 0
        for lane in range(4):
            if not mask & (1 << lane):
                continue
            last = lane_last.get(lane)
            if last is not None and beat - last < limit:
                continue
            lane_last[lane] = beat
            kept |= 1 << lane
        if kept:
            out[beat] = kept
    return out


def auto_snap(rows, bpm, step=0.25):
    return gridify(rows, bpm, step)




def sample_record(record, steps: int = 50, cfg: float = 1.0, seed: int = 0):
    torch = _torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio = S4Encoder().net.to(device)
    vae = PianoVAE().net.to(device)
    vae.load_state_dict(torch.load(AE_PATH, map_location=device, weights_only=True))
    vae.eval()
    model = _diffusion().to(device)
    payload = torch.load(DIFF_PATH, map_location=device, weights_only=True)
    model.load_state_dict(payload.get("ema", payload["model"]))
    model.eval()
    mel, labels = load_song(record["file"])
    mel_t = torch.from_numpy(mel).float().unsqueeze(0).to(device)
    attr = torch.from_numpy(attrs(record, labels)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        audio_feat = audio(mel_t)
    length = audio_feat.shape[-1]
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(1, LATENT, length, device=device, generator=g)
    _betas, alpha_bar = _schedule(device)
    step_list = np.linspace(TIMESTEPS - 1, 0, steps).astype(int)
    with torch.no_grad():
        for i, t in enumerate(step_list):
            tt = torch.tensor([t], device=device)
            x0 = model(x, tt, audio_feat, attr)
            if cfg != 1.0:
                x0_u = model(x, tt, torch.zeros_like(audio_feat), torch.zeros_like(attr))
                x0 = x0_u + cfg * (x0 - x0_u)
            ab = alpha_bar[t]
            if i + 1 < len(step_list):
                t_next = step_list[i + 1]
                ab_next = alpha_bar[t_next]
                x = ab_next.sqrt() * x0 + ((1 - ab_next) / (1 - ab)).sqrt() * (x - ab.sqrt() * x0)
            else:
                x = x0
        recon = vae.dec(x)
    prob = torch.sigmoid(recon)[0].cpu().numpy()
    return prob
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["vae", "diff"], default="vae", nargs="?")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    if args.command == "vae":
        train_vae(epochs=args.epochs)
    else:
        train_diffusion(epochs=args.epochs)
