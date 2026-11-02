"""Exploratory: how do raw tempo candidates compare with chart BPMs?

Prints, per song, the raw (no-prior) ACF comb candidates and how several
octave-resolution rules would classify them.  Used to tune the tempo prior;
not needed at runtime.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.audio import load_audio
from core.bpm import _autocorrelation, _acf_at, onset_envelopes

from tests.bench_charts import chart_audio, chart_bpms, classify  # noqa: E402


def raw_curve(acf: np.ndarray, hop: float, lo=50.0, hi=280.0):
    bpms = np.arange(lo, hi + 1e-9, 0.1)
    score = np.zeros_like(bpms)
    for h, w in ((1, 1.0), (2, 0.5), (3, 0.33), (4, 0.25)):
        lags = 60.0 * h / (bpms * hop)
        score += w * np.array([_acf_at(acf, lag) for lag in lags])
    return bpms, score


def pick_raw(bpms, score, prior_center=None, prior_oct=None, top=6):
    s = score.copy()
    if prior_center and prior_oct:
        s = s * np.exp(-0.5 * (np.log2(bpms / prior_center) / prior_oct) ** 2)
    order = np.argsort(s)[::-1]
    picked = []
    for idx in order:
        b = float(bpms[idx])
        if all(abs(b - p[0]) > 0.03 * b for p in picked):
            picked.append((b, float(s[idx])))
        if len(picked) >= top:
            break
    return picked


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    mc_files = sorted(ROOT.rglob("*.mc"))
    seen = set()
    songs = []
    for mc in mc_files:
        audio = chart_audio(mc)
        if audio is None or audio in seen:
            continue
        seen.add(audio)
        songs.append((audio, chart_bpms(mc)))
        if len(songs) >= limit:
            break

    rules = {
        "raw": dict(),
        "p140_0.9": dict(prior_center=140, prior_oct=0.9),
        "p160_0.6": dict(prior_center=160, prior_oct=0.6),
        "p165_0.55": dict(prior_center=165, prior_oct=0.55),
        "p170_0.6": dict(prior_center=170, prior_oct=0.6),
        "p175_0.5": dict(prior_center=175, prior_oct=0.5),
    }
    results = {k: [] for k in rules}
    for audio, ref in songs:
        y, sr = load_audio(audio, duration=120)
        env, low, hop, t0 = onset_envelopes(y, sr)
        bpms, score = raw_curve(_autocorrelation(env), hop)
        print(f"\n{audio.name[:50]}  ref={[round(b,1) for b in ref[:4]]}")
        raw = pick_raw(bpms, score, top=6)
        print("  raw:", "  ".join(f"{b:.1f}({s:.2f})" for b, s in raw))
        for name, kwargs in rules.items():
            picked = pick_raw(bpms, score, **kwargs)
            results[name].append(classify(picked[0][0], ref))

    print("\n" + "=" * 72)
    for name in rules:
        c = Counter(results[name])
        n = len(results[name])
        print(f"{name:10} exact={c['exact']:3}/{n}  half={c['half']:2}  double={c['double']:2}  "
              f"miss={c['miss']:2}  exact_rate={c['exact']/n*100:5.1f}%")


if __name__ == "__main__":
    main()
