"""Synthetic validation for BPM detection + waveform extraction.

These signals have a known ground truth, so they isolate algorithm errors from
"is this track really 174 BPM?" questions.

Run:
    python tests/test_synthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.bpm import estimate_rhythm          # noqa: E402
from core.waveform import waveform_peaks      # noqa: E402

SR = 44100


def _kick(sr: int, dur: float = 0.22) -> np.ndarray:
    n = int(sr * dur)
    t = np.arange(n) / sr
    freq = 140.0 * np.exp(-t * 28.0) + 48.0
    phase = 2 * np.pi * np.cumsum(freq) / sr
    env = np.exp(-t * 16.0)
    return (np.sin(phase) * env).astype(np.float32)


def _snare(sr: int, dur: float = 0.18, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * dur)
    t = np.arange(n) / sr
    noise = rng.standard_normal(n).astype(np.float32)
    noise = np.diff(noise, prepend=0.0)  # crude high-pass
    env = np.exp(-t * 22.0)
    tone = np.sin(2 * np.pi * 190.0 * t) * np.exp(-t * 30.0)
    return ((0.6 * noise + 0.3 * tone) * env).astype(np.float32)


def _hat(sr: int, dur: float = 0.045, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * dur)
    t = np.arange(n) / sr
    noise = rng.standard_normal(n).astype(np.float32)
    noise = np.diff(noise, n=2, prepend=[0.0, 0.0])
    return (noise * np.exp(-t * 90.0) * 0.35).astype(np.float32)


def _add(y: np.ndarray, sample: np.ndarray, at: float, sr: int, gain: float = 1.0) -> None:
    i = int(round(at * sr))
    if i < 0 or i >= y.size:
        return
    end = min(y.size, i + sample.size)
    y[i:end] += gain * sample[: end - i]


def synth_track(
    bpm: float,
    seconds: float = 40.0,
    offset: float = 0.0,
    sr: int = SR,
    subdivision_hats: bool = True,
) -> np.ndarray:
    """4/4 pattern: kick on 1/3, snare on 2/4, hats on eighths, light bass."""
    y = np.zeros(int(seconds * sr), dtype=np.float32)
    beat = 60.0 / bpm
    kick, snare, hat = _kick(sr), _snare(sr), _hat(sr)

    beat_index = 0
    t = offset
    while t < seconds:
        if beat_index % 4 in (0, 2):
            _add(y, kick, t, sr, 0.9)
        else:
            _add(y, snare, t, sr, 0.55)
        if subdivision_hats:
            _add(y, hat, t + beat * 0.5, sr, 0.5)
        # simple bass note on the beat keeps tonal content realistic
        bass = np.sin(2 * np.pi * 55.0 * np.arange(int(0.12 * sr)) / sr)
        _add(y, (bass * np.exp(-np.arange(bass.size) / (0.03 * sr)) * 0.25).astype(np.float32), t, sr)
        beat_index += 1
        t += beat

    y += np.random.default_rng(7).standard_normal(y.size).astype(np.float32) * 0.002
    peak = float(np.abs(y).max()) or 1.0
    return (y / peak * 0.9).astype(np.float32)


def _circular_error_ms(a: float, b: float, period_ms: float) -> float:
    d = abs(a - b) % period_ms
    return min(d, period_ms - d)


def main() -> int:
    # 70 BPM with one onset per beat is octave-ambiguous (see below); start at 90.
    bpms = [90, 100, 120, 128, 145, 160, 174, 190, 200, 222, 240]
    failures = 0

    print("== strict: quarter-note pulse only (unambiguous)")
    print(f"{'true':>6} {'detected':>9} {'ratio':>7} {'conf':>6} {'off err':>8}  verdict")
    for bpm in bpms:
        offset = 0.30
        y = synth_track(bpm, seconds=45.0, offset=offset, subdivision_hats=False)
        res = estimate_rhythm(y, SR)
        period_ms = 60_000.0 / bpm
        ratio = res.bpm / bpm
        close = abs(ratio - 1.0) < 0.02
        off_err = _circular_error_ms(res.offset_ms, offset * 1000.0, period_ms) if close else float("nan")
        ok = close and off_err < 35.0
        if not ok:
            failures += 1
        print(f"{bpm:6.0f} {res.bpm:9.2f} {ratio:7.3f} {res.confidence:6.2f} "
              f"{off_err:8.1f}  {'OK' if ok else 'FAIL'}")

    print("\n== slow pulse: default may pick the double, but the half must be offered")
    slow = synth_track(70, seconds=45.0, offset=0.30, subdivision_hats=False)
    slow_res = estimate_rhythm(slow, SR)
    offered = [round(c.bpm, 2) for c in slow_res.candidates]
    has_half = any(abs(b - 70.0) < 1.5 for b in offered)
    print(f"   default={slow_res.bpm:.2f}  candidates={offered[:5]}  half offered={has_half}")
    if not has_half:
        failures += 1

    print("\n== 8th-note hats: octave ambiguity is expected (informational)")
    exact = 0
    total = 0
    for bpm in bpms:
        y = synth_track(bpm, seconds=45.0, offset=0.30, subdivision_hats=True)
        res = estimate_rhythm(y, SR)
        ratio = res.bpm / bpm
        total += 1
        if abs(ratio - 1.0) < 0.02:
            exact += 1
            label = "exact"
        elif abs(ratio - 2.0) < 0.02:
            label = "x2"
        elif abs(ratio - 0.5) < 0.02:
            label = "x0.5"
        else:
            label = "other"
        print(f"{bpm:6.0f} -> {res.bpm:9.2f}  {label}")
    print(f"   exact on hat-heavy patterns: {exact}/{total}")

    # waveform peaks sanity
    y = synth_track(128, seconds=5.0)
    peaks = waveform_peaks(y, 800)
    assert len(peaks["max"]) == 800 and len(peaks["min"]) == 800
    assert max(peaks["max"]) <= 1.0 and min(peaks["min"]) >= -1.0
    assert max(peaks["rms"]) <= 1.0
    print("\nwaveform peaks: OK (800 columns, normalised)")

    tiny = estimate_rhythm(np.zeros(100, dtype=np.float32), SR)
    print(f"tiny input: bpm={tiny.bpm} (no crash)")

    print(f"\n{failures} strict failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
