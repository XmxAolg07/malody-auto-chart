"""Tempo-change detection by block-wise estimation.

The song is split into 3-4 blocks; each block gets an independent BPM estimate,
octave-related results are folded to the global level (a half/double section is
not a tempo change), and a change is only reported when two adjacent blocks are
stable on both sides and differ by a non-octave ratio.  That keeps
constant-tempo songs (and octave-prone estimates) at ``has_changes == False``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bpm import estimate_rhythm, onset_envelopes, refine_beat_grid, smooth_envelopes

MIN_BLOCK_SEC = 20.0


@dataclass
class TempoSegment:
    start_ms: float
    end_ms: float
    bpm: float
    confidence: float = 0.0
    beats_ms: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "start_ms": round(float(self.start_ms), 1),
            "end_ms": round(float(self.end_ms), 1),
            "bpm": round(float(self.bpm), 3),
            "confidence": round(float(self.confidence), 4),
            "beats_ms": [round(float(t), 1) for t in self.beats_ms],
        }


SIMPLE_RATIOS = (0.5, 2.0 / 3.0, 0.75, 1.0, 4.0 / 3.0, 1.5, 2.0)


def _octave_norm(bpm: float, ref: float) -> float:
    """Fold octave / dotted-tempo estimate errors back to the global level."""
    if bpm <= 0 or ref <= 0:
        return bpm
    lr = float(np.log2(bpm / ref))
    for ratio in SIMPLE_RATIOS:
        if abs(lr - float(np.log2(ratio))) < 0.05:
            return ref * ratio
    return bpm


def _stable(a: float, b: float, tol: float = 0.05) -> bool:
    if a <= 0 or b <= 0:
        return False
    return abs(float(np.log2(a / b))) < tol


def detect_tempo_map(y: np.ndarray, sr: int, global_bpm: float) -> dict:
    duration = float(y.size) / float(sr)
    if duration < 45.0 or global_bpm <= 0:
        return {"has_changes": False, "segments": [], "bpm": global_bpm}

    n_blocks = 4 if duration >= 90.0 else 3
    block = duration / n_blocks
    bpms: list[float] = []
    for i in range(n_blocks):
        start = i * block
        end = duration if i == n_blocks - 1 else (i + 1) * block
        segment = y[int(start * sr):int(end * sr)]
        if segment.size < sr * MIN_BLOCK_SEC:
            bpms.append(0.0)
            continue
        try:
            r = estimate_rhythm(segment, sr)
            bpms.append(r.bpm if r.bpm > 0 else 0.0)
        except Exception:
            bpms.append(0.0)

    # fold octave-related block results to the global metrical level
    norm = [_octave_norm(b, global_bpm) if b > 0 else 0.0 for b in bpms]
    changes: list[tuple[int, float]] = []
    for i in range(1, n_blocks):
        a, b = norm[i - 1], norm[i]
        if a <= 0 or b <= 0:
            continue
        if abs(float(np.log2(b / a))) <= 0.12:
            continue
        left_ok = i - 1 == 0 or _stable(norm[i - 1], norm[i - 2])
        right_ok = i == n_blocks - 1 or _stable(norm[i + 1], norm[i])
        if left_ok and right_ok:
            if i < 2 or n_blocks - i < 2:
                continue
            if (i * block) >= MIN_BLOCK_SEC and ((n_blocks - i) * block) >= MIN_BLOCK_SEC:
                changes.append((i, b))
    if not changes:
        return {"has_changes": False, "segments": [], "bpm": global_bpm}

    env_raw, env_low_raw, hop, t0 = onset_envelopes(y, sr)
    env_grid, low_grid = smooth_envelopes(env_raw, env_low_raw, hop)

    boundaries = [0.0] + [i * block for i, _b in changes] + [duration]
    segments: list[TempoSegment] = []
    for k in range(len(boundaries) - 1):
        start, end = boundaries[k], boundaries[k + 1]
        if end - start < 2.0:
            continue
        b0 = 0 if k == 0 else changes[k - 1][0]
        b1 = n_blocks if k == len(boundaries) - 2 else changes[k][0]
        vals = [norm[j] for j in range(b0, b1) if norm[j] > 0]
        seed = float(np.median(vals)) if vals else global_bpm
        segments.append(_refine_segment(env_grid, low_grid, hop, t0, start, end, seed))

    if len(segments) < 2:
        return {"has_changes": False, "segments": [], "bpm": global_bpm}

    real = []
    for seg in segments:
        ratio = float(np.log2(max(seg.bpm, 1e-9) / global_bpm))
        k = round(ratio)
        if abs(ratio) < 0.05 or abs(ratio - k) < 0.07:
            seg.bpm = _octave_norm(seg.bpm, global_bpm)
        else:
            real.append(seg)
    if not real:
        return {"has_changes": False, "segments": [], "bpm": global_bpm}

    dominant = max(segments, key=lambda s: s.end_ms - s.start_ms)
    return {"has_changes": True, "segments": segments, "bpm": dominant.bpm}


def _refine_segment(env_grid, low_grid, hop, t0, start, end, seed_bpm) -> TempoSegment:
    dur = end - start
    i0 = max(0, int(round((start - t0) / hop)))
    i1 = min(env_grid.size, int(round((end - t0) / hop)))
    if i1 - i0 < 48:
        return TempoSegment(start * 1000, end * 1000, seed_bpm)
    grid = env_grid[i0:i1]
    low = low_grid[i0:i1]
    t0_local = t0 + i0 * hop - start
    bpm, offset, _strength, _score, captured, density = refine_beat_grid(
        grid, low, hop, seed_bpm, dur, t0_local)
    total = float(grid.sum() + 0.35 * low.sum()) + 1e-9
    coverage = captured / total
    conf = float(np.clip(0.55 * coverage / 0.6 + 0.45 * density, 0.0, 1.0))
    period = 60.0 / bpm
    rel = offset % period
    beats = []
    while rel < dur:
        beats.append((start + rel) * 1000.0)
        rel += period
    return TempoSegment(start * 1000.0, end * 1000.0, bpm, conf, beats)
