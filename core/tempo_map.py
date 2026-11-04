"""Tempo map from a windowed BPM curve (low / high precision).

``low`` uses a 12 s window / 2 s hop and produces segment-level changes.
``high`` first runs ``low`` (so constant-tempo songs are never shredded by
short-window noise), then re-scans *inside each low segment* with a 3 s window
/ 0.5 s hop and splits only on sustained, validated changes.  The final
segments get an exact BPM/offset/beat grid from their own audio slice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bpm import (
    _autocorrelation, _grid_score, _prior, onset_envelopes, refine_beat_grid,
    smooth_envelopes, tempo_candidates,
)

PRESETS = {
    "low": {
        "window": 12.0, "hop": 2.0, "k": 3, "top": 5,
        "prior_oct": 0.6, "threshold": 0.05, "confirm": 3, "gap": 6.0,
        "min_seg_ms": 10000.0, "merge_ratio": 0.04, "max_segments": 16,
        "refine_min_sec": 2.0,
    },
    "high": {
        "window": 3.0, "hop": 0.5, "k": 2, "top": 8,
        "prior_oct": 1.2, "threshold": 0.05, "confirm": 2, "gap": 0.75,
        "min_seg_ms": 1200.0, "merge_ratio": 0.02, "max_segments": 120,
        "refine_min_sec": 1.5,
    },
}


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


def _window_curve(env_raw, env_grid, low_grid, hop, t0, duration, global_bpm, cfg):
    window_sec = cfg["window"]
    starts = np.arange(0.0, max(duration - window_sec * 0.5, 0.1), cfg["hop"])
    curve = []
    for start in starts:
        win = min(window_sec, duration - start)
        if win < window_sec * 0.7:
            continue
        i0 = max(0, int(round((start - t0) / hop)))
        i1 = min(env_raw.size, int(round((start + win - t0) / hop)))
        if i1 - i0 < 48:
            continue
        raw = env_raw[i0:i1]
        grid = env_grid[i0:i1]
        low = low_grid[i0:i1]
        t0l = t0 + i0 * hop - start
        acf = _autocorrelation(raw)
        cands, _ = tempo_candidates(acf, hop, 50, 340, top=cfg["top"],
                                    prior_center=global_bpm,
                                    prior_octaves=cfg["prior_oct"])
        total = float(grid.sum() + 0.35 * low.sum()) + 1e-9
        best = (-1.0, global_bpm)
        for cand in cands:
            offsets = np.linspace(0.0, 60.0 / cand.bpm, 32, endpoint=False)
            mean, _off, captured, density = _grid_score(
                grid, low, hop, cand.bpm, offsets, win, t0l)
            if mean <= 0:
                continue
            cov = captured / total
            rank = cov * (0.5 + 0.5 * density) * _prior(cand.bpm, global_bpm,
                                                        cfg["prior_oct"])
            if rank > best[0]:
                best = (rank, cand.bpm)
        curve.append((float(start), float(win), float(best[1])))
    return curve


def _median_filter(values, radius=1):
    out = list(values)
    for i in range(len(values)):
        lo, hi = max(0, i - radius), min(len(values), i + radius + 1)
        out[i] = float(np.median(values[lo:hi]))
    return out


def _refine(env_grid, low_grid, hop, t0, start, end, seed, min_sec):
    win = end - start
    if win < min_sec:
        return TempoSegment(start * 1000, end * 1000, seed)
    i0 = max(0, int(round((start - t0) / hop)))
    i1 = min(env_grid.size, int(round((end - t0) / hop)))
    if i1 - i0 < 48:
        return TempoSegment(start * 1000, end * 1000, seed)
    grid = env_grid[i0:i1]
    low = low_grid[i0:i1]
    t0l = t0 + i0 * hop - start
    bpm, offset, _strength, _score, captured, density = refine_beat_grid(
        grid, low, hop, seed, win, t0l)
    total = float(grid.sum() + 0.35 * low.sum()) + 1e-9
    coverage = captured / total
    conf = float(np.clip(0.55 * coverage / 0.6 + 0.45 * density, 0.0, 1.0))
    period = 60.0 / bpm
    rel = offset % period
    beats = []
    while rel < win:
        beats.append((start + rel) * 1000.0)
        rel += period
    return TempoSegment(start * 1000.0, end * 1000.0, bpm, conf, beats)


def _find_changes(starts, smooth, cfg):
    k = cfg["k"]
    markers: list[float] = []
    i = k
    while i <= len(smooth) - k:
        left = float(np.median(smooth[i - k:i]))
        right = float(np.median(smooth[i:i + k]))
        if abs(float(np.log2(right / left))) > cfg["threshold"]:
            confirm = smooth[i:i + cfg["confirm"]]
            if len(confirm) >= cfg["confirm"] and all(
                    abs(float(np.log2(v / right))) < 0.05 for v in confirm):
                t = float(starts[i])
                if not markers or t - markers[-1] > cfg["gap"]:
                    markers.append(t)
                i += k
                continue
        i += 1
    return markers[: cfg["max_segments"]]


def _merge(segments, cfg):
    merged: list[TempoSegment] = []
    for seg in segments:
        if merged and abs(float(np.log2(seg.bpm / merged[-1].bpm))) < cfg["merge_ratio"]:
            merged[-1].end_ms = seg.end_ms
            merged[-1].beats_ms.extend(seg.beats_ms)
        else:
            merged.append(seg)
    segments = merged
    changed = True
    while changed and len(segments) > 1:
        changed = False
        for idx, seg in enumerate(segments):
            if seg.end_ms - seg.start_ms >= cfg["min_seg_ms"]:
                continue
            if idx == 0 or idx == len(segments) - 1:
                continue
            prevs, nexts = segments[idx - 1], segments[idx + 1]
            if abs(np.log2(prevs.bpm / seg.bpm)) <= abs(np.log2(nexts.bpm / seg.bpm)):
                prevs.end_ms = seg.end_ms
                prevs.beats_ms.extend(seg.beats_ms)
            else:
                nexts.start_ms = seg.start_ms
                nexts.beats_ms = seg.beats_ms + nexts.beats_ms
            segments.pop(idx)
            changed = True
            break
    return segments[: cfg["max_segments"]]


def _summarize(segments, global_bpm):
    if not segments:
        return {"has_changes": False, "segments": [], "markers": [], "bpm": global_bpm}
    for i in range(len(segments)):
        if i > 0:
            segments[i].start_ms = segments[i - 1].end_ms
    has_changes = False
    for a, b in zip(segments, segments[1:]):
        if abs(float(np.log2(max(b.bpm, 1e-9) / max(a.bpm, 1e-9)))) > 0.02:
            has_changes = True
            break
    dominant = max(segments, key=lambda s: s.end_ms - s.start_ms)
    markers = [{"time_ms": round(s.start_ms, 1), "bpm": round(s.bpm, 3)}
               for s in segments]
    return {"has_changes": has_changes, "segments": segments,
            "markers": markers, "bpm": dominant.bpm}


def _detect_high(y: np.ndarray, sr: int, global_bpm: float) -> dict:
    """Conservative high precision: start from low and only merge, never split."""
    low = detect_tempo_map(y, sr, global_bpm, "low")
    segments = low["segments"]
    if len(segments) < 2:
        return low

    merged: list[TempoSegment] = []
    for seg in segments:
        if merged and abs(float(np.log2(seg.bpm / merged[-1].bpm))) < 0.05:
            merged[-1].end_ms = seg.end_ms
            merged[-1].beats_ms.extend(seg.beats_ms)
        else:
            merged.append(seg)

    changed = True
    while changed and len(merged) > 1:
        changed = False
        for idx, seg in enumerate(merged):
            if idx == 0 or idx == len(merged) - 1:
                continue
            if seg.end_ms - seg.start_ms >= 4000.0:
                continue
            prevs, nexts = merged[idx - 1], merged[idx + 1]
            if abs(np.log2(prevs.bpm / seg.bpm)) <= abs(np.log2(nexts.bpm / seg.bpm)):
                prevs.end_ms = seg.end_ms
                prevs.beats_ms.extend(seg.beats_ms)
            else:
                nexts.start_ms = seg.start_ms
                nexts.beats_ms = seg.beats_ms + nexts.beats_ms
            merged.pop(idx)
            changed = True
            break

    return _summarize(merged, global_bpm)


def detect_tempo_map(y: np.ndarray, sr: int, global_bpm: float,
                     precision: str = "low") -> dict:
    if precision == "high":
        return _detect_high(y, sr, global_bpm)
    cfg = PRESETS["low"]
    duration = float(y.size) / float(sr)
    empty = {"has_changes": False, "segments": [], "markers": [], "bpm": global_bpm}
    if duration < 15.0 or global_bpm <= 0:
        return empty
    env_raw, env_low_raw, hop, t0 = onset_envelopes(y, sr)
    env_grid, low_grid = smooth_envelopes(env_raw, env_low_raw, hop)
    curve = _window_curve(env_raw, env_grid, low_grid, hop, t0, duration,
                          global_bpm, cfg)
    if len(curve) < 6:
        return empty
    starts = np.array([c[0] for c in curve])
    smooth = _median_filter([c[2] for c in curve], radius=1)
    markers = _find_changes(starts, smooth, cfg)
    bounds = [0.0] + markers + [duration]
    segments: list[TempoSegment] = []
    for idx in range(len(bounds) - 1):
        start, end = bounds[idx], bounds[idx + 1]
        if end - start < 0.2:
            continue
        in_seg = [smooth[j] for j in range(len(starts))
                  if start - 0.01 <= starts[j] < end - 0.01]
        seed = float(np.median(in_seg)) if in_seg else global_bpm
        segments.append(_refine(env_grid, low_grid, hop, t0, start, end, seed,
                                cfg["refine_min_sec"]))
    segments = _merge(segments, cfg)
    return _summarize(segments, global_bpm)
