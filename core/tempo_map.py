"""Tempo map from a short-window BPM curve.

A 12 s window sliding every 2 s yields a per-window BPM estimate.  The curve is
smoothed and change markers are placed where the *local median* shifts and the
new level persists; each marker starts a segment whose BPM/offset/beat grid is
refined on that slice.  The markers double as jump points in the GUI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bpm import (
    _autocorrelation, _grid_score, _prior, onset_envelopes, refine_beat_grid,
    smooth_envelopes, tempo_candidates,
)

WINDOW_SEC = 12.0
WINDOW_HOP_SEC = 2.0
MAX_SEGMENTS = 16
MIN_MARKER_GAP = 6.0

PRESETS = {
    "low": {"window": 12.0, "hop": 2.0, "k": 3, "top": 5,
            "prior_oct": 0.6, "threshold": 0.05, "confirm": 3, "gap": 6.0},
    "high": {"window": 12.0, "hop": 2.0, "k": 3, "top": 5,
             "prior_oct": 0.6, "threshold": 0.05, "confirm": 3, "gap": 6.0,
             "validate": True},
    "ultra": {"window": 3.0, "hop": 0.5, "k": 2, "top": 10,
              "prior_oct": 1.2, "threshold": 0.02, "confirm": 2, "gap": 0.5,
              "validate": False, "no_refine": True},
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
        if i1 - i0 < 96:
            continue
        raw = env_raw[i0:i1]
        grid = env_grid[i0:i1]
        low = low_grid[i0:i1]
        t0l = t0 + i0 * hop - start
        acf = _autocorrelation(raw)
        cands, _ = tempo_candidates(acf, hop, 50, 320, top=cfg["top"],
                                    prior_center=global_bpm,
                                    prior_octaves=cfg["prior_oct"])
        total = float(grid.sum() + 0.35 * low.sum()) + 1e-9
        best = (-1.0, global_bpm)
        for cand in cands:
            offsets = np.linspace(0.0, 60.0 / cand.bpm, 40, endpoint=False)
            mean, _off, captured, density = _grid_score(
                grid, low, hop, cand.bpm, offsets, win, t0l)
            if mean <= 0:
                continue
            cov = captured / total
            rank = cov * (0.5 + 0.5 * density) * _prior(cand.bpm, global_bpm, cfg["prior_oct"])
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


def detect_tempo_map(y: np.ndarray, sr: int, global_bpm: float,
                     precision: str = "low") -> dict:
    cfg = PRESETS.get(precision, PRESETS["low"])
    duration = float(y.size) / float(sr)
    empty = {"has_changes": False, "segments": [], "markers": [], "bpm": global_bpm}
    if duration < 15.0 or global_bpm <= 0:
        return empty

    env_raw, env_low_raw, hop, t0 = onset_envelopes(y, sr)
    env_grid, low_grid = smooth_envelopes(env_raw, env_low_raw, hop)
    curve = _window_curve(env_raw, env_grid, low_grid, hop, t0, duration, global_bpm, cfg)
    if len(curve) < 4:
        return empty

    starts = np.array([c[0] for c in curve])
    wins = np.array([c[1] for c in curve])
    smooth = _median_filter([c[2] for c in curve], radius=1)

    # change points: compare median of the previous 10 s with the next 10 s
    k = cfg["k"]
    markers: list[float] = []
    i = k
    while i <= len(smooth) - k:
        left = float(np.median(smooth[i - k:i]))
        right = float(np.median(smooth[i:i + k]))
        if abs(float(np.log2(right / left))) > cfg["threshold"]:
            confirm = smooth[i:i + cfg["confirm"]]
            if len(confirm) >= cfg["confirm"] and all(
                    abs(float(np.log2(v / right))) < 0.04 for v in confirm):
                t = float(starts[i])
                if not markers or t - markers[-1] > cfg["gap"]:
                    markers.append(t)
                i += k
                continue
        i += 1
    markers = markers[:MAX_SEGMENTS]

    bounds = [0.0] + markers + [duration]
    segments: list[TempoSegment] = []
    for idx in range(len(bounds) - 1):
        start, end = bounds[idx], bounds[idx + 1]
        if end - start < 1.0:
            continue
        win = end - start
        in_seg = [smooth[j] for j in range(len(starts))
                  if start - 0.01 <= starts[j] < end - 0.01]
        seed = float(np.median(in_seg)) if in_seg else global_bpm
        i0 = max(0, int(round((start - t0) / hop)))
        i1 = min(env_grid.size, int(round((end - t0) / hop)))
        if i1 - i0 < 24 or cfg.get("no_refine"):
            segments.append(TempoSegment(start * 1000, end * 1000, seed))
            continue
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
        segments.append(TempoSegment(start * 1000.0, end * 1000.0, bpm, conf, beats))

    # merge adjacent segments that are basically the same tempo
    merged: list[TempoSegment] = []
    for seg in segments:
        if merged and abs(float(np.log2(seg.bpm / merged[-1].bpm))) < 0.02:
            prev = merged[-1]
            prev.end_ms = seg.end_ms
            prev.beats_ms.extend(seg.beats_ms)
        else:
            merged.append(seg)
    segments = merged

    # merge very short segments into the neighbour with the closest tempo
    changed = (not cfg.get("no_refine"))
    while changed and len(segments) > 1:
        changed = False
        for idx, seg in enumerate(segments):
            if idx == 0 or idx == len(segments) - 1:
                continue
            if seg.end_ms - seg.start_ms >= 10000.0:
                continue
            prevs = segments[idx - 1] if idx > 0 else None
            nexts = segments[idx + 1] if idx + 1 < len(segments) else None
            if prevs and (not nexts or abs(np.log2(prevs.bpm / seg.bpm)) <=
                          abs(np.log2(nexts.bpm / seg.bpm))):
                prevs.end_ms = seg.end_ms
                prevs.beats_ms.extend(seg.beats_ms)
            elif nexts:
                nexts.start_ms = seg.start_ms
                nexts.beats_ms = seg.beats_ms + nexts.beats_ms
            else:
                continue
            segments.pop(idx)
            changed = True
            break
    for i in range(len(segments)):
        segments[i].start_ms = 0.0 if i == 0 else segments[i - 1].end_ms

    # high precision: keep only segments with enough window support
    if cfg.get("validate") and len(segments) > 1:
        support = []
        for seg in segments:
            s0, e0 = seg.start_ms / 1000.0, seg.end_ms / 1000.0
            vals = [bp for (st, _w, bp) in curve if s0 - 0.01 <= st < e0 - 0.01]
            sup = sum(1 for bp in vals if abs(float(np.log2(bp / seg.bpm))) < 0.045)
            support.append((sup, len(vals)))
        kept: list[TempoSegment] = []
        pending: TempoSegment | None = None
        for seg, (sup, total) in zip(segments, support):
            long_enough = (seg.end_ms - seg.start_ms) >= 15000.0
            if sup >= 2 or total == 0 or long_enough:
                if pending is not None:
                    seg.start_ms = pending.start_ms
                    seg.beats_ms = pending.beats_ms + seg.beats_ms
                    pending = None
                kept.append(seg)
            else:
                if kept:
                    kept[-1].end_ms = seg.end_ms
                    kept[-1].beats_ms.extend(seg.beats_ms)
                else:
                    pending = seg
        if pending is not None and kept:
            kept[0].start_ms = pending.start_ms
            kept[0].beats_ms = pending.beats_ms + kept[0].beats_ms
        segments = kept or segments

    # final merge of adjacent segments that ended up at the same tempo
    merged_final: list[TempoSegment] = []
    if cfg.get("no_refine"):
        merged_final = segments
    for seg in ([] if cfg.get("no_refine") else segments):
        if merged_final and abs(float(np.log2(seg.bpm / merged_final[-1].bpm))) < 0.04:
            merged_final[-1].end_ms = seg.end_ms
            merged_final[-1].beats_ms.extend(seg.beats_ms)
        else:
            merged_final.append(seg)
    segments = merged_final

    has_changes = False
    for a, b in zip(segments, segments[1:]):
        if abs(float(np.log2(max(b.bpm, 1e-9) / max(a.bpm, 1e-9)))) > 0.04:
            has_changes = True
            break
    dominant = max(segments, key=lambda s: s.end_ms - s.start_ms)
    markers_out = [{"time_ms": round(s.start_ms, 1), "bpm": round(s.bpm, 3)}
                   for s in segments]
    return {
        "has_changes": has_changes,
        "segments": segments,
        "markers": markers_out,
        "bpm": dominant.bpm,
    }
