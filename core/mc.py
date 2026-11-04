"""Read the exact tempo list from a Malody .mc chart.

A .mc file is JSON with ``time`` entries: ``{"beat": [measure, num, den],
"bpm": 150.0}``.  Beat position is ``measure + num/den``; the wall-clock start
of each tempo section is accumulated from the previous BPM.  That gives the
exact tempo map the charter used, without guessing from audio.
"""

from __future__ import annotations

import json
from pathlib import Path


def _beat_position(beat) -> float:
    if not beat:
        return 0.0
    if len(beat) == 1:
        return float(beat[0])
    return float(beat[0]) + float(beat[1]) / float(beat[2] or 1)


def tempo_segments_from_mc(path: str | Path, duration_ms: float | None = None) -> list[dict]:
    """Return [{'start_ms','end_ms','bpm','confidence','beats_ms'}] from a .mc."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    times = data.get("time") or []
    entries = []
    for item in times:
        try:
            bpm = float(item["bpm"])
        except (KeyError, TypeError, ValueError):
            continue
        if bpm <= 0:
            continue
        entries.append((_beat_position(item.get("beat", [0])), bpm))
    entries.sort(key=lambda e: e[0])
    if not entries:
        return []

    starts: list[tuple[float, float]] = []   # (time_sec, bpm)
    time_sec = 0.0
    prev_beat, prev_bpm = entries[0]
    starts.append((0.0, prev_bpm))
    for beat, bpm in entries[1:]:
        time_sec += (beat - prev_beat) * 60.0 / prev_bpm
        starts.append((time_sec, bpm))
        prev_beat, prev_bpm = beat, bpm

    total_ms = float(duration_ms) if duration_ms else (starts[-1][0] * 1000.0 + 1.0)
    segments = []
    for i, (start_sec, bpm) in enumerate(starts):
        end_sec = starts[i + 1][0] if i + 1 < len(starts) else total_ms / 1000.0
        if end_sec <= start_sec:
            continue
        period = 60.0 / bpm
        beats = []
        t = start_sec
        while t < end_sec - 1e-6:
            beats.append(round(t * 1000.0, 1))
            t += period
        segments.append({
            "start_ms": round(start_sec * 1000.0, 1),
            "end_ms": round(end_sec * 1000.0, 1),
            "bpm": round(bpm, 3),
            "confidence": 1.0,
            "beats_ms": beats,
        })
    return segments
