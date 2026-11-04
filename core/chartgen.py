"""Generate 4K Malody charts in four key-pattern styles.

Styles follow the four regular patterns from the Malody 4K tutorial series:

* ``stream``  switching / stream: single stream with jumps and occasional hands
* ``speed``   dense single stream (modern speed), sparse jumps
* ``jack``    repeated-column stacks, jump/hand chords
* ``tech``    mixed phrases: stream, jack, offbeat, trill, hands

The generator works in beat space (Malody ``.mc`` uses measures + fractions)
and turns the analysed tempo map into the chart's ``time`` array, so variable
BPM songs keep their tempo changes.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

LANES = (0, 1, 2, 3)

STYLES = {
    "stream": "切换 / Stream",
    "speed": "大乱 / Speed",
    "jack": "叠键 / Jack",
    "tech": "技巧 / Technical",
}

DESCRIPTIONS = {
    "stream": "单乱为主，每 4 拍加双押，偶尔三押；相邻行不重复同列。",
    "speed": "高速单乱，1/4 拍密集排列，稀疏双押，几乎无三押。",
    "jack": "同列连续 2-4 键，成组切换；8 拍双押叠、16 拍三押叠。",
    "tech": "8 拍乐句循环：乱键、叠键、反拍、交互轮指混合。",
}


def _beat_list(beat: float, den: int = 16) -> list[int]:
    measure = int(beat)
    num = int(round((beat - measure) * den))
    if num >= den:
        measure += 1
        num = 0
    return [measure, num, den]


def tempo_timeline(rhythm: dict, duration_s: float):
    """Return [(start_beat, end_beat, bpm, start_sec)] covering the song."""
    segments = rhythm.get("segments") or []
    if not segments:
        bpm = float(rhythm.get("bpm") or 180.0)
        return [(0.0, duration_s * bpm / 60.0, bpm, 0.0)]
    out = []
    beat = 0.0
    for seg in segments:
        bpm = float(seg["bpm"])
        start = seg["start_ms"] / 1000.0
        end = seg["end_ms"] / 1000.0
        if end <= start:
            continue
        length = (end - start) * bpm / 60.0
        out.append((beat, beat + length, bpm, start))
        beat += length
    return out or [(0.0, duration_s * 180.0 / 60.0, 180.0, 0.0)]


def _pick_single(rng: random.Random, last: int | None) -> int:
    candidates = [lane for lane in LANES if lane != last]
    if last is not None:
        opposite = [lane for lane in candidates if lane // 2 != last // 2]
        if opposite and rng.random() < 0.72:
            candidates = opposite
    return rng.choice(candidates)


def _pick_jump(rng: random.Random, last: int | None) -> list[int]:
    left = [0, 1] if last not in (0, 1) else [0, 1]
    right = [2, 3] if last not in (2, 3) else [2, 3]
    return [rng.choice(left), rng.choice(right)]


def _pick_hand(rng: random.Random, last: int | None) -> list[int]:
    pool = [0, 1, 2, 3]
    rng.shuffle(pool)
    return sorted(pool[:3])


def _stream_block(rng, start_beat, end_beat, step=0.5,
                  jump_rate=0.18, hand_rate=0.03, offset=0.0):
    notes = []
    last = None
    count = int((end_beat - start_beat - offset) / step)
    for k in range(count):
        beat = start_beat + offset + k * step
        roll = rng.random()
        if roll < hand_rate and k % 16 == 0:
            lanes = _pick_hand(rng, last)
        elif roll < hand_rate + jump_rate and k % 4 == 0:
            lanes = _pick_jump(rng, last)
        else:
            lanes = [_pick_single(rng, last)]
        for lane in lanes:
            notes.append((beat, lane))
        last = lanes[-1]
    return notes


def _speed_block(rng, start_beat, end_beat, bpm):
    step = 0.25 if bpm <= 210 else 0.5
    return _stream_block(rng, start_beat, end_beat, step=step,
                         jump_rate=0.08, hand_rate=0.0)


def _jack_block(rng, start_beat, end_beat, bpm):
    notes = []
    step = 0.5
    beat = start_beat
    last_lane = None
    block_index = 0
    while beat < end_beat - 1e-6:
        lanes = [rng.choice([lane for lane in LANES if lane != last_lane])]
        if block_index % 16 == 8:
            lanes = _pick_jump(rng, last_lane)
        elif block_index % 32 == 16:
            lanes = _pick_hand(rng, last_lane)
        run = rng.choice([2, 3, 4])
        for _ in range(run):
            if beat >= end_beat - 1e-6:
                break
            for lane in lanes:
                notes.append((beat, lane))
            beat += step
        last_lane = lanes[-1]
        block_index += 1
    return notes


def _tech_block(rng, start_beat, end_beat, bpm):
    notes = []
    phrase = 0
    beat = start_beat
    while beat < end_beat - 1e-6:
        remaining = min(8.0, end_beat - beat)
        mode = phrase % 4
        if mode == 0:
            notes += _stream_block(rng, beat, beat + remaining, step=0.5,
                                   jump_rate=0.12, hand_rate=0.0)
        elif mode == 1:
            notes += _jack_block(rng, beat, beat + remaining, bpm)
        elif mode == 2:
            notes += _stream_block(rng, beat, beat + remaining, step=0.5,
                                   jump_rate=0.06, hand_rate=0.02, offset=0.25)
        else:
            lane_a, lane_b = rng.sample([0, 1, 2, 3], 2)
            for k in range(int(remaining / 0.25)):
                lane = lane_a if k % 2 == 0 else lane_b
                notes.append((beat + k * 0.25, lane))
            if remaining >= 4:
                notes.append((beat, rng.choice([lane for lane in LANES
                                                if lane not in (lane_a, lane_b)])))
        beat += remaining
        phrase += 1
    return notes


def generate_chart(style: str, rhythm: dict, duration_s: float,
                   title: str = "", artist: str = "Unknown",
                   seed: int | None = None) -> dict:
    """Build a Malody ``.mc`` dict for one of the four styles."""
    if style not in STYLES:
        raise ValueError(f"unknown style: {style}")
    rng = random.Random(seed)
    timeline = tempo_timeline(rhythm, duration_s)

    notes: list[tuple[float, int]] = []
    for start_beat, end_beat, bpm, _start_sec in timeline:
        if style == "stream":
            block = _stream_block(rng, start_beat, end_beat, step=0.5,
                                  jump_rate=0.18, hand_rate=0.03)
        elif style == "speed":
            block = _speed_block(rng, start_beat, end_beat, bpm)
        elif style == "jack":
            block = _jack_block(rng, start_beat, end_beat, bpm)
        else:
            block = _tech_block(rng, start_beat, end_beat, bpm)
        notes.extend(block)

    unique = {}
    for beat, lane in notes:
        if beat < 0 or lane not in LANES:
            continue
        unique[(round(beat, 4), lane)] = (round(beat, 4), lane)
    ordered = sorted(unique.values())

    note_json = [{"beat": _beat_list(beat), "column": lane} for beat, lane in ordered]
    time_json = []
    for start_beat, _end_beat, bpm, _start_sec in timeline:
        item = {"beat": _beat_list(start_beat), "bpm": round(bpm, 6)}
        if not time_json or time_json[-1]["bpm"] != item["bpm"]:
            time_json.append(item)
    if not time_json:
        time_json = [{"beat": [0, 0, 1], "bpm": float(rhythm.get("bpm") or 180.0)}]

    meta = {
        "$ver": 0,
        "creator": "MalodyAutoChart",
        "background": "",
        "version": f"Auto {STYLES[style]}",
        "id": 0,
        "mode": 0,
        "time": 0,
        "song": {
            "title": title or "Untitled",
            "artist": artist or "Unknown",
            "id": 0,
            "titleorg": title or "Untitled",
            "artistorg": artist or "Unknown",
        },
        "mode_ext": {"column": 4},
    }
    return {
        "meta": meta,
        "time": time_json,
        "effect": [],
        "note": note_json,
        "extra": {},
    }


def save_chart(chart: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chart, ensure_ascii=False, separators=(",", ":")),
                    encoding="utf-8")
    return path


def chart_stats(chart: dict) -> dict:
    notes = chart.get("note") or []
    by_beat = {}
    for note in notes:
        if isinstance(note, dict):
            beat_list, column = note.get("beat"), note.get("column")
        elif isinstance(note, (list, tuple)) and len(note) >= 2:
            beat_list, column = note[0], note[1]
        else:
            continue
        if not beat_list or column is None:
            continue
        beat = beat_list[0] + (beat_list[1] / beat_list[2] if len(beat_list) > 2 else 0.0)
        by_beat.setdefault(round(beat, 4), []).append(int(column))
    chords = [len(v) for v in by_beat.values()]
    repeats = 0
    last_columns = None
    for beat in sorted(by_beat):
        current = set(by_beat[beat])
        if last_columns:
            repeats += len(current & last_columns)
        last_columns = current
    return {
        "notes": len(notes),
        "rows": len(by_beat),
        "jumps": sum(1 for c in chords if c == 2),
        "hands": sum(1 for c in chords if c >= 3),
        "same_column_steps": repeats,
        "max_chord": max(chords) if chords else 0,
    }
