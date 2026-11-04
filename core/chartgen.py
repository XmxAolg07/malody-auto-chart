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
import bisect
import math
import random
from pathlib import Path

from .energy import EnergyMap
from .pattern_model import load_models as load_pattern_models, sample_mask, sample_spacing
from .row_model import load_models as load_row_models, predict_size
from .row_timing import load_models as load_timing_models, predict_timing

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



STYLE_RULES = {
    "stream": {"lane_gap": 0.5, "max_jumps_per_beat": 1},
    "speed": {"lane_gap": 0.25, "max_jumps_per_beat": 2},
    "jack": {"lane_gap": 0.25, "max_jumps_per_beat": 99},
    "tech": {"lane_gap": 0.5, "max_jumps_per_beat": 2},
}


def _lanes_of(mask: int) -> list[int]:
    return [lane for lane in range(4) if mask & (1 << lane)]


def _mask_allowed(mask: int, beat: float, lane_last: dict, jump_count: dict,
                  rules: dict) -> bool:
    for lane in _lanes_of(mask):
        if beat - lane_last.get(lane, -99.0) < rules["lane_gap"] - 1e-9:
            return False
    if mask.bit_count() >= 2 and rules["max_jumps_per_beat"] < 90:
        bucket = int(math.floor(beat + 1e-9))
        if jump_count.get(bucket, 0) >= rules["max_jumps_per_beat"]:
            return False
    return True


def _select_mask(rng, model, prev, beat, lane_last, jump_count, rules,
                 max_size=4, single_bias=0.0) -> int:
    for _ in range(12):
        mask = sample_mask(model, rng, prev)
        if mask.bit_count() > max_size:
            continue
        if single_bias and mask.bit_count() > 1 and rng.random() < single_bias:
            continue
        if _mask_allowed(mask, beat, lane_last, jump_count, rules):
            return mask
    lanes = [lane for lane in range(4)
             if beat - lane_last.get(lane, -99.0) >= rules["lane_gap"]]
    if not lanes:
        lanes = list(range(4))
    return 1 << rng.choice(lanes)


def _place(mask: int, beat: float, notes: list, lane_last: dict,
           jump_count: dict) -> None:
    lanes = _lanes_of(mask)
    for lane in lanes:
        notes.append((beat, lane))
        lane_last[lane] = beat
    if len(lanes) >= 2:
        bucket = int(math.floor(beat + 1e-9))
        jump_count[bucket] = jump_count.get(bucket, 0) + 1


def _stream_notes(rng, model, start, end, bpm, state):
    notes = []
    lane_last = state["lane_last"]
    jump_count = state["jump_count"]
    rules = STYLE_RULES["stream"]
    beat = start
    prev = state.get("prev")
    run = 0
    while beat < end - 1e-6:
        mask = _select_mask(rng, model, prev, beat, lane_last, jump_count,
                            rules, max_size=3)
        _place(mask, beat, notes, lane_last, jump_count)
        prev = mask
        state["prev"] = prev
        run += 1
        gap = 0.25
        if run >= 8:          # at most two beats of continuous switching
            gap = 0.5
            run = 0
        elif rng.random() < 0.12:
            gap = 0.5
        beat += gap
    return notes


def _speed_notes(rng, model, start, end, bpm, state):
    notes = []
    lane_last = state["lane_last"]
    jump_count = state["jump_count"]
    rules = STYLE_RULES["speed"]
    step = 0.125 if bpm <= 220 else 0.25
    beat = start
    prev = state.get("prev")
    while beat < end - 1e-6:
        burst = 0
        while burst < 16 and beat < end - 1e-6:
            mask = _select_mask(rng, model, prev, beat, lane_last, jump_count,
                                rules, max_size=2, single_bias=0.85)
            _place(mask, beat, notes, lane_last, jump_count)
            prev = mask
            state["prev"] = prev
            beat += step
            burst += 1
        beat += 0.75
    return notes


def _tech_notes(rng, model, start, end, bpm, state):
    notes = []
    lane_last = state["lane_last"]
    jump_count = state["jump_count"]
    rules = STYLE_RULES["tech"]
    beat = start
    prev = state.get("prev")
    while beat < end - 1e-6:
        rows = rng.choice([4, 6, 8])
        for _ in range(rows):
            if beat >= end - 1e-6:
                break
            mask = _select_mask(rng, model, prev, beat, lane_last, jump_count,
                                rules, max_size=3)
            _place(mask, beat, notes, lane_last, jump_count)
            prev = mask
            state["prev"] = prev
            beat += 0.25 if rng.random() < 0.75 else 0.125
        beat += rng.choice([1.0, 1.5, 2.0])
    return notes


def _jack_notes(rng, model, start, end, bpm, state):
    notes, lane_last, jump_count = [], {}, {}
    beat = start
    while beat < end - 1e-6:
        size = rng.choices([1, 2, 3, 4], weights=[24, 43, 27, 6], k=1)[0]
        lanes = rng.sample(range(4), size)
        mask = sum(1 << lane for lane in lanes)
        repeats = rng.choice([2, 3, 4])
        for _ in range(repeats):
            if beat >= end - 1e-6:
                break
            _place(mask, beat, notes, lane_last, jump_count)
            beat += 0.25
        beat += rng.choice([0.5, 0.75])
    return notes


def _style_notes(style, rng, model, start, end, bpm, state):
    if style == "stream":
        return _stream_notes(rng, model, start, end, bpm, state)
    if style == "speed":
        return _speed_notes(rng, model, start, end, bpm, state)
    if style == "jack":
        return _jack_notes(rng, model, start, end, bpm, state)
    return _tech_notes(rng, model, start, end, bpm, state)


POST_RULES = {
    "stream": {"lane_gap": 0.5, "max_jumps_per_beat": 1,
               "max_run": 8, "run_gap": 0.25, "max_chord": 3},
    "speed": {"lane_gap": 0.25, "max_jumps_per_beat": 2,
              "max_run": 16, "run_gap": 0.25, "max_chord": 2},
    "jack": {"lane_gap": 0.25, "max_jumps_per_beat": 99,
             "max_run": 99, "jack_repeat": 4, "max_chord": 4},
    "tech": {"lane_gap": 0.25, "max_jumps_per_beat": 2,
             "max_run": 8, "run_gap": 0.25, "max_chord": 3},
}


def _enforce_constraints(style: str, notes: list[tuple[float, int]]) -> list[tuple[float, int]]:
    """Hard-filter rows so every style rule holds in the final chart."""
    rules = POST_RULES.get(style)
    if not rules:
        return notes
    rows = {}
    for beat, lane in notes:
        key = round(beat * 48) / 48
        rows[key] = rows.get(key, 0) | (1 << lane)
    kept = []
    lane_last = {}
    jump_by_beat = {}
    prev_beat = None
    prev_mask = None
    run = 0
    repeat = 0
    for key in sorted(rows):
        mask = rows[key]
        gap = None if prev_beat is None else key - prev_beat
        if gap is not None and gap <= rules.get("run_gap", 0.0) + 1e-9:
            run += 1
        else:
            run = 1
        if run > rules["max_run"]:
            continue
        allowed = [lane for lane in range(4)
                   if mask & (1 << lane)
                   and key - lane_last.get(lane, -99.0) >= rules["lane_gap"] - 1e-9]
        if not allowed:
            continue
        limit = rules.get("max_chord", 4)
        if len(allowed) > limit:
            allowed = allowed[:limit]
        mask = sum(1 << lane for lane in allowed)
        if mask.bit_count() >= 2 and rules["max_jumps_per_beat"] < 90:
            bucket = int(math.floor(key + 1e-9))
            if jump_by_beat.get(bucket, 0) >= rules["max_jumps_per_beat"]:
                mask = 1 << allowed[0]
        if rules.get("jack_repeat"):
            if mask == prev_mask:
                repeat += 1
            else:
                prev_mask = mask
                repeat = 1
            if repeat > rules["jack_repeat"]:
                continue
        for lane in range(4):
            if mask & (1 << lane):
                lane_last[lane] = key
        if mask.bit_count() >= 2:
            bucket = int(math.floor(key + 1e-9))
            jump_by_beat[bucket] = jump_by_beat.get(bucket, 0) + 1
        kept.append((key, mask))
        prev_beat = key
    return [(beat, lane) for beat, mask in kept
            for lane in range(4) if mask & (1 << lane)]


def _beat_to_sec(beat: float, timeline) -> float:
    seg = timeline[-1]
    for candidate in timeline:
        if beat <= candidate[1] + 1e-9:
            seg = candidate
            break
    return seg[3] + max(0.0, beat - seg[0]) * 60.0 / seg[2]


def _sec_to_beat(seconds: float, timeline) -> float:
    seg = timeline[-1]
    for candidate in timeline:
        start_beat, end_beat, bpm, start_sec = candidate
        if seconds <= start_sec + (end_beat - start_beat) * 60.0 / bpm + 1e-9:
            seg = candidate
            break
    return seg[0] + max(0.0, seconds - seg[3]) * seg[2] / 60.0


def _grow_mask(mask: int, extra: int, style: str, rng) -> int:
    max_size = {"stream": 3, "speed": 2, "jack": 4, "tech": 3}[style]
    lanes = [lane for lane in range(4) if not mask & (1 << lane)]
    rng.shuffle(lanes)
    for lane in lanes[:max(0, max_size - mask.bit_count())]:
        mask |= 1 << lane
    return mask


def _timeline_bpm(beat: float, timeline) -> float:
    seg = timeline[-1]
    for candidate in timeline:
        if beat <= candidate[1] + 1e-9:
            seg = candidate
            break
    return seg[2]


def _apply_timing(style: str, notes, energy, timeline, timing_models):
    model = timing_models.get(style)
    if not model or energy is None:
        return notes
    peak_times = list(getattr(energy, "_peak_times", []))
    peak_strength = list(getattr(energy, "_peak_strength", []))
    if len(peak_times) < 8:
        return notes
    selected = []
    threshold = float(model.get("threshold", 0.5))
    for index, seconds in enumerate(peak_times):
        beat = _sec_to_beat(seconds, timeline)
        bpm = _timeline_bpm(beat, timeline)
        lo = max(0, index - 40)
        hi = min(len(peak_strength), index + 40)
        local = peak_strength[lo:hi]
        rank = sum(1 for value in local if value <= peak_strength[index]) / max(1, len(local))
        features = [
            peak_strength[index],
            min(1.0, energy.onset_at(seconds) * 4.0),
            min(1.0, energy.low_at(seconds) * 4.0),
            min(2.0, energy.ratio_at(seconds)),
            min(2.0, energy.section_ratio(seconds)),
            beat - int(beat),
            (beat / 4.0) - int(beat / 4.0),
            bpm / 180.0,
            min(0.5, seconds - peak_times[index - 1]) if index else 0.5,
            min(0.5, peak_times[index + 1] - seconds) if index + 1 < len(peak_times) else 0.5,
            min(8.0, energy.peak_density(seconds, 1.0)),
            rank,
        ]
        if predict_timing(model, features) >= threshold:
            selected.append(seconds)
    if not selected:
        return notes
    rows = {}
    for beat, lane in notes:
        key = round(beat * 48) / 48
        rows[key] = rows.get(key, 0) | (1 << lane)
    existing = sorted(rows)
    inserted = 0
    for seconds in selected:
        beat = _sec_to_beat(seconds, timeline)
        key = round(beat * 48) / 48
        pos = bisect.bisect_left(existing, key)
        near = False
        for index in (pos - 1, pos):
            if 0 <= index < len(existing):
                if abs(existing[index] - key) <= 0.12:
                    near = True
                    break
        if near or inserted >= 120:
            continue
        rows[key] = 1  # single note; row-size model may grow it later
        existing = sorted(rows)
        inserted += 1
    return [(beat, lane) for beat, mask in sorted(rows.items())
            for lane in range(4) if mask & (1 << lane)]


def _apply_row_model(style: str, notes, energy, timeline, row_models) -> list:
    model = row_models.get(style)
    if not model or energy is None:
        return notes
    rows = {}
    for beat, lane in notes:
        key = round(beat * 48) / 48
        rows[key] = rows.get(key, 0) | (1 << lane)
    prev1 = prev2 = prev3 = 0.0
    out = []
    for beat in sorted(rows):
        mask = rows[beat]
        seconds = _beat_to_sec(beat, timeline)
        peak = energy.nearest_peak(seconds, 0.12)
        strength = peak[1] if peak else 0.0
        dist = abs(peak[0] - seconds) if peak else 0.12
        bpm = _timeline_bpm(beat, timeline)
        features = [
            energy.ratio_at(seconds),
            energy.section_ratio(seconds),
            energy.onset_at(seconds),
            energy.low_at(seconds),
            energy.ratio_at(seconds + 0.1) - energy.ratio_at(seconds - 0.1),
            energy.section_ratio(seconds + 1.0) - energy.section_ratio(seconds - 1.0),
            strength,
            dist,
            energy.peak_density(seconds, 1.0),
            beat - int(beat),
            (beat / 4.0) - int(beat / 4.0),
            bpm / 180.0,
            prev1,
            prev2,
            prev3,
        ]
        size = predict_size(model, features)
        lanes = [lane for lane in range(4) if mask & (1 << lane)]
        if size < len(lanes):
            lanes = lanes[:size]
        elif size > len(lanes):
            for lane in range(4):
                if lane not in lanes:
                    lanes.append(lane)
                    if len(lanes) >= size:
                        break
        mask = sum(1 << lane for lane in lanes)
        out.append((beat, mask))
        prev3, prev2, prev1 = prev2, prev1, len(lanes) / 4.0
    return [(beat, lane) for beat, mask in out
            for lane in range(4) if mask & (1 << lane)]


def _apply_energy(style: str, notes, energy, timeline, rng):
    rows = {}
    for beat, lane in notes:
        key = round(beat * 48) / 48
        rows[key] = rows.get(key, 0) | (1 << lane)
    out = []
    for beat in sorted(rows):
        mask = rows[beat]
        seconds = _beat_to_sec(beat, timeline)
        level = energy.section_ratio(seconds)
        peak = energy.nearest_peak(seconds, 0.12)
        if level <= 0.95:
            if peak is None:
                if int(round(beat)) % 2:
                    continue
            else:
                if level > 0.65 and int(round(peak[0] * 4)) % 2:
                    continue
                beat = _sec_to_beat(peak[0], timeline)
                if level <= 0.65:
                    low_lane = (mask & -mask).bit_length() - 1
                    mask = 1 << low_lane
        if level >= 1.15 and peak is not None:
            extra = 1 if level < 1.35 else 2
            if style == "speed":
                extra = 1
            mask = _grow_mask(mask, extra, style, rng)
        elif style == "speed" and peak is None:
            low_lane = (mask & -mask).bit_length() - 1
            mask = 1 << low_lane
        out.append((beat, mask))
    return [(beat, lane) for beat, mask in out
            for lane in range(4) if mask & (1 << lane)]


def generate_chart(style: str, rhythm: dict, duration_s: float,
                   title: str = "", artist: str = "Unknown",
                   seed: int | None = None, energy=None,
                   samples=None, sample_rate: int | None = None) -> dict:
    """Build a Malody ``.mc`` dict for one of the four styles."""
    if style not in STYLES:
        raise ValueError(f"unknown style: {style}")
    rng = random.Random(seed)
    timeline = tempo_timeline(rhythm, duration_s)
    if energy is None and samples is not None and sample_rate:
        energy = EnergyMap(samples, sample_rate)

    models = load_pattern_models()
    row_models = load_row_models()
    timing_models = load_timing_models()
    model = models.get(style)
    notes: list[tuple[float, int]] = []
    if model:
        state = {"lane_last": {}, "jump_count": {}, "prev": None}
        for start_beat, end_beat, bpm, _start_sec in timeline:
            notes.extend(_style_notes(style, rng, model, start_beat, end_beat,
                                      bpm, state))
    else:
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

    if energy is not None:
        notes = _apply_timing(style, notes, energy, timeline, timing_models)
        notes = _apply_row_model(style, notes, energy, timeline, row_models)
        notes = _apply_energy(style, notes, energy, timeline, rng)
    notes = _enforce_constraints(style, notes)

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
