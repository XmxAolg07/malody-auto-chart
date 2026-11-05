"""Convert osu!mania 4K .osu charts to Malody .mc JSON.

Handles uninherited timing points, circles and holds, and 4-key column mapping.
The output is a normal Malody chart that the training pipeline can parse.
"""

from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path


def _beat_list(value: float) -> list[int]:
    value = max(0.0, float(value))
    whole = int(value)
    frac = Fraction(value - whole).limit_denominator(192)
    return [whole, frac.numerator, frac.denominator]


def _sections(text: str) -> dict[str, list[str]]:
    out, current = {}, None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            out[current] = []
        elif current and line:
            out[current].append(line)
    return out


def _key_value(lines: list[str], key: str) -> str | None:
    for line in lines:
        if ":" in line:
            name, value = line.split(":", 1)
            if name.strip() == key:
                return value.strip()
    return None


def parse_osu(text: str) -> dict | None:
    sections = _sections(text)
    mode = _key_value(sections.get("General", []), "Mode")
    if mode is not None and mode != "3":
        return None
    circle = _key_value(sections.get("Difficulty", []), "CircleSize")
    if circle is not None and int(float(circle)) != 4:
        return None
    timing = []
    for line in sections.get("TimingPoints", []):
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            offset = float(parts[0])
            beat_length = float(parts[1])
            uninherited = int(float(parts[6])) if len(parts) > 6 else 1
        except ValueError:
            continue
        if uninherited == 1 and beat_length > 0:
            timing.append((offset, beat_length))
    if not timing:
        return None
    timing.sort()
    segments = []
    beat = 0.0
    for index, (offset, length) in enumerate(timing):
        if index:
            prev_offset, prev_length = timing[index - 1][0], timing[index - 1][1]
            beat += (offset - prev_offset) / prev_length
        segments.append((offset, beat, length))

    def time_to_beat(ms: float) -> float:
        seg = segments[0]
        for candidate in segments:
            if candidate[0] <= ms + 1e-6:
                seg = candidate
            else:
                break
        return seg[1] + (ms - seg[0]) / seg[2]

    notes = []
    for line in sections.get("HitObjects", []):
        parts = line.split(",")
        if len(parts) < 4:
            continue
        try:
            x = float(parts[0])
            start = float(parts[2])
            kind = int(parts[3])
        except ValueError:
            continue
        if not kind & 1:
            continue
        lane = max(0, min(3, int(x * 4 / 512)))
        end = None
        if kind & 128 and len(parts) >= 6:
            try:
                end = float(parts[5].split(":")[0])
            except ValueError:
                end = None
        notes.append((start, lane, end))
    if not notes:
        return None
    notes.sort()
    meta = sections.get("Metadata", [])
    general = sections.get("General", [])
    events = sections.get("Events", [])
    background = ""
    for line in events:
        if line.startswith("0,0,"):
            match = re.search(r'"([^"]+)"', line)
            if match:
                background = match.group(1)
                break
    version = _key_value(sections.get("Metadata", []), "Version") or ""
    audio = _key_value(general, "AudioFilename") or ""
    return {
        "meta": {
            "version": version,
            "creator": 0,
            "background": background,
            "mode_ext": {"column": 4, "time": 0},
            "song": {
                "title": _key_value(meta, "Title") or "",
                "artist": _key_value(meta, "Artist") or "",
                "id": 0,
                "titleorg": _key_value(meta, "TitleUnicode") or "",
                "artistorg": _key_value(meta, "ArtistUnicode") or "",
            },
            "audio": audio,
        },
        "time": [{"beat": [0, 0, 1], "bpm": round(60000.0 / length, 6)}
                 for _offset, _beat, length in segments],
        "note": [
            {"beat": _beat_list(time_to_beat(start)), "column": lane,
             **({"endbeat": _beat_list(time_to_beat(end))}
                if end is not None and end > start else {})}
            for start, lane, end in notes
        ],
        "extra": {"osu": True, "audio": audio},
    }


def convert_file(path: Path) -> Path | None:
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return None
    data = parse_osu(text)
    if data is None:
        return None
    out = path.with_suffix(".mc")
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:]:
        result = convert_file(Path(arg))
        print(result or "skip")
