"""Quality filter for the imported community chart corpus.

Computes anchor length, density, duration, BPM, chord sizes and duplicate
signatures, drops clearly unreasonable charts, and writes a clean manifest plus
a rejection report.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "...AAA实例谱子存放处！" / "beatmap"
OUT_DIR = ROOT / "community_clean"
OUT_INDEX = OUT_DIR / "index.json"
OUT_REJECT = OUT_DIR / "rejected.json"


def _beat(value) -> float:
    if not value:
        return 0.0
    if len(value) == 1:
        return float(value[0])
    return float(value[0]) + float(value[1]) / float(value[2] or 1)


AUDIO_SUFFIXES = (".ogg", ".mp3", ".wav", ".flac", ".m4a")
AUDIO_EXTS = {".ogg", ".mp3", ".wav", ".flac", ".m4a"}


def _norm(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def find_audio(path: Path, meta: dict) -> str:
    folders = [path.parent, path.parent.parent]
    audios = []
    for folder in folders:
        try:
            for item in folder.iterdir():
                if item.is_file() and item.suffix.lower() in AUDIO_SUFFIXES:
                    audios.append(item)
        except OSError:
            continue
    if not audios:
        return ""
    chart_stem = _norm(path.stem)
    background = (meta.get("background") or "")
    bg_stem = _norm(Path(background).stem)
    song = meta.get("song") or {}
    title = _norm(str(song.get("title") or ""))
    best, best_score = None, 0
    for audio in audios:
        stem = _norm(audio.stem)
        score = 0
        if stem == chart_stem:
            score = 100
        elif bg_stem and stem == bg_stem:
            score = 90
        elif len(stem) >= 4 and (stem in bg_stem or bg_stem in stem):
            score = 80
        elif len(title) >= 4 and (stem in title or title in stem):
            score = 70
        elif len(audios) == 1:
            score = 60
        if score > best_score:
            best, best_score = audio, score
    return str(best) if best else ""


def analyze(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    times = []
    for item in data.get("time") or []:
        try:
            times.append((_beat(item["beat"]), float(item["bpm"])))
        except Exception:
            continue
    if not times:
        return None
    times.sort()
    segments = []
    seconds = 0.0
    for index, (beat, bpm) in enumerate(times):
        if index:
            prev_beat, prev_seconds, prev_bpm = segments[-1]
            seconds = prev_seconds + (beat - prev_beat) * 60.0 / prev_bpm
        segments.append((beat, seconds, bpm))

    def time_of(beat: float) -> float:
        segment = segments[0]
        for candidate in segments:
            if candidate[0] <= beat + 1e-9:
                segment = candidate
            else:
                break
        return segment[1] + (beat - segment[0]) * 60.0 / segment[2]

    rows = {}
    ln_notes = 0
    for note in data.get("note") or []:
        if isinstance(note, dict):
            beat_list, lane = note.get("beat"), note.get("column")
            end_list = note.get("endbeat")
        else:
            beat_list, lane = note[0], note[1]
            end_list = None
        if beat_list is None or lane is None:
            continue
        lane = int(lane)
        if not 0 <= lane < 4:
            return None
        beat = _beat(beat_list)
        key = round(beat * 48) / 48
        rows[key] = rows.get(key, 0) | (1 << lane)
        if end_list is not None and _beat(end_list) > beat:
            ln_notes += 1
    if len(rows) < 20:
        return None
    keys = sorted(rows)
    duration = max(0.1, time_of(keys[-1]) - time_of(keys[0]))
    notes = sum(mask.bit_count() for mask in rows.values())
    sizes = Counter(mask.bit_count() for mask in rows.values())
    runs = {lane: 0 for lane in range(4)}
    anchor = 0
    previous = None
    for beat in keys:
        if previous is not None and beat - previous > 1.0:
            runs = {lane: 0 for lane in range(4)}
        for lane in range(4):
            if rows[beat] & (1 << lane):
                runs[lane] += 1
                anchor = max(anchor, runs[lane])
            else:
                runs[lane] = 0
        previous = beat
    meta = data.get("meta") or {}
    audio = find_audio(path, meta)
    bpm = float(np.median([bpm for _beat, bpm in times]))
    notes_per_sec = notes / duration
    rows_per_sec = len(rows) / duration
    signature = hashlib.md5(
        f"{round(duration, 1)}|{notes}|{keys[0]:.2f}|{keys[-1]:.2f}|{bpm:.1f}".encode()
    ).hexdigest()
    return {
        "path": str(path),
        "name": path.stem,
        "duration": round(duration, 2),
        "bpm": round(bpm, 2),
        "notes": notes,
        "rows": len(rows),
        "notes_per_sec": round(notes_per_sec, 2),
        "rows_per_sec": round(rows_per_sec, 2),
        "max_anchor": int(anchor),
        "max_chord": int(max(sizes)),
        "singles": sizes.get(1, 0),
        "jumps": sizes.get(2, 0),
        "hands": sizes.get(3, 0),
        "quads": sizes.get(4, 0),
        "ln_ratio": round(ln_notes / max(1, notes), 3),
        "signature": signature,
        "audio": audio,
    }


def main() -> None:
    charts = []
    for path in sorted(CORPUS.rglob("*.mc")):
        metrics = analyze(path)
        if metrics:
            charts.append(metrics)
    print("parsed", len(charts))
    if not charts:
        return
    for key in ("max_anchor", "notes_per_sec", "rows_per_sec", "duration", "bpm", "max_chord"):
        values = np.array([item[key] for item in charts], dtype=float)
        print(key, "p1", np.percentile(values, 1), "p50", np.percentile(values, 50),
              "p99", np.percentile(values, 99), "max", values.max())
    accepted, rejected = [], []
    seen = set()
    for item in charts:
        reasons = []
        if item["notes"] < 80:
            reasons.append("too_few_notes")
        if not 20 <= item["duration"] <= 600:
            reasons.append("duration")
        if not 60 <= item["bpm"] <= 400:
            reasons.append("bpm")
        if item["max_chord"] > 4:
            reasons.append("chord")
        if item["max_anchor"] > 8:
            reasons.append("anchor")
        if not 1.5 <= item["notes_per_sec"] <= 35:
            reasons.append("density")
        if not 0.8 <= item["rows_per_sec"] <= 20:
            reasons.append("row_density")
        if not item["audio"] or not Path(item["audio"]).is_file():
            reasons.append("no_audio")
        if item["signature"] in seen:
            reasons.append("duplicate")
        if reasons:
            item["reasons"] = reasons
            rejected.append(item)
        else:
            seen.add(item["signature"])
            accepted.append(item)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_INDEX.write_text(json.dumps(accepted, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT_REJECT.write_text(json.dumps(rejected, ensure_ascii=False, indent=1), encoding="utf-8")
    print("accepted", len(accepted), "rejected", len(rejected))
    counts = Counter(reason for item in rejected for reason in item["reasons"])
    print("reasons", counts.most_common())
    print("accepted density p50 p95", np.percentile([a["notes_per_sec"] for a in accepted], [50, 95]))
    print("accepted anchor max", max(a["max_anchor"] for a in accepted))


if __name__ == "__main__":
    main()
