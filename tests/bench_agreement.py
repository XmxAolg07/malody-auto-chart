"""Benchmark: per-row key-count agreement between generated and reference charts.

For every chart/audio pair in the eight dan folders:

* load the audio, estimate its rhythm, generate a chart in the folder's style,
* convert both charts to (time, row key count),
* for each reference row, look for a generated row within ``--tol`` seconds,
  and count a hit when the key counts are identical.

This is the strict metric requested for training: "same number of keys at the
same row".  It is intentionally hard; a row can only match if the generator
places a row at the same time *and* chooses the same chord size.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

from core.audio import load_audio  # noqa: E402
from core.bpm import estimate_rhythm  # noqa: E402
from core.chartgen import generate_chart  # noqa: E402

FOLDERS = {
    "stream": ["ex stream/0", "reg stream/0"],
    "speed": ["ex speed/0", "reg speed/0"],
    "jack": ["ex jack/0", "reg jack/0"],
    "tech": ["ex tech/0", "reg tech/0"],
}


def chart_rows(chart: dict):
    times = []
    for item in chart.get("time") or []:
        beat = item["beat"]
        times.append((beat[0] + beat[1] / beat[2], float(item["bpm"])))
    times.sort()
    segments = []
    seconds = 0.0
    for index, (beat, bpm) in enumerate(times):
        if index:
            prev_beat, prev_seconds, prev_bpm = segments[-1]
            seconds = prev_seconds + (beat - prev_beat) * 60.0 / prev_bpm
        segments.append((beat, seconds, bpm))
    rows = {}
    for note in chart.get("note") or []:
        if isinstance(note, dict):
            beat_list, lane = note.get("beat"), note.get("column")
        elif isinstance(note, (list, tuple)) and len(note) >= 2:
            beat_list, lane = note[0], note[1]
        else:
            continue
        if beat_list is None or lane is None:
            continue
        beat = beat_list[0] + beat_list[1] / beat_list[2]
        segment = segments[0]
        for candidate in segments:
            if candidate[0] <= beat + 1e-9:
                segment = candidate
            else:
                break
        sec = segment[1] + (beat - segment[0]) * 60.0 / segment[2]
        key = round(sec * 96) / 96
        rows[key] = rows.get(key, 0) | (1 << int(lane))
    return [(key, mask.bit_count()) for key, mask in sorted(rows.items())]


def agreement(reference, generated, tolerance: float):
    generated_times = [time for time, _size in generated]
    hits = matched = 0
    for time, size in reference:
        position = np.searchsorted(generated_times, time)
        best = None
        for index in (position - 1, position):
            if 0 <= index < len(generated):
                distance = abs(generated_times[index] - time)
                if distance <= tolerance and (best is None or distance < best[0]):
                    best = (distance, generated[index][1])
        if best is not None:
            matched += 1
            if best[1] == size:
                hits += 1
    total = max(1, len(reference))
    return hits / total, matched / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2, help="charts per folder")
    parser.add_argument("--tol", type=float, default=0.08)
    args = parser.parse_args()

    results = []
    for style, folders in FOLDERS.items():
        for folder in folders:
            base = ROOT / folder
            charts = sorted(base.glob("*.mc"))[: args.limit]
            for path in charts:
                data = json.loads(path.read_text(encoding="utf-8"))
                background = (data.get("meta") or {}).get("background") or ""
                audio = base / (Path(background).stem + ".ogg")
                if not audio.exists():
                    continue
                samples, sr = load_audio(str(audio))
                rhythm = estimate_rhythm(samples, sr).to_dict()
                generated = generate_chart(
                    style, rhythm, len(samples) / sr, title=path.stem,
                    seed=7, samples=samples, sample_rate=sr)
                reference_rows = chart_rows(data)
                generated_rows = chart_rows(generated)
                hit, matched = agreement(reference_rows, generated_rows, args.tol)
                results.append((style, audio.stem, len(reference_rows),
                                len(generated_rows), hit, matched))
                print(f"{style:6} {audio.stem[:36]:36} ref={len(reference_rows):4} "
                      f"gen={len(generated_rows):4} hit={hit:.1%} matched={matched:.1%}")

    if results:
        hits = [r[4] for r in results]
        matched = [r[5] for r in results]
        print(f"\nsummary: charts={len(results)} "
              f"mean hit={np.mean(hits):.1%} "
              f"best={np.max(hits):.1%} worst={np.min(hits):.1%} "
              f"mean matched={np.mean(matched):.1%}")


if __name__ == "__main__":
    main()
