"""Measure row coverage, key-count match and exact-mask match (mirror allowed)."""

from __future__ import annotations

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

CORPUS = ROOT / "...AAA实例谱子存放处！"
if not (CORPUS / "ex stream").exists():
    CORPUS = ROOT
FOLDERS = {
    "stream": ["ex stream/0", "reg stream/0"],
    "speed": ["ex speed/0", "reg speed/0"],
    "jack": ["ex jack/0", "reg jack/0"],
    "tech": ["ex tech/0", "reg tech/0"],
}


def chart_rows(chart):
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
        else:
            beat_list, lane = note[0], note[1]
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
    return sorted(rows.items())


def mirror(mask: int) -> int:
    return ((mask & 1) << 3) | ((mask & 2) << 1) | ((mask & 4) >> 1) | ((mask & 8) >> 3)


def compare(reference, generated, tol=0.08):
    generated_times = [time for time, _ in generated]
    matched = key_hit = mask_hit = mirror_hit = 0
    for time, ref_mask in reference:
        pos = np.searchsorted(generated_times, time)
        best = None
        for index in (pos - 1, pos):
            if 0 <= index < len(generated):
                distance = abs(generated_times[index] - time)
                if distance <= tol and (best is None or distance < best[0]):
                    best = (distance, generated[index][1])
        if best is None:
            continue
        matched += 1
        gen_mask = best[1]
        if gen_mask.bit_count() == ref_mask.bit_count():
            key_hit += 1
        if gen_mask == ref_mask:
            mask_hit += 1
        if gen_mask == ref_mask or gen_mask == mirror(ref_mask):
            mirror_hit += 1
    total = max(1, len(reference))
    return {
        "rows": len(reference),
        "coverage": matched / total,
        "key": key_hit / total,
        "mask": mask_hit / total,
        "mirror": mirror_hit / total,
    }


def main() -> None:
    all_stats = []
    for style, folders in FOLDERS.items():
        for folder in folders:
            base = CORPUS / folder
            for path in sorted(base.glob("*.mc"))[:2]:
                data = json.loads(path.read_text(encoding="utf-8"))
                background = (data.get("meta") or {}).get("background") or ""
                audio = base / (Path(background).stem + ".ogg")
                if not audio.exists():
                    continue
                samples, sr = load_audio(str(audio))
                rhythm = estimate_rhythm(samples, sr).to_dict()
                chart = generate_chart(style, rhythm, len(samples) / sr,
                                       title=path.stem, seed=7,
                                       samples=samples, sample_rate=sr)
                ref = chart_rows(data)
                gen = chart_rows(chart)
                stats = compare(ref, gen)
                stats.update(style=style, name=audio.stem)
                all_stats.append(stats)
                print(f"{style:6} {audio.stem[:34]:34} rows={stats['rows']:4} "
                      f"cov={stats['coverage']:.1%} key={stats['key']:.1%} "
                      f"mask={stats['mask']:.1%} mirror={stats['mirror']:.1%}")
    if all_stats:
        print("\nsummary: "
              f"coverage={np.mean([s['coverage'] for s in all_stats]):.1%} "
              f"key={np.mean([s['key'] for s in all_stats]):.1%} "
              f"mask={np.mean([s['mask'] for s in all_stats]):.1%} "
              f"mirror={np.mean([s['mirror'] for s in all_stats]):.1%}")


if __name__ == "__main__":
    main()
