"""Learn 4K row models from the local dan charts.

The corpus is the eight folders ``ex/reg × jack/stream/speed/tech``; every
folder is labelled with its key-pattern type.  For each type we collect, in
beat space:

* the distribution of row sizes (single / jump / hand / quad),
* the distribution of gaps between rows (in beats),
* a 15×15 mask-transition matrix (a row is a 4-bit column mask),
* the initial-row distribution.

``chartgen`` samples this model instead of hand-written rules, so generated
patterns follow the rhythm and chord habits of the reference charts.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "assets" / "pattern_models.json"
FOLDERS = {
    "stream": ["ex stream/0", "reg stream/0"],
    "speed": ["ex speed/0", "reg speed/0"],
    "jack": ["ex jack/0", "reg jack/0"],
    "tech": ["ex tech/0", "reg tech/0"],
}
MASKS = tuple(range(1, 16))


def _beat_float(beat) -> float:
    if len(beat) == 1:
        return float(beat[0])
    return float(beat[0]) + float(beat[1]) / float(beat[2] or 1)


def _parse_rows(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for note in data.get("note") or []:
        if isinstance(note, dict):
            beat, lane = note.get("beat"), note.get("column")
        elif isinstance(note, (list, tuple)) and len(note) >= 2:
            beat, lane = note[0], note[1]
        else:
            continue
        if beat is None or lane is None:
            continue
        b = _beat_float(beat)
        key = round(b * 48) / 48
        rows[key] = rows.get(key, 0) | (1 << int(lane))
    ordered = sorted(rows.items())
    times = []
    for item in data.get("time") or []:
        try:
            times.append((_beat_float(item["beat"]), float(item["bpm"])))
        except Exception:
            continue
    times.sort()
    return ordered, times, data.get("meta") or {}


def _beat_to_seconds(beat: float, times) -> float:
    if not times:
        return beat * 0.5
    seg = times[0]
    for cand in times:
        if cand[0] <= beat + 1e-9:
            seg = cand
        else:
            break
    return (beat - seg[0]) * 60.0 / seg[1]


def _normalize(counts: Counter) -> dict[str, float]:
    total = float(sum(counts.values())) or 1.0
    return {str(key): round(value / total, 5)
            for key, value in sorted(counts.items()) if value > 0}


def build_models() -> dict:
    models = {}
    for style, folders in FOLDERS.items():
        sizes = Counter()
        spacing = Counter()
        init = Counter()
        transitions = defaultdict(Counter)
        charts = rows_total = same_mask = 0
        bpms, rps = [], []
        examples = []
        for folder in folders:
            for path in sorted((ROOT / folder).glob("*.mc")):
                rows, times, meta = _parse_rows(path)
                if len(rows) < 80:
                    continue
                charts += 1
                rows_total += len(rows)
                init[rows[0][1]] += 1
                for _beat, mask in rows:
                    sizes[mask.bit_count()] += 1
                for (b1, m1), (b2, m2) in zip(rows, rows[1:]):
                    gap = b2 - b1
                    if 0 < gap <= 4:
                        spacing[round(gap * 8) / 8] += 1
                    transitions[m1][m2] += 1
                    if m1 == m2:
                        same_mask += 1
                bpm = sorted(t[1] for t in times)[len(times) // 2] if times else 180.0
                bpms.append(bpm)
                if times:
                    seconds = _beat_to_seconds(rows[-1][0], times)
                    if seconds > 5:
                        rps.append(len(rows) / seconds)
                if len(examples) < 5:
                    examples.append({
                        "file": str(path.relative_to(ROOT)).replace("\\", "/"),
                        "version": meta.get("version") or path.stem,
                        "background": meta.get("background") or "",
                        "rows": len(rows),
                    })
        if not charts:
            continue
        trans = {str(mask): _normalize(counter) for mask, counter in transitions.items()}
        models[style] = {
            "charts": charts,
            "rows": rows_total,
            "bpm_median": round(sorted(bpms)[len(bpms) // 2], 3) if bpms else 180.0,
            "rows_per_sec": round(sorted(rps)[len(rps) // 2], 3) if rps else 10.0,
            "sizes": _normalize(sizes),
            "spacing": _normalize(spacing),
            "same_mask": round(same_mask / max(1, rows_total - charts), 5),
            "init": _normalize(init),
            "transitions": trans,
            "examples": examples,
        }
    return models


def save_models(models: dict, path: Path = MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(models, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_models(path: Path = MODEL_PATH) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def sample_mask(model: dict, rng: random.Random, prev: int | None = None) -> int:
    if prev is not None:
        dist = model.get("transitions", {}).get(str(prev))
        if dist:
            masks = [int(key) for key in dist]
            weights = list(dist.values())
            return rng.choices(masks, weights=weights, k=1)[0]
    dist = model.get("init") or {"1": 1.0}
    masks = [int(key) for key in dist]
    weights = list(dist.values())
    return rng.choices(masks, weights=weights, k=1)[0]


def sample_spacing(model: dict, rng: random.Random, bpm: float) -> float:
    dist = model.get("spacing") or {"0.25": 1.0}
    values = [float(key) for key in dist]
    weights = list(dist.values())
    gap = rng.choices(values, weights=weights, k=1)[0]
    # keep generated density playable: at most ~20 rows/second
    min_gap = bpm / (60.0 * 20.0)
    gap = max(gap, min_gap)
    # snap to 1/8 beat
    gap = round(gap * 8) / 8
    return max(0.125, min(2.0, gap))


if __name__ == "__main__":
    result = build_models()
    path = save_models(result)
    for style, model in result.items():
        print(style, "charts", model["charts"], "rows", model["rows"],
              "bpm", model["bpm_median"], "rps", model["rows_per_sec"],
              "sizes", model["sizes"], "spacing", model["spacing"])
    print("saved", path)
