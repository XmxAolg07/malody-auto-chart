"""Lane-arrangement model trained on the example charts.

Row timing and chord sizes are fixed by the other models; this model only
decides *which columns* a row uses.  It learns P(next mask | previous mask,
chord size) from the reference charts, with add-alpha smoothing over masks of
the same size, and samples arrangements that follow the corpus's hand feel.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .row_model import chart_rows

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "...AAA实例谱子存放处！"
if not (CORPUS / "ex stream").exists():
    CORPUS = ROOT
MODEL_PATH = ROOT / "assets" / "lane_models.json"
FOLDERS = {
    "stream": ["ex stream/0", "reg stream/0"],
    "speed": ["ex speed/0", "reg speed/0"],
    "jack": ["ex jack/0", "reg jack/0"],
    "tech": ["ex tech/0", "reg tech/0"],
}
MASKS_BY_SIZE = {size: [mask for mask in range(1, 16) if mask.bit_count() == size]
                 for size in range(1, 5)}


def _mirror(mask: int) -> int:
    return ((mask & 1) << 3) | ((mask & 2) << 1) | ((mask & 4) >> 1) | ((mask & 8) >> 3)


def _parse_folder(folder: str):
    base = CORPUS / folder
    charts = []
    for path in sorted(base.glob("*.mc")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = chart_rows(data)
        if len(rows) < 80:
            continue
        masks = []
        for _beat, _sec, mask, _bpm in rows:
            masks.append(mask)
        charts.append((path.stem, masks))
    return charts


def _normalize(counter: Counter, masks: list[int], alpha=0.15) -> dict[str, float]:
    total = float(sum(counter.values())) + alpha * len(masks)
    return {str(mask): (counter.get(mask, 0) + alpha) / total for mask in masks}


def split_charts(charts, val_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    order = np.arange(len(charts))
    rng.shuffle(order)
    cut = max(1, int(len(charts) * val_ratio))
    val = set(order[:cut].tolist())
    return ([charts[i] for i in range(len(charts)) if i not in val],
            [charts[i] for i in range(len(charts)) if i in val])


def build_model(style: str, train_charts, val_charts):
    by_size_init = defaultdict(Counter)
    transitions = defaultdict(Counter)  # (prev_mask, size) -> next mask
    for _name, masks in train_charts:
        if not masks:
            continue
        for variant in (masks, [_mirror(mask) for mask in masks]):
            by_size_init[variant[0].bit_count()][variant[0]] += 1
            for prev, cur in zip(variant, variant[1:]):
                size = cur.bit_count()
                transitions[(prev, size)][cur] += 1

    init = {}
    for size in range(1, 5):
        init[str(size)] = _normalize(by_size_init[size], MASKS_BY_SIZE[size])
    trans = {}
    for (prev, size), counter in transitions.items():
        trans[f"{prev}|{size}"] = _normalize(counter, MASKS_BY_SIZE[size])

    # held-out accuracy / log-loss
    hits = total = 0
    log_loss = 0.0
    for _name, masks in val_charts:
        for prev, cur in zip(masks, masks[1:]):
            size = cur.bit_count()
            dist = trans.get(f"{prev}|{size}")
            if not dist:
                dist = init[str(size)]
            best = max(dist, key=dist.get)
            hits += int(int(best) == cur)
            total += 1
            log_loss -= float(np.log(max(dist.get(str(cur), 1e-6), 1e-6)))

    return {
        "init": init,
        "trans": trans,
        "val_acc": hits / max(1, total),
        "val_logloss": log_loss / max(1, total),
        "train_charts": len(train_charts),
        "val_charts": len(val_charts),
    }


def main() -> None:
    models = {}
    for style, folders in FOLDERS.items():
        charts = []
        for folder in folders:
            charts.extend(_parse_folder(folder))
        if len(charts) < 4:
            print(style, "not enough charts", flush=True)
            continue
        train, val = split_charts(charts)
        model = build_model(style, train, val)
        models[style] = model
        print(f"{style}: charts={len(charts)} val_acc={model['val_acc']:.1%} "
              f"logloss={model['val_logloss']:.3f}", flush=True)
    MODEL_PATH.write_text(json.dumps(models, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    print("saved", MODEL_PATH)



def _blend(dist, extra, key, size, weight):
    if not extra:
        return dist
    community = extra.get(key)
    if not community:
        return dist
    if not dist:
        return community
    merged = {}
    for mask, prob in dist.items():
        merged[str(mask)] = merged.get(str(mask), 0.0) + (1.0 - weight) * float(prob)
    for mask, prob in community.items():
        merged[str(mask)] = merged.get(str(mask), 0.0) + weight * float(prob)
    return merged


def sample_mask_penalized(model: dict, prev_mask, size: int, rng,
                          penalty: float = 0.55, extra=None,
                          extra_weight: float = 0.3) -> int:
    key = f"{prev_mask}|{size}"
    dist = model.get("trans", {}).get(key)
    if not dist:
        dist = model.get("init", {}).get(str(size))
    if not dist:
        return rng.choice(MASKS_BY_SIZE[size])
    dist = _blend(dist, extra, key, size, extra_weight)
    masks = [int(mask) for mask in dist]
    weights = []
    for mask in masks:
        weight = float(dist[str(mask)])
        if prev_mask:
            shared = (mask & prev_mask).bit_count()
            weight *= max(0.04, 1.0 - penalty * shared / max(1, size))
        weights.append(weight)
    if size == 1 and prev_mask:
        blocked = [mask for mask in masks if mask == prev_mask]
        if len(blocked) < len(masks):
            for index, mask in enumerate(masks):
                if mask == prev_mask:
                    weights[index] = 0.0
    return rng.choices(masks, weights=weights, k=1)[0]

def load_models(path: Path = MODEL_PATH) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def sample_mask(model: dict, prev_mask, size: int, rng, extra=None,
                extra_weight: float = 0.2) -> int:
    key = f"{prev_mask}|{size}"
    dist = model.get("trans", {}).get(key)
    if not dist:
        dist = model.get("init", {}).get(str(size))
    if not dist:
        masks = MASKS_BY_SIZE[size]
        return rng.choice(masks)
    dist = _blend(dist, extra, key, size, extra_weight)
    masks = [int(mask) for mask in dist]
    weights = list(dist.values())
    return rng.choices(masks, weights=weights, k=1)[0]


if __name__ == "__main__":
    main()
