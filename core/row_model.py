"""Train a generalising row-size model from the 78 chart/audio pairs.

No song identity, no absolute time: features are only local audio/beat context.
Trained per style with a softmax classifier (numpy), split by **song** so the
validation score measures generalisation, not memorisation.

Run ``python -m core.row_model`` to rebuild the cache, train variants and write
``assets/row_models.json``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .audio import load_audio
from .energy import EnergyMap

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "...AAA实例谱子存放处！"
if not (CORPUS / "ex stream").exists():
    CORPUS = ROOT
CACHE_DIR = ROOT / ".screen"
MODEL_PATH = ROOT / "assets" / "row_models.json"
FOLDERS = {
    "stream": ["ex stream/0", "reg stream/0"],
    "speed": ["ex speed/0", "reg speed/0"],
    "jack": ["ex jack/0", "reg jack/0"],
    "tech": ["ex tech/0", "reg tech/0"],
}
FEATURES = [
    "energy", "section", "onset", "low", "energy_slope", "section_slope",
    "peak_strength", "peak_dist", "peak_density", "beat_phase", "bar_phase",
    "bpm", "prev1", "prev2", "prev3",
]


def _beat_float(beat) -> float:
    if len(beat) == 1:
        return float(beat[0])
    return float(beat[0]) + float(beat[1]) / float(beat[2] or 1)


def chart_rows(chart: dict):
    times = []
    for item in chart.get("time") or []:
        try:
            times.append((_beat_float(item["beat"]), float(item["bpm"])))
        except Exception:
            continue
    times.sort()
    if not times:
        times = [(0.0, 180.0)]
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
        beat = _beat_float(beat_list)
        segment = segments[0]
        for candidate in segments:
            if candidate[0] <= beat + 1e-9:
                segment = candidate
            else:
                break
        seconds = segment[1] + (beat - segment[0]) * 60.0 / segment[2]
        key = round(beat * 48) / 48
        rows[key] = rows.get(key, 0) | (1 << int(lane))
    ordered = []
    for beat in sorted(rows):
        segment = segments[0]
        for candidate in segments:
            if candidate[0] <= beat + 1e-9:
                segment = candidate
            else:
                break
        seconds = segment[1] + (beat - segment[0]) * 60.0 / segment[2]
        ordered.append((beat, seconds, rows[beat], segment[2]))
    return ordered


def build_dataset(force: bool = False) -> dict:
    arrays = {}
    for style, folders in FOLDERS.items():
        cache = CACHE_DIR / f"row_{style}.npz"
        if cache.exists() and not force:
            data = np.load(cache)
            arrays[style] = (data["x"], data["y"], data["g"])
            print(style, "cached rows", len(data["y"]), flush=True)
            continue
        feats, targets, groups = [], [], []
        chart_id = 0
        for folder in folders:
            base = CORPUS / folder
            for path in sorted(base.glob("*.mc")):
                data = json.loads(path.read_text(encoding="utf-8"))
                background = (data.get("meta") or {}).get("background") or ""
                audio = base / (Path(background).stem + ".ogg")
                if not audio.exists():
                    continue
                rows = chart_rows(data)
                samples, sr = load_audio(str(audio))
                energy = EnergyMap(samples, sr)
                print(f"{style} chart {chart_id + 1}: {audio.stem[:44]} rows={len(rows)}",
                      flush=True)
                prev1 = prev2 = prev3 = 0.0
                for beat, seconds, mask, bpm in rows:
                    size = mask.bit_count()
                    peak = energy.nearest_peak(seconds, 0.12)
                    peak_strength = peak[1] if peak else 0.0
                    peak_dist = abs(peak[0] - seconds) if peak else 0.12
                    feats.append([
                        energy.ratio_at(seconds),
                        energy.section_ratio(seconds),
                        energy.onset_at(seconds),
                        energy.low_at(seconds),
                        energy.ratio_at(seconds + 0.1) - energy.ratio_at(seconds - 0.1),
                        energy.section_ratio(seconds + 1.0) - energy.section_ratio(seconds - 1.0),
                        peak_strength,
                        peak_dist,
                        energy.peak_density(seconds, 1.0),
                        beat - int(beat),
                        (beat / 4.0) - int(beat / 4.0),
                        bpm / 180.0,
                        prev1,
                        prev2,
                        prev3,
                    ])
                    targets.append(size - 1)
                    groups.append(chart_id)
                    prev3, prev2, prev1 = prev2, prev1, size / 4.0
                chart_id += 1
        if feats:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, x=np.array(feats, dtype=np.float32),
                     y=np.array(targets, dtype=np.int64),
                     g=np.array(groups, dtype=np.int64))
            arrays[style] = (np.array(feats, dtype=np.float32),
                             np.array(targets, dtype=np.int64),
                             np.array(groups, dtype=np.int64))
            print(style, "built rows", len(targets), "charts", chart_id, flush=True)
    return arrays


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def train_softmax(x, y, groups, epochs=400, lr=0.15, l2=1e-3,
                  val_ratio=0.2, seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    rng.shuffle(unique)
    val_groups = set(unique[: max(1, int(len(unique) * val_ratio))].tolist())
    val = np.array([g in val_groups for g in groups])
    x_train, y_train = x[~val], y[~val]
    x_val, y_val, g_val = x[val], y[val], groups[val]
    if len(x_train) < 50 or len(x_val) < 20:
        return None

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-6
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    classes = 4
    weights = np.ones(classes)
    counts = np.bincount(y_train, minlength=classes).astype(np.float64)
    weights = 1.0 / np.sqrt(counts + 1.0)
    weights /= weights.mean()

    w = np.zeros((x_train.shape[1], classes), dtype=np.float64)
    b = np.zeros(classes, dtype=np.float64)
    y_onehot = np.eye(classes)[y_train]
    sample_weight = weights[y_train][:, None]
    best = (0.0, w.copy(), b.copy())
    history = []
    for epoch in range(epochs):
        probs = softmax(x_train @ w + b)
        grad_w = x_train.T @ ((probs - y_onehot) * sample_weight) / len(x_train) + l2 * w
        grad_b = ((probs - y_onehot) * sample_weight).mean(axis=0)
        w -= lr * grad_w
        b -= lr * grad_b
        if (epoch + 1) % 25 == 0 or epoch == epochs - 1:
            train_acc = (probs.argmax(axis=1) == y_train).mean()
            val_probs = softmax(x_val @ w + b)
            val_acc = (val_probs.argmax(axis=1) == y_val).mean()
            history.append((epoch + 1, train_acc, val_acc))
            if verbose:
                print(f"    epoch {epoch + 1:3d}: train={train_acc:.1%} val={val_acc:.1%}",
                      flush=True)
            if val_acc > best[0]:
                best = (val_acc, w.copy(), b.copy())
    val_acc, best_w, best_b = best
    return {
        "features": FEATURES,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "w": best_w.tolist(),
        "b": best_b.tolist(),
        "train_acc": float(history[-1][1]) if history else 0.0,
        "val_acc": float(val_acc),
        "history": history,
        "val_groups": sorted(val_groups),
    }


def per_chart_accuracy(x, y, groups, model):
    mean = np.array(model["mean"])
    std = np.array(model["std"])
    w = np.array(model["w"])
    b = np.array(model["b"])
    probs = softmax(((x - mean) / std) @ w + b)
    pred = probs.argmax(axis=1)
    scores = {}
    for group in np.unique(groups):
        mask = groups == group
        scores[int(group)] = float((pred[mask] == y[mask]).mean())
    return scores



def load_models(path: Path = MODEL_PATH) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def predict_size(model: dict, features) -> int:
    mean = np.array(model["mean"], dtype=np.float64)
    std = np.array(model["std"], dtype=np.float64)
    x = (np.array(features, dtype=np.float64) - mean) / std
    kind = model.get("type", "softmax")
    if kind == "mlp":
        w1 = np.array(model["w1"]); b1 = np.array(model["b1"])
        w2 = np.array(model["w2"]); b2 = np.array(model["b2"])
        logits = np.maximum(0.0, x @ w1 + b1) @ w2 + b2
    elif kind == "nb":
        means = np.array(model["means"]); variances = np.array(model["vars"])
        priors = np.array(model["priors"])
        diff = x[None, :] - means
        logp = -0.5 * (np.log(2 * np.pi * variances)
                       + diff ** 2 / variances).sum(axis=1) + np.log(priors)
        return int(np.argmax(logp)) + 1
    elif kind == "knn":
        proto_x = np.array(model["x"]); proto_y = np.array(model["y"])
        distances = ((x[None, :] - proto_x) ** 2).sum(axis=1)
        k = int(model["k"])
        nearest = np.argpartition(distances, k)[:k]
        votes = np.zeros(4)
        for index in nearest:
            votes[proto_y[index]] += 1.0 / (np.sqrt(max(distances[index], 1e-9)) + 1e-3)
        return int(np.argmax(votes)) + 1
    else:
        w = np.array(model["w"]); b = np.array(model["b"])
        logits = x @ w + b
    return int(np.argmax(logits)) + 1


def main() -> None:
    data = build_dataset()
    models = {}
    for style, (x, y, groups) in data.items():
        print(f"=== training {style} rows={len(y)} ===", flush=True)
        model = train_softmax(x, y, groups, epochs=250, lr=0.10, l2=1e-3,
                              verbose=True)
        if model:
            models[style] = model
            print(f"{style}: train={model['train_acc']:.1%} "
                  f"val={model['val_acc']:.1%} (by song)", flush=True)
    MODEL_PATH.write_text(json.dumps(models, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    print("saved", MODEL_PATH)


if __name__ == "__main__":
    main()
