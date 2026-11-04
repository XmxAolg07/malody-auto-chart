"""Row-timing model: decide which audio onsets become chart rows.

Candidates are spectral-flux peaks.  A peak is labelled positive when it is
within 45 ms of a row in the reference chart.  Features are local audio/beat
context only (no song id, no absolute time).  Four binary classifiers are
trained per style and the best F1 on held-out songs wins.
"""

from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np

from .audio import load_audio
from .energy import EnergyMap
from .row_model import chart_rows

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "...AAA实例谱子存放处！"
if not (CORPUS / "ex stream").exists():
    CORPUS = ROOT
CACHE_DIR = ROOT / ".screen"
MODEL_PATH = ROOT / "assets" / "timing_models.json"
FOLDERS = {
    "stream": ["ex stream/0", "reg stream/0"],
    "speed": ["ex speed/0", "reg speed/0"],
    "jack": ["ex jack/0", "reg jack/0"],
    "tech": ["ex tech/0", "reg tech/0"],
}
FEATURES = [
    "strength", "onset", "low", "energy", "section", "beat_phase",
    "bar_phase", "bpm", "gap_prev", "gap_next", "density", "rank",
]


def _feature_row(energy, times, strengths, index, beat, bpm):
    t = times[index]
    strength = strengths[index]
    gap_prev = t - times[index - 1] if index > 0 else 0.5
    gap_next = times[index + 1] - t if index + 1 < len(times) else 0.5
    lo = bisect.bisect_left(times, t - 2.0)
    hi = bisect.bisect_right(times, t + 2.0)
    local = strengths[lo:hi]
    rank = sum(1 for s in local if s <= strength) / max(1, len(local))
    return [
        strength,
        min(1.0, energy.onset_at(t) * 4.0),
        min(1.0, energy.low_at(t) * 4.0),
        min(2.0, energy.ratio_at(t)),
        min(2.0, energy.section_ratio(t)),
        beat - int(beat),
        (beat / 4.0) - int(beat / 4.0),
        bpm / 180.0,
        min(0.5, gap_prev),
        min(0.5, gap_next),
        min(8.0, energy.peak_density(t, 1.0)),
        rank,
    ]


def build_dataset(force: bool = False) -> dict:
    arrays = {}
    for style, folders in FOLDERS.items():
        cache = CACHE_DIR / f"timing_{style}.npz"
        if cache.exists() and not force:
            data = np.load(cache)
            arrays[style] = (data["x"], data["y"], data["g"])
            print(style, "cached", len(data["y"]), flush=True)
            continue
        feats, labels, groups = [], [], []
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
                if len(rows) < 80:
                    continue
                samples, sr = load_audio(str(audio))
                energy = EnergyMap(samples, sr)
                times = list(energy._peak_times)
                strengths = list(energy._peak_strength)
                if not times:
                    continue
                ref_times = [row[1] for row in rows]
                # beat for each peak: interpolate from reference rows (beat, sec)
                beats = [row[0] for row in rows]
                bpm = rows[len(rows) // 2][3]
                print(f"{style} chart {chart_id + 1}: {audio.stem[:40]} "
                      f"peaks={len(times)} rows={len(rows)}", flush=True)
                for index, t in enumerate(times):
                    pos = bisect.bisect_left(ref_times, t)
                    near = min([abs(ref_times[j] - t) for j in (pos - 1, pos)
                                if 0 <= j < len(ref_times)] or [9.0])
                    label = 1 if near <= 0.045 else 0
                    ref_pos = bisect.bisect_left(ref_times, t)
                    if 0 <= ref_pos < len(ref_times):
                        beat = beats[ref_pos]
                    elif ref_pos > 0:
                        beat = beats[min(ref_pos - 1, len(beats) - 1)]
                    else:
                        beat = 0.0
                    feats.append(_feature_row(energy, times, strengths, index, beat, bpm))
                    labels.append(label)
                    groups.append(chart_id)
                chart_id += 1
        if feats:
            np.savez(cache, x=np.array(feats, dtype=np.float32),
                     y=np.array(labels, dtype=np.int64),
                     g=np.array(groups, dtype=np.int64))
            arrays[style] = (np.array(feats, dtype=np.float32),
                             np.array(labels, dtype=np.int64),
                             np.array(groups, dtype=np.int64))
            print(style, "built", len(labels), "peaks", chart_id, "charts", flush=True)
    return arrays


def split_by_song(groups, val_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    rng.shuffle(unique)
    val_groups = set(unique[: max(1, int(len(unique) * val_ratio))].tolist())
    val = np.array([g in val_groups for g in groups])
    return np.where(~val)[0], np.where(val)[0]


def f1_at(prob, y, threshold):
    pred = prob >= threshold
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return f1, precision, recall


def best_threshold(prob, y):
    best = (0.0, 0.5, 0.0, 0.0)
    for threshold in np.arange(0.2, 0.81, 0.02):
        f1, precision, recall = f1_at(prob, y, threshold)
        if f1 > best[0]:
            best = (f1, float(threshold), precision, recall)
    return best


def train_logistic(x, y, groups, epochs=300, lr=0.2, l2=1e-3):
    train_idx, val_idx = split_by_song(groups)
    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-6
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    w = np.zeros(x_train.shape[1])
    b = 0.0
    best = (0.0, 0.0, w.copy(), b)
    for epoch in range(epochs):
        logits = x_train @ w + b
        prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        grad_w = x_train.T @ (prob - y_train) / len(x_train) + l2 * w
        grad_b = float((prob - y_train).mean())
        w -= lr * grad_w
        b -= lr * grad_b
        if (epoch + 1) % 50 == 0:
            val_prob = 1.0 / (1.0 + np.exp(-np.clip(x_val @ w + b, -30, 30)))
            f1, threshold, precision, recall = best_threshold(val_prob, y_val)
            if f1 > best[0]:
                best = (f1, threshold, w.copy(), b)
    f1, threshold, w, b = best
    return {"type": "logistic", "mean": mean.tolist(), "std": std.tolist(),
            "w": w.tolist(), "b": b, "f1": f1, "threshold": threshold}


def train_mlp(x, y, groups, epochs=300, hidden=32, lr=0.08, l2=1e-4):
    train_idx, val_idx = split_by_song(groups)
    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-6
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    rng = np.random.default_rng(3)
    w1 = rng.normal(0, np.sqrt(2.0 / x_train.shape[1]), (x_train.shape[1], hidden))
    b1 = np.zeros(hidden)
    w2 = rng.normal(0, np.sqrt(1.0 / hidden), hidden)
    b2 = 0.0
    best = (0.0, 0.0, w1.copy(), b1.copy(), w2.copy(), b2)
    for epoch in range(epochs):
        h = np.maximum(0.0, x_train @ w1 + b1)
        logits = h @ w2 + b2
        prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        delta = (prob - y_train) / len(x_train)
        grad_w2 = h.T @ delta + l2 * w2
        grad_b2 = float(delta.sum())
        grad_h = np.outer(delta, w2)
        grad_h[h <= 0] = 0.0
        grad_w1 = x_train.T @ grad_h + l2 * w1
        grad_b1 = grad_h.sum(axis=0)
        w1 -= lr * grad_w1
        b1 -= lr * grad_b1
        w2 -= lr * grad_w2
        b2 -= lr * grad_b2
        if (epoch + 1) % 50 == 0:
            h_val = np.maximum(0.0, x_val @ w1 + b1)
            val_prob = 1.0 / (1.0 + np.exp(-np.clip(h_val @ w2 + b2, -30, 30)))
            f1, threshold, _, _ = best_threshold(val_prob, y_val)
            if f1 > best[0]:
                best = (f1, threshold, w1.copy(), b1.copy(), w2.copy(), b2)
    f1, threshold, w1, b1, w2, b2 = best
    return {"type": "mlp", "mean": mean.tolist(), "std": std.tolist(),
            "w1": w1.tolist(), "b1": b1.tolist(), "w2": w2.tolist(), "b2": b2,
            "f1": f1, "threshold": threshold}


def train_knn(x, y, groups, prototypes=2500, k=35):
    train_idx, val_idx = split_by_song(groups)
    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    rng = np.random.default_rng(4)
    if len(x_train) > prototypes:
        chosen = rng.choice(len(x_train), prototypes, replace=False)
        proto_x, proto_y = x_train[chosen], y_train[chosen]
    else:
        proto_x, proto_y = x_train, y_train
    positives = (proto_y == 1).sum()
    base_rate = positives / len(proto_y)
    probs = []
    for start in range(0, len(x_val), 512):
        chunk = x_val[start:start + 512]
        distances = (chunk ** 2).sum(axis=1)[:, None] + (proto_x ** 2).sum(axis=1)[None, :] \
            - 2.0 * chunk @ proto_x.T
        nearest = np.argpartition(distances, k, axis=1)[:, :k]
        for row in range(len(chunk)):
            idx = nearest[row]
            weights = 1.0 / (np.sqrt(np.maximum(distances[row, idx], 1e-9)) + 1e-3)
            probs.append(float((weights * proto_y[idx]).sum() / weights.sum()))
    f1, threshold, _, _ = best_threshold(np.array(probs), y_val)
    return {"type": "knn", "x": proto_x.tolist(), "y": proto_y.tolist(),
            "k": k, "f1": f1, "threshold": threshold, "base_rate": base_rate}


def train_nb(x, y, groups):
    train_idx, val_idx = split_by_song(groups)
    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-6
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    pos, neg = x_train[y_train == 1], x_train[y_train == 0]
    mu1, mu0 = pos.mean(axis=0), neg.mean(axis=0)
    va1, va0 = pos.var(axis=0) + 1e-4, neg.var(axis=0) + 1e-4
    prior = len(pos) / len(x_train)
    log_pos = (-0.5 * (np.log(2 * np.pi * va1) + (x_val - mu1) ** 2 / va1).sum(axis=1)
               + np.log(prior))
    log_neg = (-0.5 * (np.log(2 * np.pi * va0) + (x_val - mu0) ** 2 / va0).sum(axis=1)
               + np.log(1 - prior))
    prob = 1.0 / (1.0 + np.exp(-(log_pos - log_neg)))
    f1, threshold, _, _ = best_threshold(prob, y_val)
    return {"type": "nb", "mean": mean.tolist(), "std": std.tolist(),
            "mu1": mu1.tolist(), "mu0": mu0.tolist(),
            "va1": va1.tolist(), "va0": va0.tolist(), "prior": prior,
            "f1": f1, "threshold": threshold}



def predict_timing(model: dict, features) -> float:
    mean = np.array(model["mean"], dtype=np.float64) if "mean" in model else None
    std = np.array(model["std"], dtype=np.float64) if "std" in model else None
    x = np.array(features, dtype=np.float64)
    if mean is not None:
        x = (x - mean) / std
    kind = model.get("type", "logistic")
    if kind == "logistic":
        z = float(x @ np.array(model["w"]) + model["b"])
    elif kind == "mlp":
        h = np.maximum(0.0, x @ np.array(model["w1"]) + np.array(model["b1"]))
        z = float(h @ np.array(model["w2"]) + model["b2"])
    elif kind == "nb":
        mu1 = np.array(model["mu1"]); mu0 = np.array(model["mu0"])
        va1 = np.array(model["va1"]); va0 = np.array(model["va0"])
        prior = float(model["prior"])
        lp = (-0.5 * (np.log(2 * np.pi * va1) + (x - mu1) ** 2 / va1).sum()
              + np.log(prior))
        ln = (-0.5 * (np.log(2 * np.pi * va0) + (x - mu0) ** 2 / va0).sum()
              + np.log(1 - prior))
        z = lp - ln
    else:
        proto_x = np.array(model["x"]); proto_y = np.array(model["y"])
        distances = ((x[None, :] - proto_x) ** 2).sum(axis=1)
        k = int(model["k"])
        nearest = np.argpartition(distances, k)[:k]
        weights = 1.0 / (np.sqrt(np.maximum(distances[nearest], 1e-9)) + 1e-3)
        z = float((weights * proto_y[nearest]).sum() / weights.sum())
        return z
    return 1.0 / (1.0 + np.exp(-max(-30.0, min(30.0, z))))


def load_models(path: Path = MODEL_PATH) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

def main() -> None:
    data = build_dataset()
    best_models = {}
    for style, (x, y, groups) in data.items():
        print(f"=== timing {style}: peaks={len(y)} positives={int(y.sum())} ===", flush=True)
        candidates = []
        for name, fn in (("logistic", train_logistic), ("mlp", train_mlp),
                         ("nb", train_nb), ("knn", train_knn)):
            model = fn(x, y, groups)
            candidates.append(model)
            print(f"  {name}: f1={model['f1']:.1%} thr={model['threshold']:.2f}",
                  flush=True)
        best = max(candidates, key=lambda item: item["f1"])
        best_models[style] = best
        print(f"  WINNER {style}: {best['type']} f1={best['f1']:.1%}\n", flush=True)
    MODEL_PATH.write_text(json.dumps(best_models, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    print("saved", MODEL_PATH)


if __name__ == "__main__":
    main()
