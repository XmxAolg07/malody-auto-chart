"""Try four classifiers per key-pattern style and keep the best on held-out songs.

Algorithms: softmax regression, one-hidden-layer MLP, Gaussian naive Bayes and
distance-weighted k-NN.  Every model uses the same song-level split, so the
comparison is fair.  The winner per style is written to assets/row_models.json
and loaded by chartgen through ``predict_size``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .row_model import (FEATURES, MODEL_PATH, build_dataset, softmax,
                        train_softmax)

ROOT = Path(__file__).resolve().parent.parent
CLASSES = 4


def split_by_song(groups, val_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    rng.shuffle(unique)
    val_groups = set(unique[: max(1, int(len(unique) * val_ratio))].tolist())
    val = np.array([group in val_groups for group in groups])
    return np.where(~val)[0], np.where(val)[0]


def _standardize(x_train, x_val):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-6
    return (x_train - mean) / std, (x_val - mean) / std, mean, std


def train_mlp(x, y, groups, epochs=300, hidden=24, lr=0.05, l2=1e-4, verbose=True):
    train_idx, val_idx = split_by_song(groups)
    x_train, x_val, mean, std = _standardize(x[train_idx], x[val_idx])
    y_train, y_val = y[train_idx], y[val_idx]
    rng = np.random.default_rng(1)
    w1 = rng.normal(0, np.sqrt(2.0 / x_train.shape[1]),
                    (x_train.shape[1], hidden)).astype(np.float64)
    b1 = np.zeros(hidden)
    w2 = rng.normal(0, np.sqrt(1.0 / hidden), (hidden, CLASSES)).astype(np.float64)
    b2 = np.zeros(CLASSES)
    onehot = np.eye(CLASSES)[y_train]
    best = (0.0, w1.copy(), b1.copy(), w2.copy(), b2.copy())
    for epoch in range(epochs):
        hidden_out = np.maximum(0.0, x_train @ w1 + b1)
        probs = softmax(hidden_out @ w2 + b2)
        delta = (probs - onehot) / len(x_train)
        grad_w2 = hidden_out.T @ delta + l2 * w2
        grad_b2 = delta.sum(axis=0)
        grad_hidden = delta @ w2.T
        grad_hidden[hidden_out <= 0] = 0.0
        grad_w1 = x_train.T @ grad_hidden + l2 * w1
        grad_b1 = grad_hidden.sum(axis=0)
        w1 -= lr * grad_w1
        b1 -= lr * grad_b1
        w2 -= lr * grad_w2
        b2 -= lr * grad_b2
        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            val_prob = softmax(np.maximum(0.0, x_val @ w1 + b1) @ w2 + b2)
            val_acc = float((val_prob.argmax(axis=1) == y_val).mean())
            train_acc = float((probs.argmax(axis=1) == y_train).mean())
            if verbose:
                print(f"    mlp epoch {epoch + 1:3d}: train={train_acc:.1%} val={val_acc:.1%}",
                      flush=True)
            if val_acc > best[0]:
                best = (val_acc, w1.copy(), b1.copy(), w2.copy(), b2.copy())
    val_acc, w1, b1, w2, b2 = best
    return {
        "type": "mlp",
        "features": FEATURES,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "w1": w1.tolist(), "b1": b1.tolist(),
        "w2": w2.tolist(), "b2": b2.tolist(),
        "val_acc": val_acc,
    }


def train_nb(x, y, groups, verbose=True):
    train_idx, val_idx = split_by_song(groups)
    x_train, x_val, mean, std = _standardize(x[train_idx], x[val_idx])
    y_train, y_val = y[train_idx], y[val_idx]
    means, variances, priors = [], [], []
    for cls in range(CLASSES):
        part = x_train[y_train == cls]
        means.append(part.mean(axis=0))
        variances.append(part.var(axis=0) + 1e-4)
        priors.append(len(part) / len(x_train))
    means = np.array(means)
    variances = np.array(variances)
    priors = np.array(priors)

    def log_prob(features):
        diff = features[:, None, :] - means[None, :, :]
        logp = -0.5 * (np.log(2 * np.pi * variances)[None, :, :]
                       + diff ** 2 / variances[None, :, :]).sum(axis=2)
        return logp + np.log(priors)[None, :]

    val_acc = float((log_prob(x_val).argmax(axis=1) == y_val).mean())
    if verbose:
        print(f"    nb: val={val_acc:.1%}", flush=True)
    return {
        "type": "nb",
        "features": FEATURES,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "means": means.tolist(),
        "vars": variances.tolist(),
        "priors": priors.tolist(),
        "val_acc": val_acc,
    }


def train_knn(x, y, groups, prototypes=1800, k=25, verbose=True):
    train_idx, val_idx = split_by_song(groups)
    x_train, x_val, mean, std = _standardize(x[train_idx], x[val_idx])
    y_train, y_val = y[train_idx], y[val_idx]
    rng = np.random.default_rng(2)
    if len(x_train) > prototypes:
        chosen = rng.choice(len(x_train), prototypes, replace=False)
        proto_x, proto_y = x_train[chosen], y_train[chosen]
    else:
        proto_x, proto_y = x_train, y_train
    # batched prediction to keep memory bounded
    correct = 0
    for start in range(0, len(x_val), 512):
        chunk = x_val[start:start + 512]
        distances = (chunk ** 2).sum(axis=1)[:, None] + (proto_x ** 2).sum(axis=1)[None, :] \
            - 2.0 * chunk @ proto_x.T
        nearest = np.argpartition(distances, k, axis=1)[:, :k]
        votes = np.zeros((len(chunk), CLASSES))
        for row in range(len(chunk)):
            idx = nearest[row]
            weights = 1.0 / (np.sqrt(np.maximum(distances[row, idx], 1e-9)) + 1e-3)
            for index, weight in zip(idx, weights):
                votes[row, proto_y[index]] += weight
        correct += int((votes.argmax(axis=1) == y_val[start:start + 512]).sum())
    val_acc = correct / len(y_val)
    if verbose:
        print(f"    knn: val={val_acc:.1%}", flush=True)
    return {
        "type": "knn",
        "features": FEATURES,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "x": proto_x.tolist(),
        "y": proto_y.tolist(),
        "k": k,
        "val_acc": val_acc,
    }


def main() -> None:
    data = build_dataset()
    best_models = {}
    for style, (x, y, groups) in data.items():
        print(f"=== {style}: rows={len(y)} ===", flush=True)
        candidates = []
        for lr, l2 in ((0.10, 1e-3), (0.08, 3e-3), (0.15, 3e-4)):
            model = train_softmax(x, y, groups, epochs=250, lr=lr, l2=l2, verbose=False)
            if model:
                model["type"] = "softmax"
                candidates.append(model)
                print(f"  softmax lr={lr} l2={l2}: val={model['val_acc']:.1%}", flush=True)
        for hidden, epochs in ((24, 300), (40, 400)):
            model = train_mlp(x, y, groups, hidden=hidden, epochs=epochs, verbose=False)
            candidates.append(model)
            print(f"  mlp hidden={hidden} epochs={epochs}: val={model['val_acc']:.1%}",
                  flush=True)
        candidates.append(train_nb(x, y, groups, verbose=False))
        print(f"  nb: val={candidates[-1]['val_acc']:.1%}", flush=True)
        for k, prototypes in ((15, 1800), (30, 2200)):
            model = train_knn(x, y, groups, k=k, prototypes=prototypes, verbose=False)
            candidates.append(model)
            print(f"  knn k={k} prototypes={prototypes}: val={model['val_acc']:.1%}",
                  flush=True)
        best = max(candidates, key=lambda item: item["val_acc"])
        best_models[style] = best
        print(f"  WINNER {style}: {best['type']} val={best['val_acc']:.1%}\n", flush=True)
    MODEL_PATH.write_text(json.dumps(best_models, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    print("saved", MODEL_PATH)


if __name__ == "__main__":
    main()
