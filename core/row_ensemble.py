"""Average the probability outputs of softmax, MLP and NB size models.

A small CPU-friendly ensemble: each component is trained on the same song
split, their class probabilities are averaged, and the ensemble replaces the
single model only when held-out accuracy improves.
"""

from __future__ import annotations

import json

import numpy as np

from .row_model import MODEL_PATH, build_dataset, softmax, train_softmax
from .row_train import train_mlp, train_nb


def _proba(model, x_raw):
    mean = np.array(model["mean"], dtype=np.float64)
    std = np.array(model["std"], dtype=np.float64)
    x = (np.asarray(x_raw, dtype=np.float64) - mean) / std
    kind = model["type"]
    if kind == "softmax":
        w = np.array(model["w"]); b = np.array(model["b"])
        return softmax(x @ w + b)
    if kind == "mlp":
        w1 = np.array(model["w1"]); b1 = np.array(model["b1"])
        w2 = np.array(model["w2"]); b2 = np.array(model["b2"])
        hidden = np.maximum(0.0, x @ w1 + b1)
        return softmax(hidden @ w2 + b2)
    if kind == "nb":
        means = np.array(model["means"]); variances = np.array(model["vars"])
        priors = np.array(model["priors"])
        diff = x[:, None, :] - means[None, :, :]
        logp = -0.5 * (np.log(2 * np.pi * variances)
                       + diff ** 2 / variances).sum(axis=2)
        logp += np.log(priors)[None, :]
        logp -= logp.max(axis=1, keepdims=True)
        exp = np.exp(logp)
        return exp / exp.sum(axis=1, keepdims=True)
    raise ValueError(kind)


def main() -> None:
    data = build_dataset()
    current = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    for style, (x, y, groups) in data.items():
        from .row_train import split_by_song
        train_idx, val_idx = split_by_song(groups)
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        components = []
        soft = train_softmax(x_train, y_train, groups[train_idx],
                             epochs=300, lr=0.1, l2=1e-3, verbose=False)
        if soft:
            soft["type"] = "softmax"
            components.append(soft)
        components.append(train_mlp(x_train, y_train, groups[train_idx],
                                    hidden=40, epochs=400, verbose=False))
        components.append(train_nb(x_train, y_train, groups[train_idx], verbose=False))

        total = np.zeros((len(y_val), 4))
        for model in components:
            total += _proba(model, x_val)
        total /= len(components)
        accuracy = float((total.argmax(axis=1) == y_val).mean())
        previous = current.get(style, {}).get("val_acc", 0.0)
        print(f"{style}: ensemble={accuracy:.1%} previous={previous:.1%}", flush=True)
        if accuracy > previous:
            current[style] = {"type": "ensemble", "components": components,
                              "val_acc": accuracy}
    MODEL_PATH.write_text(json.dumps(current, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    print("saved", MODEL_PATH)


if __name__ == "__main__":
    main()
