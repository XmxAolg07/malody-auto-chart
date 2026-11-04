"""One distinct algorithm per style, tuned on the held-out songs.

Assignment chosen by one-to-one matching of validation accuracy:

    Stream -> softmax, Speed -> k-NN, Jack -> MLP, Technical -> NB
"""

from __future__ import annotations

import json

from .row_model import MODEL_PATH, build_dataset, train_softmax
from .row_train import train_knn, train_mlp, train_nb


def main() -> None:
    data = build_dataset()
    chosen = {}

    if "stream" in data:
        x, y, g = data["stream"]
        best = None
        for lr, l2 in ((0.10, 3e-3), (0.08, 5e-3), (0.15, 1e-3)):
            model = train_softmax(x, y, g, epochs=400, lr=lr, l2=l2, verbose=False)
            if model:
                model["type"] = "softmax"
                print(f"stream softmax lr={lr} l2={l2}: {model['val_acc']:.1%}", flush=True)
                if best is None or model["val_acc"] > best["val_acc"]:
                    best = model
        chosen["stream"] = best

    if "speed" in data:
        x, y, g = data["speed"]
        best = None
        for k, prototypes in ((20, 2500), (30, 2500), (40, 3000)):
            model = train_knn(x, y, g, k=k, prototypes=prototypes, verbose=False)
            print(f"speed knn k={k} p={prototypes}: {model['val_acc']:.1%}", flush=True)
            if best is None or model["val_acc"] > best["val_acc"]:
                best = model
        chosen["speed"] = best

    if "jack" in data:
        x, y, g = data["jack"]
        best = None
        for hidden, epochs in ((24, 400), (32, 600), (40, 600)):
            model = train_mlp(x, y, g, hidden=hidden, epochs=epochs, verbose=False)
            print(f"jack mlp h={hidden} e={epochs}: {model['val_acc']:.1%}", flush=True)
            if best is None or model["val_acc"] > best["val_acc"]:
                best = model
        chosen["jack"] = best

    if "tech" in data:
        x, y, g = data["tech"]
        model = train_nb(x, y, g, verbose=False)
        print(f"tech nb: {model['val_acc']:.1%}", flush=True)
        chosen["tech"] = model

    MODEL_PATH.write_text(json.dumps(chosen, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    for style, model in chosen.items():
        print(style, model.get("type"), f"{model.get('val_acc', 0):.1%}")
    print("saved", MODEL_PATH)


if __name__ == "__main__":
    main()
