"""Build the combined neural dataset from the filtered community corpus.

Reads ``community_clean/index.json`` (accepted charts), computes 64-band
log-mel at 10ms and 4-lane piano-roll labels, writes ``.screen/neural/*.npz``
and a combined index used by the diffusion pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .audio import load_audio
from .neural_chart import DATA_DIR, HOP_SEC, chart_rows, log_mel

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "community_clean" / "index.json"
COMBINED = DATA_DIR / "index_combined.json"


def build(force: bool = False) -> list[dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    base = json.loads((DATA_DIR / "index.json").read_text(encoding="utf-8"))
    community = json.loads(INDEX.read_text(encoding="utf-8"))
    output = list(base)
    for index, record in enumerate(community):
        name = f"community_{index:03d}"
        npz = DATA_DIR / f"{name}.npz"
        if npz.exists() and not force:
            output.append({"file": name, "style": "community",
                           "bpm": record["bpm"], "audio": record["audio"],
                           "chart": record["path"],
                           "frames": int(np.load(npz)["mel"].shape[1])})
            continue
        audio = record["audio"]
        if not audio or not Path(audio).exists():
            continue
        try:
            samples, sr = load_audio(audio)
            mel = log_mel(samples, sr)
            data = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
            rows, times = chart_rows(data)
            labels = np.zeros((4, mel.shape[1]), dtype=np.uint8)
            for seconds, mask in rows:
                frame = int(round(seconds / HOP_SEC))
                if 0 <= frame < labels.shape[1]:
                    for lane in range(4):
                        if mask & (1 << lane):
                            labels[lane, frame] = 1
            np.savez_compressed(npz, mel=mel, labels=labels, name=Path(record["path"]).stem)
            output.append({"file": name, "style": "community",
                           "bpm": record["bpm"], "audio": audio,
                           "chart": record["path"], "frames": int(mel.shape[1])})
            print(f"{index:03d}/{len(community)} {Path(record['path']).stem[:40]} "
                  f"{mel.shape[1]}", flush=True)
        except Exception as exc:
            print(f"{index:03d} ERR {record['path']}: {exc}", flush=True)
    COMBINED.write_text(json.dumps(output, ensure_ascii=False, indent=1), encoding="utf-8")
    print("records", len(output), "saved", COMBINED)
    return output


if __name__ == "__main__":
    build()
