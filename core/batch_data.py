"""Convert newly downloaded community charts into the neural dataset.

Scans community_downloads/* for a .mc (native or converted osu) and an audio
file, computes 64-band log-mel + 4-lane labels, appends records to
.screen/neural/index_combined.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .audio import load_audio
from .neural_chart import DATA_DIR, HOP_SEC, chart_rows, log_mel

ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS = ROOT / "community_downloads"
INDEX = DATA_DIR / "index_combined.json"
AUDIO_EXT = (".ogg", ".mp3", ".wav", ".flac")


def build() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    index = json.loads(INDEX.read_text(encoding="utf-8")) if INDEX.exists() else []
    known = {record.get("chart") for record in index}
    added = 0
    for folder in sorted(p for p in DOWNLOADS.iterdir() if p.is_dir()):
        charts = sorted(folder.glob("*.mc"))
        if not charts:
            continue
        chart = charts[0]
        if str(chart) in known:
            continue
        audio = next((p for p in folder.iterdir()
                      if p.suffix.lower() in AUDIO_EXT), None)
        if audio is None:
            continue
        try:
            samples, sr = load_audio(str(audio))
            mel = log_mel(samples, sr)
            data = json.loads(chart.read_text(encoding="utf-8"))
            rows, times = chart_rows(data)
            labels = np.zeros((4, mel.shape[1]), dtype=np.uint8)
            for seconds, mask in rows:
                frame = int(round(seconds / HOP_SEC))
                if 0 <= frame < labels.shape[1]:
                    for lane in range(4):
                        if mask & (1 << lane):
                            labels[lane, frame] = 1
            name = f"download_{folder.name}"[:80]
            np.savez_compressed(DATA_DIR / f"{name}.npz", mel=mel, labels=labels,
                                name=chart.stem)
            bpm = times[0][1] if times else 180.0
            index.append({"file": name, "style": "community", "bpm": bpm,
                          "audio": str(audio), "chart": str(chart),
                          "frames": int(mel.shape[1])})
            added += 1
            print(f"added {folder.name} {mel.shape[1]}", flush=True)
        except Exception as exc:
            print(f"skip {folder.name}: {type(exc).__name__} {exc}", flush=True)
    INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print("added", added, "total", len(index))
    return added


if __name__ == "__main__":
    build()
