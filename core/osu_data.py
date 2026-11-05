"""Add converted osu! 4K charts (community_osu_mc/index.json) to the neural dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .audio import load_audio
from .neural_chart import DATA_DIR, HOP_SEC, chart_rows, log_mel

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "community_osu_mc" / "index.json"
COMBINED = DATA_DIR / "index_combined.json"


def build() -> int:
    if not INDEX.exists():
        print("no osu index")
        return 0
    osu_records = json.loads(INDEX.read_text(encoding="utf-8"))
    combined = json.loads(COMBINED.read_text(encoding="utf-8")) if COMBINED.exists() else []
    known = {record.get("chart") for record in combined}
    added = 0
    mirror_pairs = []
    mirror_root = ROOT / "osu_mirror"
    if mirror_root.exists():
        for chart in sorted(mirror_root.rglob("*.mc")):
            audio = next((p for p in chart.parent.iterdir()
                          if p.suffix.lower() in (".mp3", ".ogg", ".wav")), None)
            if audio:
                mirror_pairs.append({"mc": str(chart), "audio": str(audio),
                                     "title": chart.stem, "artist": "", "version": ""})
    for index, item in enumerate(list(osu_records) + mirror_pairs):
        chart_path = Path(item["mc"])
        if not chart_path.is_absolute():
            chart_path = ROOT / chart_path
        audio_path = Path(item["audio"]) if item.get("audio") else None
        if audio_path is None or not audio_path.is_file():
            continue
        if str(chart_path) in known:
            continue
        try:
            samples, sr = load_audio(str(audio_path))
            mel = log_mel(samples, sr)
            if mel.shape[1] < 1500:
                continue
            data = json.loads(chart_path.read_text(encoding="utf-8"))
            rows, times = chart_rows(data)
            labels = np.zeros((4, mel.shape[1]), dtype=np.uint8)
            for seconds, mask in rows:
                frame = int(round(seconds / HOP_SEC))
                if 0 <= frame < labels.shape[1]:
                    for lane in range(4):
                        if mask & (1 << lane):
                            labels[lane, frame] = 1
            name = f"osu_{index:04d}"
            np.savez_compressed(DATA_DIR / f"{name}.npz", mel=mel, labels=labels,
                                name=chart_path.stem)
            combined.append({"file": name, "style": "community",
                             "bpm": times[0][1] if times else 180.0,
                             "audio": str(audio_path), "chart": str(chart_path),
                             "frames": int(mel.shape[1])})
            added += 1
            if added % 20 == 0:
                print("added", added, flush=True)
        except Exception as exc:
            print("skip", index, type(exc).__name__, exc, flush=True)
    COMBINED.write_text(json.dumps(combined, ensure_ascii=False, indent=1), encoding="utf-8")
    print("added", added, "total", len(combined))
    return added


if __name__ == "__main__":
    build()
