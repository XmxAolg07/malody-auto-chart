"""Generate docs/training_charts.md for every chart used by the training corpus.

Sources:
  * the 76 dan charts under `...AAA实例谱子存放处！/{ex,reg} */`
  * the 315 accepted community charts in community_clean/index.json
  * anything downloaded into community_downloads/ (native .mc or converted osu)
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "...AAA实例谱子存放处！"
CLEAN = ROOT / "community_clean" / "index.json"
DOWNLOADS = ROOT / "community_downloads"
OUT = ROOT / "docs" / "training_charts.md"


def read_mc(path: Path) -> dict | None:
    try:
        raw = path.read_bytes()
        if raw[:2] == b"PK":
            with zipfile.ZipFile(path) as zf:
                members = [n for n in zf.namelist() if n.lower().endswith(".mc")]
                if not members:
                    return None
                raw = zf.read(members[0])
        return json.loads(raw.decode("utf-8-sig", errors="replace"))
    except Exception:
        return None


def beat(value) -> float:
    return float(value[0]) + float(value[1]) / float(value[2] or 1)


def metrics(path: Path, data: dict) -> dict:
    times = []
    for item in data.get("time") or []:
        try:
            times.append((beat(item["beat"]), float(item["bpm"])))
        except Exception:
            continue
    times.sort()
    segments, seconds = [], 0.0
    for index, (b, bpm) in enumerate(times):
        if index:
            prev_b, prev_s, prev_bpm = segments[-1]
            seconds = prev_s + (b - prev_b) * 60.0 / prev_bpm
        segments.append((b, seconds, bpm))
    rows = {}
    ln = 0
    for note in data.get("note") or []:
        if isinstance(note, dict):
            beat_list, lane = note.get("beat"), note.get("column")
            end = note.get("endbeat")
        else:
            beat_list, lane = note[0], note[1]
            end = None
        if beat_list is None or lane is None:
            continue
        b = beat(beat_list)
        rows[round(b * 48) / 48] = rows.get(round(b * 48) / 48, 0) | (1 << int(lane))
        if end is not None and beat(end) > b:
            ln += 1
    keys = sorted(rows)
    if not keys:
        return {}
    seg = segments[0]
    for candidate in segments:
        if candidate[0] <= keys[0]:
            seg = candidate
        else:
            break
    start = seg[1] + (keys[0] - seg[0]) * 60.0 / seg[2]
    seg = segments[0]
    for candidate in segments:
        if candidate[0] <= keys[-1]:
            seg = candidate
        else:
            break
    end = seg[1] + (keys[-1] - seg[0]) * 60.0 / seg[2]
    meta = data.get("meta") or {}
    song = meta.get("song") or {}
    return {
        "title": song.get("title") or path.stem,
        "artist": song.get("artist") or "",
        "version": str(meta.get("version") or ""),
        "bpm": round(times[0][1], 2) if times else 0,
        "notes": sum(m.bit_count() for m in rows.values()),
        "rows": len(rows),
        "duration": round(max(0.0, end - start), 1),
        "ln": ln,
    }


def rows_for(entries: list[dict]) -> list[dict]:
    out = []
    for item in entries:
        path = Path(item["path"])
        data = read_mc(path)
        if not data:
            continue
        m = metrics(path, data)
        if not m:
            continue
        out.append({"source": item.get("source", ""), "style": item.get("style", ""),
                    "path": str(path.relative_to(ROOT)), **m})
    return out


def main() -> None:
    entries = []
    for style in ("stream", "speed", "jack", "tech"):
        for kind in ("ex", "reg"):
            folder = CORPUS / f"{kind} {style}" / "0"
            for path in sorted(folder.glob("*.mc")):
                entries.append({"path": str(path), "source": f"dan/{kind}",
                                "style": style})
    if CLEAN.exists():
        for item in json.loads(CLEAN.read_text(encoding="utf-8")):
            entries.append({"path": item["path"], "source": "community_clean",
                            "style": "community"})
    if DOWNLOADS.exists():
        for path in sorted(DOWNLOADS.rglob("*.mc")):
            entries.append({"path": str(path), "source": "malody_download",
                            "style": "community"})
    records = rows_for(entries)
    records.sort(key=lambda r: (r["source"], r["style"], r["title"].lower()))
    lines = ["# 训练谱面清单", "",
             f"共 **{len(records)}** 张。", "",
             "| # | 曲名 | 艺术家 | 难度 | BPM | 音符 | 时长(s) | 来源 | 路径 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for index, r in enumerate(records, 1):
        title = r["title"].replace("|", "/")
        artist = r["artist"].replace("|", "/")
        version = r["version"].replace("|", "/")
        lines.append(
            f"| {index} | {title} | {artist} | {version} | {r['bpm']} | "
            f"{r['notes']} | {r['duration']} | {r['source']}/{r['style']} | `{r['path']}` |")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("charts", len(records), "->", OUT)


if __name__ == "__main__":
    main()
