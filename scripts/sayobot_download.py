"""Download osu!mania 4K charts from the Sayobot mirror.

Flow:
  1. POST https://api.sayobot.cn/?post  {"cmd":"beatmaplist", "mode":3, ...}
  2. GET  https://dl.sayobot.cn/beatmaps/download/novideo/<sid>
  3. extract every Mode:3 / CircleSize:4 .osu + its audio
  4. convert .osu -> Malody .mc via scripts/osu_to_mc.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from osu_to_mc import parse_osu  # noqa: E402

API = "https://api.sayobot.cn/?post"
CDN = "https://dl.sayobot.cn/beatmaps/download/novideo"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "osu_mirror"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://osu.sayobot.cn/"}


def list_maps(keyword: str, amount: int = 20, offset: int = 0) -> list[dict]:
    payload = {"cmd": "beatmaplist", "amount": amount, "offset": offset,
               "mode": 3, "keyword": keyword}
    req = urllib.request.Request(API, data=json.dumps(payload).encode(),
                                 headers={**HEADERS, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=40) as response:
        data = json.loads(response.read().decode("utf-8", errors="replace"))
    return data.get("data") or []


def download_osz(sid: int, target: Path) -> bytes:
    last = None
    for attempt in range(3):
        req = urllib.request.Request(f"{CDN}/{sid}", headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                raw = response.read()
            target.write_bytes(raw)
            return raw
        except Exception as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise last


def extract_set(sid: int, raw: bytes, out_root: Path) -> int:
    folder = out_root / str(sid)
    folder.mkdir(parents=True, exist_ok=True)
    accepted = 0
    try:
        archive = zipfile.ZipFile(__import__("io").BytesIO(raw))
    except zipfile.BadZipFile:
        return 0
    names = archive.namelist()
    for name in names:
        if not name.lower().endswith(".osu"):
            continue
        text = archive.read(name).decode("utf-8-sig", errors="replace")
        data = parse_osu(text)
        if data is None:
            continue
        audio = (data.get("meta") or {}).get("audio") or ""
        audio_name = next((n for n in names if Path(n).name == audio), None)
        if audio_name is None:
            audio_name = next((n for n in names
                               if Path(n).suffix.lower() in (".mp3", ".ogg", ".wav")), None)
        if audio_name is None:
            continue
        stem = Path(name).stem.replace("/", "_")
        (folder / f"{stem}.osu").write_bytes(archive.read(name))
        (folder / f"{stem}.mc").write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        audio_target = folder / Path(audio_name).name
        if not audio_target.exists():
            audio_target.write_bytes(archive.read(audio_name))
        accepted += 1
    return accepted


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", default="mania")
    parser.add_argument("--keywords", default="", help="comma-separated keywords")
    parser.add_argument("--limit", type=int, default=5, help="number of sets to try")
    parser.add_argument("--amount", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    words = [w.strip() for w in (args.keywords or args.keyword).split(",") if w.strip()]
    maps = []
    seen = set()
    for word in words:
        for item in list_maps(word, args.amount, args.offset):
            if item.get("sid") in seen:
                continue
            seen.add(item.get("sid"))
            if not (int(item.get("modes") or 0) & 8):
                continue
            maps.append(item)
    print("mania sets", len(maps), flush=True)
    charts = 0
    for item in maps:
        sid = item.get("sid")
        if not sid:
            continue
        folder = out_root / str(sid)
        if folder.exists() and any(folder.glob("*.mc")):
            print("skip", sid, flush=True)
            continue
        try:
            raw = download_osz(sid, out_root / f"{sid}.osz")
        except Exception as exc:
            print("download fail", sid, type(exc).__name__, exc, flush=True)
            continue
        added = extract_set(sid, raw, out_root)
        print(f"sid={sid} {item.get('title','')[:30]} 4k={added}", flush=True)
        charts += added
        args.limit -= 1
        if args.limit <= 0:
            break
        time.sleep(0.5)
    print("charts", charts, flush=True)


if __name__ == "__main__":
    main()
