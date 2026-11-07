"""Enumerate mania 4K sets via osu.direct, download them from Sayobot.

osu.direct search supports offset paging and returns per-set `beatmaps` with
mode/cs, so only native 4K mania sets are downloaded.  Files come from
dl.sayobot.cn (free mirror).
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from sayobot_download import download_osz, extract_set  # noqa: E402

SEARCH = "https://osu.direct/api/v2/search?m=3&limit=50&offset={offset}"
OUT = ROOT / "osu_mirror"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def search(offset: int) -> list[dict]:
    req = urllib.request.Request(SEARCH.format(offset=offset), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=40) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--pages", type=int, default=10)
    parser.add_argument("--limit", type=int, default=100, help="4K charts target")
    parser.add_argument("--pause", type=float, default=0.5)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    charts = 0
    offset = args.offset
    for page in range(args.pages):
        try:
            items = search(offset)
        except Exception as exc:
            print("search fail", offset, type(exc).__name__, exc, flush=True)
            time.sleep(3)
            offset += 50
            continue
        if not items:
            print("no more results at", offset, flush=True)
            break
        fourk_sets = []
        for item in items:
            maps = item.get("beatmaps") or []
            if any(bm.get("mode") == "mania" and int(float(bm.get("cs") or 0)) == 4
                   and not bm.get("convert") for bm in maps):
                fourk_sets.append(item)
        print(f"offset={offset} sets={len(items)} 4k_sets={len(fourk_sets)} charts={charts}",
              flush=True)
        for item in fourk_sets:
            sid = item["id"]
            folder = OUT / str(sid)
            if folder.exists() and any(folder.glob("*.mc")):
                continue
            try:
                raw = download_osz(sid, OUT / f"{sid}.osz")
                added = extract_set(sid, raw, OUT)
            except Exception as exc:
                print("fail", sid, type(exc).__name__, exc, flush=True)
                time.sleep(2)
                continue
            print(f"  sid={sid} 4k={added} {str(item.get('title',''))[:35]}", flush=True)
            charts += added
            try:
                (OUT / f"{sid}.osz").unlink()
            except OSError:
                pass
            if charts >= args.limit:
                break
            time.sleep(args.pause)
        if charts >= args.limit:
            break
        offset += 50
    print("charts", charts, flush=True)
    import subprocess
    subprocess.run([sys.executable, "-m", "core.osu_data"], cwd=ROOT, check=False)
    subprocess.run([sys.executable, "scripts/build_chart_manifest.py"], cwd=ROOT, check=False)


if __name__ == "__main__":
    main()
