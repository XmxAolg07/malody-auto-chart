"""Batch-download mania 4K sets from Sayobot and merge them into the dataset."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from sayobot_download import download_osz, extract_set, list_maps  # noqa: E402

KEYWORDS = ["mania", "malody", "dan", "4k", "LN", "tech", "stream", "jack",
            "chordjack", "rice", "hybrid", "jumpstream"]
OUT = ROOT / "osu_mirror"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=2)
    parser.add_argument("--amount", type=int, default=50)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--pause", type=float, default=1.0)
    parser.add_argument("--keywords", default=",".join(KEYWORDS))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    words = [w.strip() for w in args.keywords.split(",") if w.strip()]
    seen, tasks = set(), []
    for word in words:
        for page in range(args.pages):
            try:
                items = list_maps(word, args.amount, page * args.amount)
            except Exception as exc:
                print("list fail", word, page, type(exc).__name__, exc, flush=True)
                continue
            for item in items:
                sid = item.get("sid")
                if not sid or sid in seen:
                    continue
                seen.add(sid)
                if not (int(item.get("modes") or 0) & 8):
                    continue
                tasks.append(item)
        print(f"{word}: tasks={len(tasks)}", flush=True)
    print("unique mania sets", len(tasks), flush=True)
    done = 0
    for item in tasks:
        sid = item["sid"]
        folder = OUT / str(sid)
        if folder.exists() and any(folder.glob("*.mc")):
            continue
        try:
            raw = download_osz(sid, OUT / f"{sid}.osz")
            added = extract_set(sid, raw, OUT)
        except Exception as exc:
            print("fail", sid, type(exc).__name__, exc, flush=True)
            time.sleep(3)
            continue
        print(f"sid={sid} 4k={added} {str(item.get('title',''))[:40]}", flush=True)
        done += 1
        if added == 0:
            try:
                (OUT / f"{sid}.osz").unlink()
            except OSError:
                pass
        if done >= args.limit:
            break
        time.sleep(args.pause)
    print("sets done", done, flush=True)
    subprocess.run([sys.executable, "-m", "core.osu_data"], cwd=ROOT,
                   env=os.environ.copy(), check=False)
    subprocess.run([sys.executable, "scripts/build_chart_manifest.py"], cwd=ROOT,
                   env=os.environ.copy(), check=False)


if __name__ == "__main__":
    main()
