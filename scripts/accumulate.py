"""Accumulate Malody charts in small sequential batches (no training).

Runs `malody_bulk.py` with a low worker count, waits between batches, and
refreshes the chart manifest at the end. Existing directories are skipped, so
re-running resumes where it stopped.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def count_dirs() -> int:
    root = ROOT / "community_downloads"
    return len([p for p in root.iterdir() if p.is_dir()]) if root.exists() else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--query", default="a")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-songs", type=int, default=250)
    parser.add_argument("--pause", type=int, default=30)
    args = parser.parse_args()
    for batch in range(1, args.batches + 1):
        before = count_dirs()
        print(f"=== batch {batch}/{args.batches} start, dirs={before} ===", flush=True)
        cmd = [PY, "-u", str(ROOT / "scripts" / "malody_bulk.py"),
               "--queries", args.query, "--limit", str(args.limit),
               "--workers", str(args.workers), "--max-songs", str(args.max_songs),
               "--out", str(ROOT / "community_downloads")]
        subprocess.run(cmd, cwd=ROOT, env=os.environ.copy(), check=False)
        print(f"=== batch {batch} done, dirs={count_dirs()} "
              f"(+{count_dirs() - before}) ===", flush=True)
        time.sleep(args.pause)
    subprocess.run([PY, str(ROOT / "scripts" / "build_chart_manifest.py")],
                   cwd=ROOT, env=os.environ.copy(), check=False)
    print("all batches done, dirs", count_dirs(), flush=True)


if __name__ == "__main__":
    main()
