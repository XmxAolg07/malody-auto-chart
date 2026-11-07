"""Run osu mirror download batches forever.

Stops when:
  * E: free space drops below --min-free-gb (default 50)
  * three consecutive batches add no new charts
  * the process is stopped manually
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def chart_count() -> int:
    return len(list((ROOT / "osu_mirror").rglob("*.mc")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=9700)
    parser.add_argument("--pages", type=int, default=30)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--min-free-gb", type=float, default=50.0)
    parser.add_argument("--pause", type=float, default=0.3)
    parser.add_argument("--sleep", type=int, default=10)
    args = parser.parse_args()
    min_free = int(args.min_free_gb * 1024 ** 3)
    offset = args.start
    empty_rounds = 0
    while True:
        free = shutil.disk_usage(ROOT).free
        print(f"offset={offset} free={free/1024**3:.1f}GB charts={chart_count()}", flush=True)
        if free < min_free:
            print(f"stop: free below {args.min_free_gb}GB", flush=True)
            break
        before = chart_count()
        cmd = [sys.executable, "-u", str(ROOT / "scripts" / "osu_direct_batch.py"),
               "--offset", str(offset), "--pages", str(args.pages),
               "--limit", str(args.limit), "--pause", str(args.pause)]
        subprocess.run(cmd, cwd=ROOT, check=False)
        after = chart_count()
        print(f"offset={offset} added={after - before} total={after}", flush=True)
        if after == before:
            empty_rounds += 1
            if empty_rounds >= 3:
                print("stop: three empty batches", flush=True)
                break
        else:
            empty_rounds = 0
        offset += args.pages * 50
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
