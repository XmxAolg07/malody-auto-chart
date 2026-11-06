"""Continuous Malody 4K downloader: batch after batch until a target count.

Logs in once, enumerates songs by query words, downloads every 4K chart that is
not already on disk, and repeats.  Existing directories are skipped, so the
process is resumable and safe to stop at any time.

Usage:
  set MALODY_USER=... & set MALODY_PWD=...
  python scripts/auto_download.py --target 12000 --batch 100 --workers 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import malody_download as md  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOCK = None


def out_dir_for(entry: dict, root: Path) -> Path:
    version = str(entry.get("version") or "").replace(" ", "_")
    return root / f"{entry.get('cid')}_{version}"


def download_one(song, entry, root: Path) -> bool:
    out_dir = out_dir_for(entry, root)
    if out_dir.exists():
        return False
    try:
        info = md.manifest(entry["cid"], md.KEY, md.UID)
    except Exception:
        return False
    names = [str(i.get("name") or "") for i in info.get("list") or []]
    has_mc = any(n.lower().endswith(".mc") for n in names)
    if not has_mc:
        # keep osu! ports too; extract_set converts them later
        pass
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    sid = info.get("sid") or info.get("dsid")
    uid = info.get("uid") or 0
    ok = True
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(md.fetch_file, sid, uid, item, out_dir)
                   for item in info.get("list") or []]
        for future in as_completed(futures):
            try:
                path = future.result()
            except Exception:
                ok = False
                continue
            if str(path).lower().endswith(".mc"):
                try:
                    with zipfile.ZipFile(path) as zf:
                        members = [n for n in zf.namelist() if n.lower().endswith(".mc")]
                        if members:
                            payload = zf.read(members[0])
                            path.with_suffix(".mcz").write_bytes(path.read_bytes())
                            path.write_bytes(payload)
                except zipfile.BadZipFile:
                    pass
    if not has_mc:
        try:
            from osu_to_mc import convert_file
            osu = next(iter(sorted(out_dir.glob("*.osu"))), None)
            if osu:
                convert_file(osu)
        except Exception:
            pass
    return ok


def enumerate_tasks(word: str, songs_limit: int, key: str, uid: str) -> list:
    try:
        songs = md.search(word, key, uid)[:songs_limit]
    except Exception as exc:
        print("search fail", word, type(exc).__name__, flush=True)
        return []
    print(f"query={word} songs={len(songs)}", flush=True)
    tasks = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(md.charts, song.get("id"), word, key, uid): song
                   for song in songs}
        for future in as_completed(futures):
            song = futures[future]
            try:
                entries = future.result()
            except Exception:
                continue
            for entry in entries:
                if "4K" not in str(entry.get("version") or "").upper():
                    continue
                tasks.append((song, entry))
    print(f"query={word} 4K tasks={len(tasks)}", flush=True)
    return tasks


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=12000)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--songs", type=int, default=600, help="songs per query")
    parser.add_argument("--queries", default="a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--out", default="community_downloads")
    parser.add_argument("--pause", type=float, default=1.0)
    parser.add_argument("--convert", action="store_true", default=True)
    parser.add_argument("--no-convert", dest="convert", action="store_false")
    parser.add_argument("--key", default=os.environ.get("MALODY_KEY"))
    parser.add_argument("--uid", default=os.environ.get("MALODY_UID"))
    parser.add_argument("--user", default=os.environ.get("MALODY_USER"))
    parser.add_argument("--pwd", default=os.environ.get("MALODY_PWD"))
    args = parser.parse_args()
    if args.key and args.uid:
        md.KEY, md.UID = args.key, str(args.uid)
    elif args.user and args.pwd:
        md.KEY, md.UID = md.login(args.user, args.pwd)
    else:
        raise SystemExit("need MALODY_USER/MALODY_PWD")
    print("logged in", md.UID, flush=True)
    root = Path(args.out)
    root.mkdir(exist_ok=True)
    queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    total = 0
    for word in queries:
        tasks = enumerate_tasks(word, args.songs, md.KEY, md.UID)
        pending = [task for task in tasks if not out_dir_for(task[1], root).exists()]
        print(f"{word}: pending={len(pending)} downloaded={total}", flush=True)
        for start in range(0, len(pending), args.batch):
            chunk = pending[start:start + args.batch]
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(download_one, song, entry, root)
                           for song, entry in chunk]
                for future in as_completed(futures):
                    try:
                        if future.result():
                            total += 1
                    except Exception:
                        pass
            print(f"{word}: batch done, total={total}", flush=True)
            if total >= args.target:
                break
            time.sleep(args.pause)
        if total >= args.target:
            break
    print("downloaded", total, "target", args.target, flush=True)
    if args.convert:
        import subprocess
        subprocess.run([sys.executable, "-m", "core.batch_data"], cwd=ROOT, check=False)
        subprocess.run([sys.executable, "scripts/build_chart_manifest.py"], cwd=ROOT, check=False)


if __name__ == "__main__":
    main()
