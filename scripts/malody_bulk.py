"""Concurrent bulk downloader for Malody 4K charts (native + osu! ports)."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import malody_download as md  # noqa: E402

LOCK = threading.Lock()
COUNT = 0


def has_complete(out_dir: Path) -> bool:
    if not out_dir.exists():
        return False
    files = [p for p in out_dir.iterdir() if p.is_file()]
    return (any(p.suffix.lower() in (".mc", ".mcz", ".osu") for p in files)
            and any(p.suffix.lower() in (".ogg", ".mp3", ".wav", ".flac") for p in files))


def download_chart(task, out_root: Path, native_only: bool, unpack: bool) -> bool:
    song, entry = task
    cid = entry.get("cid")
    version = str(entry.get("version") or "")
    out_dir = out_root / f"{cid}_{version.replace(' ', '_')}"
    if has_complete(out_dir):
        return False
    try:
        info = md.manifest(cid, md.KEY, md.UID)
    except Exception:
        return False
    names = [str(item.get("name") or "") for item in info.get("list") or []]
    has_mc = any(name.lower().endswith(".mc") for name in names)
    if native_only and not has_mc:
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(
        __import__("json").dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    sid = info.get("sid") or info.get("dsid")
    uid = info.get("uid") or 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(md.fetch_file, sid, uid, item, out_dir)
                   for item in info.get("list") or []]
        for future in as_completed(futures):
            try:
                path = future.result()
            except SystemExit:
                continue
            if unpack and str(path).lower().endswith(".mc"):
                import zipfile
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
        from osu_to_mc import convert_file
        osu = next(iter(sorted(out_dir.glob("*.osu"))), None)
        if osu is not None:
            convert_file(osu)
    with LOCK:
        global COUNT
        COUNT += 1
        print(f"[{COUNT}] {cid} {song.get('title','')[:30]} [{version}]", flush=True)
    return True


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="a")
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--out", default="community_downloads")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-songs", type=int, default=1600)
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--key", default=os.environ.get("MALODY_KEY"))
    parser.add_argument("--uid", type=int, default=os.environ.get("MALODY_UID"))
    parser.add_argument("--user", default=os.environ.get("MALODY_USER"))
    parser.add_argument("--pwd", default=os.environ.get("MALODY_PWD"))
    args = parser.parse_args()
    if args.key and args.uid:
        md.KEY, md.UID = args.key, int(args.uid)
    elif args.user and args.pwd:
        md.KEY, md.UID = md.login(args.user, args.pwd)
    else:
        raise SystemExit("need MALODY_USER/MALODY_PWD or key/uid")
    print("logged in", md.UID, flush=True)

    hits = []
    seen = set()
    for word in [w.strip() for w in args.queries.split(",") if w.strip()]:
        for song in md.search(word, md.KEY, md.UID):
            sid = song.get("id")
            if sid in seen:
                continue
            seen.add(sid)
            song["_word"] = word
            hits.append(song)
            if len(hits) >= args.max_songs:
                break
        if len(hits) >= args.max_songs:
            break
    print("songs", len(hits), flush=True)
    tasks = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(md.charts, song.get("id"), song.get("_word", ""),
                               md.KEY, md.UID): song for song in hits}
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
    print("4K charts", len(tasks), flush=True)
    tasks = tasks[: args.limit]
    out_root = Path(args.out)
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_chart, task, out_root, args.native_only, True)
                   for task in tasks]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                pass
            if COUNT >= args.limit:
                break
    print("done", COUNT, "in", round(time.time() - started, 1), "s", flush=True)


if __name__ == "__main__":
    main()
