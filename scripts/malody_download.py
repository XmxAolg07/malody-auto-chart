"""Download Malody 4K charts from the community CGI, based on Chartunes' client.

Usage (credentials via environment, never persisted):
  set MALODY_USER=... & set MALODY_PWD=...
  python scripts/malody_download.py --query "Sakuzyo" --limit 2 --4k-only

Or reuse a cached key:
  python scripts/malody_download.py --key ... --uid ... --query ...

API (from Chartunes):
  POST /cgi/login           name, md5(psw), v, h
  GET  /cgi/list?type=1     search songs
  GET  /cgi/list?type=2     charts of a song
  GET  /cgi/chart/download  v=2, cid -> manifest
  GET  http://cdn0.machart.top/<sid>/<uid>/<file>
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

API = "http://m.mugzone.net"
CDN = "http://cdn0.machart.top"
VERSION = 262919
DEVICE = "3665630189535747679_0"
HEADERS = {"Referer": API, "Accept": "*/*", "MaVersion": str(VERSION)}


def _request(method: str, url: str, params=None, data=None):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=HEADERS, method=method)
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def _json(raw: bytes):
    return json.loads(raw.decode("utf-8-sig", errors="replace"))


def login(user: str, password: str) -> tuple[str, int]:
    form = {
        "name": user,
        "psw": hashlib.md5(password.encode()).hexdigest(),
        "v": VERSION,
        "h": DEVICE,
    }
    payload = _json(_request("POST", f"{API}/cgi/login", data=form))
    if payload.get("code") != 0:
        raise SystemExit(f"login rejected: {payload}")
    data = payload.get("data") or {}
    return data["key"], int(data["uid"])


def cgi(path: str, params: dict, key: str, uid: int):
    merged = dict(params, key=key, uid=uid)
    payload = _json(_request("GET", f"{API}{path}", params=merged))
    if payload.get("code") != 0:
        raise SystemExit(f"{path} failed: {payload}")
    return payload.get("data") or {}


def search(word: str, key: str, uid: int):
    data = cgi("/cgi/list", {"type": 1, "word": word, "org": 1}, key, uid)
    return [item for item in (data if isinstance(data, list) else [])
            if item.get("mode")]


def charts(song_id: int, word: str, key: str, uid: int):
    data = cgi("/cgi/list", {"type": 2, "sid": song_id, "word": word}, key, uid)
    return data if isinstance(data, list) else []


def manifest(cid: int, key: str, uid: int):
    data = cgi("/cgi/chart/download", {"v": 2, "cid": cid}, key, uid)
    if "list" not in data:
        raise SystemExit(f"bad manifest for {cid}: {data}")
    return data


def fetch_file(sid, uid, item, out_dir: Path):
    url = f"{CDN}/{sid}/{uid}/{item['file']}"
    raw = _request("GET", url)
    target = raw
    if item["name"].lower().endswith(".mc"):
        try:
            with zipfile.ZipFile(__import__("io").BytesIO(raw)) as zf:
                members = [n for n in zf.namelist() if n.lower().endswith(".mc")]
                if members:
                    target = zf.read(members[0])
        except zipfile.BadZipFile:
            pass
    digest = hashlib.md5(target).hexdigest()
    if digest != item.get("hash"):
        raise SystemExit(f"md5 mismatch {item['name']}")
    path = out_dir / item["name"]
    path.write_bytes(raw)
    return path


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="")
    parser.add_argument("--queries", default="", help="comma-separated search words")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--out", default="community_downloads")
    parser.add_argument("--4k-only", dest="four_k_only", action="store_true")
    parser.add_argument("--no-unpack", dest="unpack", action="store_false",
                        help="keep .mc as zip instead of extracting JSON")
    parser.add_argument("--native-only", action="store_true",
                        help="skip osu! ports (default: include and convert)")
    parser.set_defaults(unpack=True)
    parser.add_argument("--user", default=os.environ.get("MALODY_USER"))
    parser.add_argument("--pwd", default=os.environ.get("MALODY_PWD"))
    parser.add_argument("--key", default=os.environ.get("MALODY_KEY"))
    parser.add_argument("--uid", type=int, default=os.environ.get("MALODY_UID"))
    args = parser.parse_args()
    if args.key and args.uid:
        key, uid = args.key, int(args.uid)
    elif args.user and args.pwd:
        key, uid = login(args.user, args.pwd)
    else:
        raise SystemExit("need MALODY_USER/MALODY_PWD or --key/--uid")
    print(f"logged in uid={uid}")

    words = [w.strip() for w in (args.queries or args.query).split(",") if w.strip()]
    if not words:
        raise SystemExit("need --query or --queries")
    hits = []
    seen_songs = set()
    for word in words:
        for song in search(word, key, uid):
            sid = song.get("id")
            if sid in seen_songs:
                continue
            seen_songs.add(sid)
            song["_word"] = word
            hits.append(song)
    print(f"songs={len(hits)}")
    out_root = Path(args.out)
    downloaded = 0
    for song in hits:
        sid = song.get("id")
        title = song.get("title") or ""
        entries = charts(sid, song.get("_word", ""), key, uid)
        for entry in entries:
            version = str(entry.get("version") or "")
            if args.four_k_only and "4K" not in version.upper():
                continue
            cid = entry.get("cid")
            try:
                info = manifest(cid, key, uid)
            except SystemExit as exc:
                print("skip", cid, exc)
                continue
            names = [str(item.get("name") or "") for item in info.get("list") or []]
            has_mc = any(name.lower().endswith(".mc") for name in names)
            if args.native_only and not has_mc:
                print("skip osu-port", cid, version)
                continue
            out_dir = out_root / f"{cid}_{version.replace(' ', '_')}"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "manifest.json").write_text(json.dumps(info, ensure_ascii=False, indent=1),
                                                   encoding="utf-8")
            for item in info.get("list") or []:
                try:
                    path = fetch_file(info.get("sid") or info.get("dsid"),
                                      info.get("uid") or 0, item, out_dir)
                    if args.unpack and str(path).lower().endswith(".mc"):
                        try:
                            with zipfile.ZipFile(path) as zf:
                                members = [n for n in zf.namelist() if n.lower().endswith(".mc")]
                                if members:
                                    payload = zf.read(members[0])
                                    path.with_suffix(".mcz").write_bytes(path.read_bytes())
                                    path.write_bytes(payload)
                        except zipfile.BadZipFile:
                            pass
                except SystemExit as exc:
                    print("  fail", item.get("name"), exc)
            if not has_mc:
                try:
                    from osu_to_mc import convert_file
                    osu = next(iter(sorted(out_dir.glob("*.osu"))), None)
                    if osu is not None:
                        converted = convert_file(osu)
                        if converted:
                            print("  converted osu ->", converted.name)
                except Exception as exc:
                    print("  osu convert failed", exc)
            print(f"downloaded cid={cid} {title} [{version}] -> {out_dir}")
            downloaded += 1
            time.sleep(0.4)
            if downloaded >= args.limit:
                break
        if downloaded >= args.limit:
            break
    print("done", downloaded)


if __name__ == "__main__":
    main()
