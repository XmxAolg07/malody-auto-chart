"""Benchmark BPM detection against the BPM annotated inside real .mc charts.

The chart's ``meta.background`` names the artwork, which shares its stem with
the song's audio file in the same folder.  That gives us a free ground truth:
the BPM list in ``chart["time"]``.

Read-only: source files are never modified.

Run:
    python tests/bench_charts.py --limit 30 --duration 120
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.audio import load_audio                      # noqa: E402
from core.bpm import estimate_rhythm                   # noqa: E402


def chart_bpms(mc_path: Path) -> list[float]:
    with mc_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [float(t["bpm"]) for t in data.get("time", []) if "bpm" in t]


def chart_audio(mc_path: Path) -> Path | None:
    with mc_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    background = (data.get("meta") or {}).get("background") or ""
    if not background:
        return None
    stem = Path(background).stem
    folder = mc_path.parent
    for ext in (".ogg", ".mp3", ".wav", ".flac", ".m4a"):
        candidate = folder / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    # fall back to a fuzzy match when the artwork name has extra suffixes
    for candidate in folder.glob(f"{stem}.*"):
        if candidate.suffix.lower() in {".ogg", ".mp3", ".wav", ".flac", ".m4a"}:
            return candidate
    return None


def classify(detected: float, reference: list[float]) -> str:
    if not detected or not reference:
        return "none"
    for bpm in reference:
        ratio = detected / bpm
        if abs(ratio - 1.0) <= 0.015:
            return "exact"
        if abs(ratio - 0.5) <= 0.015:
            return "half"
        if abs(ratio - 2.0) <= 0.015:
            return "double"
        if abs(ratio - 2.0 / 3.0) <= 0.02 or abs(ratio - 1.5) <= 0.02:
            return "other"
    return "miss"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="max unique songs (0 = all)")
    parser.add_argument("--duration", type=float, default=None, help="analyse only N seconds")
    parser.add_argument("--prior-center", type=float, default=150.0)
    parser.add_argument("--prior-octaves", type=float, default=1.0)
    parser.add_argument("--density-weight", type=float, default=0.5)
    parser.add_argument("--mc", default=None, help="limit to one folder of charts")
    parser.add_argument("--show-all", action="store_true")
    args = parser.parse_args()

    search_root = Path(args.mc) if args.mc else ROOT
    mc_files = sorted(search_root.rglob("*.mc"))
    seen_audio: set[Path] = set()
    rows = []

    for mc in mc_files:
        audio = chart_audio(mc)
        if audio is None or audio in seen_audio:
            continue
        seen_audio.add(audio)
        reference = chart_bpms(mc)
        try:
            samples, sr = load_audio(audio, duration=args.duration)
            result = estimate_rhythm(samples, sr,
                                     prior_center=args.prior_center,
                                     prior_octaves=args.prior_octaves,
                                     density_weight=args.density_weight)
        except Exception as exc:
            print(f"ERROR {audio.name}: {exc}")
            continue
        verdict = classify(result.bpm, reference)
        rows.append({
            "audio": audio.name,
            "chart": mc.name,
            "detected": result.bpm,
            "reference": reference,
            "verdict": verdict,
            "confidence": result.confidence,
        })
        if args.show_all or verdict != "exact":
            ref = "/".join(f"{b:.2f}" for b in reference[:4])
            print(f"{verdict:7} det={result.bpm:8.2f}  ref={ref:28}  "
                  f"conf={result.confidence:5.2f}  {audio.name[:46]}")
        if args.limit and len(rows) >= args.limit:
            break

    counts = Counter(r["verdict"] for r in rows)
    total = len(rows)
    print("\n" + "=" * 70)
    print(f"songs: {total}   " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if total:
        print(f"exact rate: {counts['exact'] / total * 100:.1f}%   "
              f"exact+octave: {(counts['exact'] + counts['half'] + counts['double']) / total * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
