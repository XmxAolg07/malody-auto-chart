"""Command line entry point.

Examples
--------
    python cli.py "song.ogg"
    python cli.py "song.ogg" --out out --width 2400
    python cli.py "song.ogg" --zoom 20        # extra 20 s close-up view
    python cli.py "song.ogg" --no-json        # only the PNG
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``import core`` work no matter which directory the user runs this from.
APP_IS_FROZEN = bool(getattr(sys, "frozen", False))
APP_DIR = Path(sys.executable).resolve().parent if APP_IS_FROZEN else Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from core.analyze import analyze_file, save_analysis
from core.waveform import render_waveform


def _configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _fmt_duration(seconds: float) -> str:
    seconds = float(seconds)
    return f"{int(seconds // 60)}:{seconds % 60:05.2f}"


def analyze_one(
    audio_path: Path,
    out_dir: Path,
    width: int,
    zoom: float | None,
    make_png: bool,
    make_json: bool,
    duration: float | None,
    sr: int | None,
    bpm_min: float,
    bpm_max: float,
    prior_center: float,
    prior_octaves: float,
    density_weight: float,
) -> dict:
    result = analyze_file(
        audio_path, width=width, sr=sr, duration=duration,
        bpm_min=bpm_min, bpm_max=bpm_max,
        prior_center=prior_center, prior_octaves=prior_octaves,
        density_weight=density_weight,
    )
    rhythm = result["rhythm"]
    info = result["audio"]
    stem = audio_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    if make_json:
        save_analysis(result, out_dir / f"{stem}.analysis.json")
        print(f"  wrote: {out_dir / (stem + '.analysis.json')}")

    if make_png and rhythm.get("bpm"):
        meta = {
            "path": str(audio_path),
            "title": audio_path.name,
            "bpm": rhythm["bpm"],
            "offset_ms": rhythm["offset_ms"],
            "confidence": rhythm["confidence"],
            "channels": info.get("channels"),
        }
        from core.audio import load_audio
        samples, sample_rate = load_audio(audio_path, sr=sr, duration=duration)
        render_waveform(
            samples, sample_rate, out_dir / f"{stem}.waveform.png",
            meta=meta, width=width, beat_times_ms=rhythm["beats_ms"],
        )
        print(f"  wrote: {out_dir / (stem + '.waveform.png')}")

        if zoom:
            zoom_span = min(float(zoom), samples.size / float(sample_rate))
            render_waveform(
                samples, sample_rate, out_dir / f"{stem}.waveform.zoom.png",
                meta=meta, width=width, height=360,
                beat_times_ms=rhythm["beats_ms"], span=zoom_span,
                title=f"{audio_path.name}  (first {zoom:g}s)",
            )
            print(f"  wrote: {out_dir / (stem + '.waveform.zoom.png')}")

    return result


def main(argv: list[str] | None = None) -> int:
    _configure_console()
    parser = argparse.ArgumentParser(
        description="Malody 4K auto-charting - stage 1 audio analysis (BPM + waveform)"
    )
    parser.add_argument("audio", nargs="*", help="audio file(s): ogg/mp3/wav/flac/...")
    parser.add_argument("-o", "--out", default=None,
                        help="output directory (default: out next to the program)")
    parser.add_argument("--width", type=int, default=2000, help="waveform image width in px")
    parser.add_argument("--zoom", type=float, default=None,
                        help="also render a close-up of the first N seconds")
    parser.add_argument("--duration", type=float, default=None,
                        help="only analyse the first N seconds")
    parser.add_argument("--sr", type=int, default=None, help="resample before analysis")
    parser.add_argument("--bpm-min", type=float, default=50.0)
    parser.add_argument("--bpm-max", type=float, default=280.0)
    parser.add_argument("--prior-center", type=float, default=150.0,
                        help="tempo prior centre (rhythm-game charts cluster around 150)")
    parser.add_argument("--prior-octaves", type=float, default=1.0,
                        help="tempo prior width in octaves; larger = flatter")
    parser.add_argument("--density-weight", type=float, default=0.5,
                        help="0..1, how strongly empty beat slots are penalised")
    parser.add_argument("--no-png", action="store_true", help="skip waveform images")
    parser.add_argument("--no-json", action="store_true", help="skip the JSON report")
    parser.add_argument("--no-pause", action="store_true",
                        help="do not wait for Enter when the packaged exe finishes")
    args = parser.parse_args(argv)

    if not args.audio:
        parser.print_help()
        print("\nTip: drag an audio file onto the exe, or run:")
        print('  MalodyAudioAnalyzer.exe "song.ogg"')
        return 0

    out_dir = Path(args.out) if args.out else APP_DIR / "out"
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir

    failures = 0
    for i, name in enumerate(args.audio, 1):
        audio_path = Path(name)
        print(f"[{i}/{len(args.audio)}] {audio_path.name}")
        try:
            result = analyze_one(
                audio_path, out_dir, args.width, args.zoom,
                make_png=not args.no_png, make_json=not args.no_json,
                duration=args.duration, sr=args.sr,
                bpm_min=args.bpm_min, bpm_max=args.bpm_max,
                prior_center=args.prior_center,
                prior_octaves=args.prior_octaves,
                density_weight=args.density_weight,
            )
        except Exception as exc:  # keep going with the next file
            failures += 1
            print(f"  ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        rhythm = result["rhythm"]
        info = result["audio"]
        channels = info.get("channels")
        mode = "stereo" if (channels or 0) > 1 else ("mono" if channels else "?")
        print(f"  audio : {_fmt_duration(rhythm['duration_sec'])}  "
              f"{rhythm['sr']} Hz  {mode}  {info.get('format', '?')}/{info.get('subtype', '?')}")
        print(f"  bpm   : {rhythm['bpm']:.3f}   (confidence {rhythm['confidence'] * 100:.1f}%)")
        print(f"  offset: {rhythm['offset_ms']:.1f} ms   beats: {len(rhythm['beats_ms'])}")
        alt = ", ".join(f"{c['bpm']:.2f}" for c in rhythm["candidates"][1:4])
        if alt:
            print(f"  alt   : {alt}")
        print()

    return 1 if failures else 0


if __name__ == "__main__":
    _configure_console()
    code = main()
    if APP_IS_FROZEN and "--no-pause" not in sys.argv:
        try:
            input("\nDone. Press Enter to close...")
        except Exception:
            pass
    raise SystemExit(code)
