"""Audio decoding helpers.

Design notes
------------
* Input audio is treated as **read-only**: this module only reads bytes.
* `soundfile` (libsndfile) is the primary decoder; it handles ogg/vorbis,
  flac, wav, mp3 and opus on Windows without any external binary.
* If soundfile is unavailable (or the format is unsupported, e.g. m4a),
  we fall back to ffmpeg when one can be found.

The bundled `vendor/` directory (project root) is added to ``sys.path`` so the
wheel-extracted soundfile/cffi packages can be imported without installing
them into the system interpreter.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PROJECT_ROOT / "vendor"
if VENDOR_DIR.is_dir() and str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

AUDIO_EXTENSIONS = {
    ".ogg", ".oga", ".opus", ".mp3", ".wav", ".flac",
    ".m4a", ".aac", ".wma", ".aiff", ".aif",
}


def _find_ffmpeg() -> Optional[str]:
    """Locate an ffmpeg executable, without hard-coding a machine path."""
    env = os.environ.get("MALODY_FFMPEG")
    if env and Path(env).is_file():
        return env

    local = PROJECT_ROOT / "tools" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if local.is_file():
        return str(local)

    found = shutil.which("ffmpeg")
    if found:
        return found

    if os.name == "nt":
        candidates = [
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "ffmpeg" / "bin" / "ffmpeg.exe",
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "ffmpeg" / "bin" / "ffmpeg.exe",
            Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
            # ffmpeg bundled with the user's sd-webui/aki install (found on this machine)
            Path(r"D:\BaiduNetdiskDownload\(sd-webui-forge-aki-v1.0\aki\bin\ffmpeg.exe"),
        ]
        for path in candidates:
            if path.is_file():
                return str(path)
    return None


def load_audio(
    path: str | Path,
    sr: Optional[int] = None,
    duration: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """Decode an audio file to mono ``float32``.

    Parameters
    ----------
    path:
        Audio file to read.
    sr:
        Target sample rate. ``None`` keeps the file's rate.
    duration:
        Only decode the first *duration* seconds (useful for quick scans).

    Returns
    -------
    (samples, sample_rate)
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"audio file not found: {p}")

    samples, file_sr = _load_with_soundfile(p, duration)
    if samples is None:
        samples, file_sr = _load_with_ffmpeg(p, sr, duration)
        # ffmpeg already produced the requested rate
        if sr is not None and file_sr != sr:
            samples = resample_audio(samples, file_sr, sr)
            file_sr = sr
        return samples, file_sr

    if sr is not None and sr != file_sr:
        samples = resample_audio(samples, file_sr, sr)
        file_sr = sr
    return samples, file_sr


def _load_with_soundfile(p: Path, duration: Optional[float]):
    try:
        import soundfile as sf
    except Exception:
        return None, None
    try:
        if duration is not None:
            info = sf.info(str(p))
            frames = max(1, int(duration * info.samplerate))
            data, file_sr = sf.read(str(p), dtype="float32", always_2d=True,
                                    frames=frames)
        else:
            data, file_sr = sf.read(str(p), dtype="float32", always_2d=True)
    except Exception:
        return None, None

    if data.size == 0:
        raise RuntimeError(f"decoded zero samples from {p}")
    mono = data.mean(axis=1, dtype=np.float32) if data.shape[1] > 1 else data[:, 0]
    return np.ascontiguousarray(mono, dtype=np.float32), int(file_sr)


def _load_with_ffmpeg(p: Path, sr: Optional[int], duration: Optional[float]):
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(
            f"cannot decode {p.name}: soundfile failed and ffmpeg was not found "
            "(set MALODY_FFMPEG to an ffmpeg executable)"
        )
    target_sr = int(sr or 44100)
    cmd = [ffmpeg, "-v", "error", "-i", str(p), "-vn"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-f", "f32le", "-ac", "1", "-ar", str(target_sr), "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed for {p.name}: {msg}")
    data = np.frombuffer(proc.stdout, dtype="<f4").copy()
    if data.size == 0:
        raise RuntimeError(f"ffmpeg decoded zero samples from {p}")
    return data, target_sr


def probe_audio(path: str | Path) -> dict:
    """Return lightweight metadata (best effort, never raises)."""
    p = Path(path)
    info: dict = {"path": str(p), "name": p.name}
    try:
        import soundfile as sf
        meta = sf.info(str(p))
        info.update(
            samplerate=int(meta.samplerate),
            channels=int(meta.channels),
            duration=float(meta.duration),
            format=str(meta.format),
            subtype=str(meta.subtype),
        )
    except Exception:
        pass
    return info


def resample_audio(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """FFT-based resampling (good enough for analysis)."""
    if sr_in == sr_out or y.size == 0:
        return y.astype(np.float32, copy=False)
    n_in = y.size
    n_out = max(1, int(round(n_in * sr_out / sr_in)))
    spectrum = np.fft.rfft(y.astype(np.float64))
    out_spectrum = np.zeros(n_out // 2 + 1, dtype=np.complex128)
    # Keep bins below the lower Nyquist frequency (with a little headroom).
    cutoff_hz = min(sr_in, sr_out) * 0.5 * 0.95
    cutoff_bin = int(cutoff_hz / (sr_in / 2) * (spectrum.size - 1))
    keep = min(cutoff_bin, spectrum.size - 1, out_spectrum.size - 1)
    out_spectrum[: keep + 1] = spectrum[: keep + 1]
    resampled = np.fft.irfft(out_spectrum, n_out) * (n_out / n_in)
    return resampled.astype(np.float32)


def export_audio(path: str | Path, samples: np.ndarray, sr: int,
                 fmt: str = "wav") -> Path:
    """Write mono float samples as WAV (PCM 16) or OGG (Vorbis).

    OGG encoding goes through ffmpeg because the bundled libsndfile 1.2.2
    crashes in its Vorbis encoder for clips longer than ~15 seconds.  Very
    short clips still fall back to libsndfile when ffmpeg is unavailable.
    """
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    fmt = fmt.lower().lstrip(".")

    if fmt == "wav":
        sf.write(str(path), data, int(sr), format="WAV", subtype="PCM_16")
        return path

    if fmt in ("ogg", "oga"):
        ffmpeg = _find_ffmpeg()
        if ffmpeg:
            temp_wav = path.with_name(path.stem + ".tmp-export.wav")
            sf.write(str(temp_wav), data, int(sr), format="WAV", subtype="PCM_16")
            try:
                proc = subprocess.run(
                    [ffmpeg, "-y", "-v", "error", "-i", str(temp_wav),
                     "-c:a", "libvorbis", "-q:a", "5", str(path)],
                    capture_output=True,
                )
                if proc.returncode != 0:
                    msg = proc.stderr.decode("utf-8", errors="replace").strip()
                    raise RuntimeError(f"ffmpeg OGG 编码失败：{msg}")
            finally:
                temp_wav.unlink(missing_ok=True)
            return path

        if data.size / float(sr) <= 8.0:
            sf.write(str(path), data, int(sr), format="OGG", subtype="VORBIS")
            return path
        raise RuntimeError(
            "OGG 导出需要 ffmpeg：把 ffmpeg.exe 放到 tools/ 目录，"
            "或设置环境变量 MALODY_FFMPEG"
        )

    raise ValueError(f"unsupported export format: {fmt}")
