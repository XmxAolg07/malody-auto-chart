"""High-level analysis entry point used by the CLI and (later) the GUI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .audio import load_audio, probe_audio
from .bpm import estimate_rhythm
from .waveform import waveform_peaks


def analyze_file(
    path: str | Path,
    width: int = 1600,
    sr: Optional[int] = None,
    duration: Optional[float] = None,
    bpm_min: float = 50.0,
    bpm_max: float = 280.0,
    prior_center: float = 150.0,
    prior_octaves: float = 1.0,
    density_weight: float = 0.5,
) -> dict:
    """Analyse one audio file: audio info + rhythm + waveform peaks."""
    p = Path(path)
    samples, sample_rate = load_audio(p, sr=sr, duration=duration)
    info = probe_audio(p)
    info["analysed_duration"] = round(samples.size / float(sample_rate), 3)
    info["samplerate"] = int(sample_rate)

    rhythm = estimate_rhythm(
        samples, sample_rate, bpm_min=bpm_min, bpm_max=bpm_max,
        prior_center=prior_center, prior_octaves=prior_octaves,
        density_weight=density_weight,
    )
    peaks = waveform_peaks(samples, width)

    return {
        "file": str(p),
        "audio": info,
        "rhythm": rhythm.to_dict(),
        "waveform": peaks,
    }


def save_analysis(result: dict, json_path: str | Path) -> Path:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    return json_path
