from __future__ import annotations

import json
import os
import site
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable, TextIO
import re
import numpy as np
import av


def configure_text_stream(stream: TextIO | None) -> None:
    """
    Configure redirected process streams to use UTF-8.

    CommsAI communicates with the Windows frontend through stdin/stdout.
    Windows may otherwise use a legacy character encoding that cannot
    represent Cyrillic, Chinese, Japanese and other non-English text.
    """
    if stream is None:
        return

    reconfigure = getattr(stream, "reconfigure", None)

    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


# Set UTF-8 as early as possible for all communication with the frontend.
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

configure_text_stream(sys.stdin)
configure_text_stream(sys.stdout)
configure_text_stream(sys.stderr)


# Make pip-provided CUDA DLLs discoverable before importing CTranslate2.
if os.name == "nt":
    roots = list(site.getsitepackages())

    user_site = site.getusersitepackages()
    if user_site:
        roots.append(user_site)

    exe_root = Path(sys.executable).resolve().parent

    roots.extend(
        [
            str(exe_root),
            str(exe_root / "_internal"),
        ]
    )

    for root in roots:
        base = Path(root)

        for relative in (
            Path("nvidia") / "cublas" / "bin",
            Path("nvidia") / "cudnn" / "bin",
            Path("_internal") / "nvidia" / "cublas" / "bin",
            Path("_internal") / "nvidia" / "cudnn" / "bin",
        ):
            folder = base / relative

            if not folder.exists():
                continue

            try:
                os.add_dll_directory(str(folder))
            except (OSError, AttributeError):
                pass

            os.environ["PATH"] = (
                str(folder)
                + os.pathsep
                + os.environ.get("PATH", "")
            )


from faster_whisper import WhisperModel


model: WhisperModel | None = None


def emit(payload: dict) -> None:
    """
    Send one JSON message to the C# frontend.

    ensure_ascii=True deliberately produces ASCII-only JSON. Non-English
    text is represented using JSON Unicode escapes and is restored
    automatically by the frontend's JSON parser. This avoids Windows
    code-page errors even if a process stream is configured incorrectly.
    """
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
    )

    if sys.stdout is None:
        return

    sys.stdout.write(encoded + "\n")
    sys.stdout.flush()


def status(message: str) -> None:
    emit(
        {
            "type": "status",
            "message": message,
        }
    )


def report_error(exc: Exception) -> None:
    """
    Send a readable error to the frontend without allowing traceback
    output itself to cause another UnicodeEncodeError.
    """
    try:
        traceback_text = "".join(
            traceback.format_exception(
                type(exc),
                exc,
                exc.__traceback__,
            )
        )

        if sys.stderr is not None:
            sys.stderr.write(
                traceback_text.encode(
                    "utf-8",
                    errors="replace",
                ).decode(
                    "utf-8",
                    errors="replace",
                )
            )
            sys.stderr.flush()
    except Exception:
        pass

    emit(
        {
            "type": "error",
            "message": str(exc),
        }
    )


def load_model(model_name: str, compute_type: str) -> None:
    global model

    status(
        f"Loading {model_name} on the NVIDIA GPU. "
        "The first download can take several minutes..."
    )

    model = WhisperModel(
        model_name,
        device="cuda",
        compute_type=compute_type,
        download_root=str(
            Path.home()
            / ".commsai"
            / "models"
        ),
    )

    status("GPU model ready. Listening...")


def normalise_text(value: str) -> str:
    value = value.lower().replace("’", "'")
    value = re.sub(r"[^a-z0-9\s']+", " ", value)
    return " ".join(value.split())


CS2_INITIAL_PROMPT = (
    "Counter-Strike 2 team voice communications. "
    "Common terms include A, B, mid, long, short, ramp, apps, palace, "
    "connector, window, heaven, hell, site, rotate, rush, eco, force, "
    "save, drop, flash, smoke, molotov, grenade, AWP, one, two, three."
)

HALLUCINATION_PHRASES = {
    "translator's note",
    "translators note",
    "translated into english",
    "translation into english",
    "english translation",
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "like and subscribe",
    "subtitles by",
    "captioned by",
    "amara org community",
    "copyright",
}


def looks_like_hallucination(text: str) -> bool:
    normalised = normalise_text(text)
    if not normalised:
        return True

    return any(
        phrase in normalised
        for phrase in HALLUCINATION_PHRASES
    )


def audio_rms_and_peak(path: Path) -> tuple[float, float, float]:
    """Return duration seconds, RMS and peak for decoded audio."""
    chunks: list[np.ndarray] = []
    sample_rate = 0

    with av.open(str(path)) as container:
        audio_stream = next(
            (stream for stream in container.streams if stream.type == "audio"),
            None,
        )
        if audio_stream is None:
            return 0.0, 0.0, 0.0

        for frame in container.decode(audio_stream):
            sample_rate = int(frame.sample_rate or sample_rate or 0)
            array = frame.to_ndarray().astype(np.float32, copy=False)

            if array.ndim > 1:
                array = array.mean(axis=0)

            # Integer audio needs normalising; float audio is already -1..1.
            if np.issubdtype(frame.to_ndarray().dtype, np.integer):
                max_value = float(np.iinfo(frame.to_ndarray().dtype).max)
                if max_value > 0:
                    array = array / max_value

            chunks.append(array.reshape(-1))

    if not chunks:
        return 0.0, 0.0, 0.0

    samples = np.concatenate(chunks)
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    duration = samples.size / sample_rate if sample_rate else 0.0
    rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
    peak = float(np.max(np.abs(samples)))
    return duration, rms, peak


def collect_segments(segments: Iterable) -> tuple[str, float, float, int]:
    texts: list[str] = []
    weighted_log_probability = 0.0
    maximum_no_speech_probability = 0.0
    segment_count = 0

    for segment in segments:
        text = (segment.text or "").strip()
        if not text:
            continue

        texts.append(text)
        segment_count += 1
        weighted_log_probability += float(
            getattr(segment, "avg_logprob", -99.0)
        )
        maximum_no_speech_probability = max(
            maximum_no_speech_probability,
            float(getattr(segment, "no_speech_prob", 0.0)),
        )

    average_log_probability = (
        weighted_log_probability / segment_count
        if segment_count
        else -99.0
    )

    return (
        " ".join(texts).strip(),
        average_log_probability,
        maximum_no_speech_probability,
        segment_count,
    )


def emit_empty_result(
    audio_path: Path,
    started: float,
    language: str = "unknown",
    language_probability: float = 0.0,
    reason: str = "No clear speech detected. Listening...",
) -> None:
    emit(
        {
            "type": "result",
            "result": {
                "language": language,
                "language_probability": language_probability,
                "original": "",
                "english": "",
                "processing_seconds": time.perf_counter() - started,
                "source_path": str(audio_path),
            },
        }
    )
    status(reason)


def transcribe(path: str) -> None:
    if model is None:
        raise RuntimeError(
            "The translation model has not been loaded."
        )

    audio_path = Path(path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Captured audio file was not found: {audio_path}"
        )

    started = time.perf_counter()

    # Reject silent/tiny WAV files before spending GPU time on Whisper.
    duration, rms, peak = audio_rms_and_peak(audio_path)
    if duration < 0.45 or rms < 0.0045 or peak < 0.018:
        emit_empty_result(
            audio_path,
            started,
            reason="Background noise ignored. Listening...",
        )
        return

    status("Checking captured speech...")

    original_segments, info = model.transcribe(
        str(audio_path),
        task="transcribe",
        language=None,
        beam_size=1,
        best_of=1,
        temperature=0.0,
        initial_prompt=CS2_INITIAL_PROMPT,
        vad_filter=True,
        vad_parameters={
            "threshold": 0.62,
            "min_speech_duration_ms": 300,
            "min_silence_duration_ms": 300,
            "speech_pad_ms": 120,
        },
        no_speech_threshold=0.55,
        log_prob_threshold=-0.8,
        compression_ratio_threshold=2.2,
        condition_on_previous_text=False,
        without_timestamps=False,
    )

    (
        original,
        average_log_probability,
        maximum_no_speech_probability,
        segment_count,
    ) = collect_segments(original_segments)

    detected_language = info.language or "unknown"
    language_probability = float(
        info.language_probability or 0.0
    )

    rejected = (
        segment_count == 0
        or not original
        or average_log_probability < -0.85
        or maximum_no_speech_probability > 0.72
        or language_probability < 0.30
        or looks_like_hallucination(original)
    )

    if rejected:
        emit_empty_result(
            audio_path,
            started,
            detected_language,
            language_probability,
            "Unclear audio ignored. Listening...",
        )
        return

    status(
        f"Speech detected as {detected_language}. "
        "Translating to English..."
    )

    english_segments, _ = model.transcribe(
        str(audio_path),
        task="translate",
        language=(
            None
            if detected_language == "unknown"
            else detected_language
        ),
        beam_size=1,
        best_of=1,
        temperature=0.0,
        initial_prompt=CS2_INITIAL_PROMPT,
        vad_filter=True,
        vad_parameters={
            "threshold": 0.62,
            "min_speech_duration_ms": 300,
            "min_silence_duration_ms": 300,
            "speech_pad_ms": 120,
        },
        no_speech_threshold=0.55,
        log_prob_threshold=-0.8,
        compression_ratio_threshold=2.2,
        condition_on_previous_text=False,
        without_timestamps=False,
    )

    (
        english,
        english_average_log_probability,
        english_no_speech_probability,
        english_segment_count,
    ) = collect_segments(english_segments)

    if (
        english_segment_count == 0
        or not english
        or english_average_log_probability < -0.85
        or english_no_speech_probability > 0.72
        or looks_like_hallucination(english)
    ):
        emit_empty_result(
            audio_path,
            started,
            detected_language,
            language_probability,
            "Unreliable translation ignored. Listening...",
        )
        return

    processing_seconds = time.perf_counter() - started

    emit(
        {
            "type": "result",
            "result": {
                "language": detected_language,
                "language_probability": language_probability,
                "original": original,
                "english": english,
                "processing_seconds": processing_seconds,
                "source_path": str(audio_path),
            },
        }
    )

    status("Translation complete. Listening...")

def main() -> None:
    status("AI backend started.")

    if sys.stdin is None:
        emit(
            {
                "type": "error",
                "message": (
                    "The backend input stream is unavailable."
                ),
            }
        )
        return

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()

        if not raw_line:
            continue

        try:
            request = json.loads(raw_line)
            command = request.get("command")

            if command == "load":
                load_model(
                    request.get("model", "large-v3"),
                    request.get(
                        "compute_type",
                        "float16",
                    ),
                )

            elif command == "transcribe":
                transcribe(request["path"])

            elif command == "ping":
                status(
                    "ready"
                    if model is not None
                    else "not loaded"
                )

            elif command == "shutdown":
                status("AI backend shutting down.")
                break

            else:
                raise ValueError(
                    f"Unknown command: {command}"
                )

        except Exception as exc:
            report_error(exc)


if __name__ == "__main__":
    main()
