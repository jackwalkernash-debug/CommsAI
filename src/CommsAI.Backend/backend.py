from __future__ import annotations

import json
import os
import site
import sys
import time
import traceback
from pathlib import Path
from typing import TextIO


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


def join_segments(segments) -> str:
    return " ".join(
        segment.text.strip()
        for segment in segments
        if segment.text and segment.text.strip()
    ).strip()


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

    status("Transcribing captured speech...")

    original_segments, info = model.transcribe(
        str(audio_path),
        task="transcribe",
        language=None,
        beam_size=5,
        best_of=5,
        temperature=0.0,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 250,
            "speech_pad_ms": 250,
        },
        condition_on_previous_text=False,
        initial_prompt=(
            "Counter-Strike 2 professional team communication. "
            "Common terms: A, B, mid, short, long, ramp, connector, "
            "heaven, hell, palace, apartments, banana, pit, window, "
            "jungle, stairs, CT, T spawn, bomb, AWP, flash, smoke, "
            "molotov, rotate, save, eco, force, one HP."
        ),
    )

    original = join_segments(original_segments)

    detected_language = info.language or "unknown"
    language_probability = float(
        info.language_probability or 0.0
    )

    if not original:
        processing_seconds = time.perf_counter() - started

        emit(
            {
                "type": "result",
                "result": {
                    "language": detected_language,
                    "language_probability": language_probability,
                    "original": "",
                    "english": "",
                    "processing_seconds": processing_seconds,
                    "source_path": str(audio_path),
                },
            }
        )

        status("No clear speech detected. Listening...")
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
        beam_size=5,
        best_of=5,
        temperature=0.0,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 250,
            "speech_pad_ms": 250,
        },
        condition_on_previous_text=False,
        initial_prompt=(
            "Counter-Strike 2 professional team communication. "
            "Translate literally into English. Preserve player counts, "
            "positions, bomb information, weapons, utility and commands. "
            "Do not embellish."
        ),
    )

    english = join_segments(english_segments)
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
