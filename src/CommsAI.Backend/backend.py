from __future__ import annotations

import json
import os
import site
import sys
import time
import traceback
from pathlib import Path

# Make pip-provided CUDA DLLs discoverable before importing CTranslate2.
if os.name == "nt":
    roots = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if user_site:
        roots.append(user_site)

    exe_root = Path(sys.executable).resolve().parent
    roots.extend([str(exe_root), str(exe_root / "_internal")])

    for root in roots:
        base = Path(root)
        for relative in (
            Path("nvidia") / "cublas" / "bin",
            Path("nvidia") / "cudnn" / "bin",
            Path("_internal") / "nvidia" / "cublas" / "bin",
            Path("_internal") / "nvidia" / "cudnn" / "bin",
        ):
            folder = base / relative
            if folder.exists():
                try:
                    os.add_dll_directory(str(folder))
                except OSError:
                    pass
                os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")

from faster_whisper import WhisperModel

model: WhisperModel | None = None


def emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def status(message: str) -> None:
    emit({"type": "status", "message": message})


def load_model(model_name: str, compute_type: str) -> None:
    global model
    status(f"Loading {model_name} on the NVIDIA GPU. The first download can take several minutes…")
    model = WhisperModel(
        model_name,
        device="cuda",
        compute_type=compute_type,
        download_root=str(Path.home() / ".commsai" / "models"),
    )
    status("GPU model ready. Listening…")


def transcribe(path: str) -> None:
    if model is None:
        raise RuntimeError("The model has not been loaded.")

    started = time.perf_counter()

    original_segments, info = model.transcribe(
        path,
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
            "Common terms: A, B, mid, short, long, ramp, connector, heaven, hell, "
            "palace, apartments, banana, pit, window, jungle, stairs, CT, T spawn, "
            "bomb, AWP, flash, smoke, molotov, rotate, save, eco, force, one HP."
        ),
    )
    original = " ".join(
        segment.text.strip()
        for segment in original_segments
        if segment.text.strip()
    ).strip()

    if not original:
        emit({
            "type": "result",
            "result": {
                "language": info.language or "unknown",
                "language_probability": float(info.language_probability or 0.0),
                "original": "",
                "english": "",
                "processing_seconds": time.perf_counter() - started,
                "source_path": path,
            },
        })
        return

    english_segments, _ = model.transcribe(
        path,
        task="translate",
        language=info.language,
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
            "Translate literally. Preserve player counts, positions, bomb information, "
            "weapons, utility and commands. Do not embellish."
        ),
    )
    english = " ".join(
        segment.text.strip()
        for segment in english_segments
        if segment.text.strip()
    ).strip()

    emit({
        "type": "result",
        "result": {
            "language": info.language or "unknown",
            "language_probability": float(info.language_probability or 0.0),
            "original": original,
            "english": english,
            "processing_seconds": time.perf_counter() - started,
            "source_path": path,
        },
    })


def main() -> None:
    status("AI backend started.")

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
                    request.get("compute_type", "float16"),
                )
            elif command == "transcribe":
                transcribe(request["path"])
            elif command == "ping":
                status("ready" if model is not None else "not loaded")
            else:
                raise ValueError(f"Unknown command: {command}")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            emit({
                "type": "error",
                "message": str(exc),
            })


if __name__ == "__main__":
    main()
