from __future__ import annotations

import time
from collections import Counter
from datetime import timedelta
from pathlib import Path

from colorama import Fore, Style, init
from faster_whisper import WhisperModel

init(autoreset=True)

# ==== CONFIGURATION ====
WHISPER_MODEL = "large-v3"      # "medium", "small" if you want lower requirements
WHISPER_DEVICE = "cuda"         # "cpu" if you do not want GPU
COMPUTE_TYPE = "int8_float16"   # "float16" | "int8" | "int8_float16"
TRANSCRIPTION_LANGUAGE = "sk"   # Language code passed to faster-whisper
INPUT_FOLDER = Path("video_files")
OUTPUT_FOLDER = Path("transcripts_out")
DELETE_ORIGINAL_MKV = True
# =======================

# Supported media extensions for scanning input folders.
VIDEO_EXTS: set[str] = {".mkv", ".mp4", ".mov", ".avi", ".webm"}
AUDIO_EXTS: set[str] = {".m4a", ".wav", ".mp3", ".flac", ".aac", ".ogg", ".wma"}
SUPPORTED_EXTS: set[str] = VIDEO_EXTS | AUDIO_EXTS


# Discover all supported media files and fail fast when inputs are missing.
def find_media_files(input_folder: Path) -> list[Path]:
    if not input_folder.exists():
        print(Fore.RED + f"Input folder '{input_folder}' does not exist.")
        raise SystemExit(1)

    files = sorted(
        p for p in input_folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    if not files:
        human_exts = ", ".join(sorted({ext.lstrip(".") for ext in SUPPORTED_EXTS}))
        print(Fore.RED + f"No supported files ({human_exts}) in '{input_folder}'.")
        raise SystemExit(1)

    ext_counts = Counter(p.suffix.lower() for p in files)
    for ext, cnt in sorted(ext_counts.items()):
        print(Fore.YELLOW + f"Found {cnt} {ext} file(s).")

    return files


# Run faster-whisper on a single file and persist the transcript text.
def transcribe_audio(model: WhisperModel, media_path: Path, transcript_path: Path, progress: str) -> None:
    print(Fore.CYAN + f"{progress} Transcribing: {media_path.name}")
    start_time = time.perf_counter()

    segments, info = model.transcribe(
        str(media_path),
        language=TRANSCRIPTION_LANGUAGE,
        vad_filter=False,
        chunk_length=None,  # disable chunking so the model sees the full context
    )
    text = "".join(seg.text for seg in segments).strip()

    duration = timedelta(seconds=int(time.perf_counter() - start_time))
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(text, encoding="utf-8")

    print(Fore.GREEN + f"✓ Transcript saved to: {transcript_path}")
    print(
        Fore.LIGHTBLUE_EX
        + "🕒 Transcription took: "
        + f"{duration} | detected language: {getattr(info, 'language', 'n/a')}"
    )


# Remove any temporary or source files slated for deletion.
def cleanup(paths: list[Path]) -> None:
    if not paths:
        return
    print(Fore.CYAN + "Cleaning up temporary files...")
    for path in paths:
        try:
            path.unlink(missing_ok=True)
            print(Fore.YELLOW + f"Deleted: {path.name}")
        except Exception as exc:
            print(Fore.RED + f"Could not delete {path}: {exc}")


# Prepare the environment, load the model, and iterate through each media file.
def main() -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    media_files = find_media_files(INPUT_FOLDER)

    print(Fore.CYAN + f"Loading faster-whisper model '{WHISPER_MODEL}' on {WHISPER_DEVICE} ({COMPUTE_TYPE})")
    try:
        model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as exc:
        print(Fore.RED + f"Model load failed: {exc}")
        raise SystemExit(1)
    print(Fore.GREEN + "✓ Model ready.")

    total_files = len(media_files)
    for index, media_path in enumerate(media_files, start=1):
        transcript_file = OUTPUT_FOLDER / f"transcript_{media_path.stem}.txt"
        progress = f"[{index}/{total_files}]"

        try:
            # faster-whisper (PyAV) reads the audio track directly; no separate extraction is needed.
            transcribe_audio(model, media_path, transcript_file, progress)

            if DELETE_ORIGINAL_MKV and media_path.suffix.lower() == ".mkv":
                cleanup([media_path])

        except Exception as exc:
            print(Fore.RED + f"Error with '{media_path.name}': {exc}")
            continue

    print(Fore.GREEN + Style.BRIGHT + "\n✅ All done! Check your transcripts!")


if __name__ == "__main__":
    main()
