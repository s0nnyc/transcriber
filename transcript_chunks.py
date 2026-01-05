from __future__ import annotations

import shutil
import subprocess
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
DELETE_ORIGINAL_MEDIA = True
CHUNK_LENGTH_SECONDS = 600       # 5-minute chunks balance memory use and overhead
DELETE_TEMP_CHUNKS = True
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
def split_media_into_chunks(media_path: Path, chunks_dir: Path) -> list[Path]:
    if not shutil.which("ffmpeg"):
        print(Fore.RED + "ffmpeg is required for chunking but was not found in PATH.")
        raise SystemExit(1)

    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_pattern = chunks_dir / "chunk_%05d.wav"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "segment",
        "-segment_time",
        str(CHUNK_LENGTH_SECONDS),
        "-reset_timestamps",
        "1",
        "-c:a",
        "pcm_s16le",
        str(chunk_pattern),
    ]
    print(Fore.CYAN + f"Splitting into chunks: {media_path.name}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(Fore.RED + f"ffmpeg failed while chunking '{media_path.name}': {exc}")
        raise SystemExit(1)

    chunks = sorted(chunks_dir.glob("chunk_*.wav"))
    if not chunks:
        print(Fore.RED + f"No chunks produced for '{media_path.name}'.")
        raise SystemExit(1)
    return chunks


def transcribe_audio(model: WhisperModel, media_path: Path, transcript_path: Path, progress: str) -> None:
    print(Fore.CYAN + f"{progress} Transcribing: {media_path.name}")
    start_time = time.perf_counter()

    chunks_dir = transcript_path.parent / f"chunks_{media_path.stem}"
    chunks = split_media_into_chunks(media_path, chunks_dir)

    text_parts: list[str] = []
    total_chunks = len(chunks)
    info = None
    for idx, chunk_path in enumerate(chunks, start=1):
        chunk_progress = f"{progress} [chunk {idx}/{total_chunks}]"
        print(Fore.LIGHTBLUE_EX + f"{chunk_progress} {chunk_path.name}")

        segments, info = model.transcribe(
            str(chunk_path),
            language=TRANSCRIPTION_LANGUAGE,
            vad_filter=False,
            chunk_length=None,
        )
        text_parts.append("".join(seg.text for seg in segments).strip())

    text = " ".join(part for part in text_parts if part).strip()

    duration = timedelta(seconds=int(time.perf_counter() - start_time))
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(text, encoding="utf-8")

    print(Fore.GREEN + f"✓ Transcript saved to: {transcript_path}")
    print(
        Fore.LIGHTBLUE_EX
        + "🕒 Transcription took: "
        + f"{duration} | detected language: {getattr(info, 'language', 'n/a')}"
    )

    if DELETE_TEMP_CHUNKS:
        cleanup(list(chunks_dir.glob("chunk_*.wav")))
        try:
            chunks_dir.rmdir()
        except OSError:
            pass


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

            if DELETE_ORIGINAL_MEDIA and media_path.suffix.lower() in SUPPORTED_EXTS:
                cleanup([media_path])

        except Exception as exc:
            print(Fore.RED + f"Error with '{media_path.name}': {exc}")
            continue

    print(Fore.GREEN + Style.BRIGHT + "\n✅ All done! Check your transcripts!")


if __name__ == "__main__":
    main()
