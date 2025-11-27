from __future__ import annotations

import dataclasses
import shutil
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from colorama import Fore, Style, init
from faster_whisper import WhisperModel

init(autoreset=True)


# =====================
# KONFIGURÁCIA (EDITUJ)
# =====================
@dataclasses.dataclass(frozen=True)
class Config:
    # I/O
    input_dir: Path = Path("video_files")
    output_dir: Path = Path("transcripts")

    # Model
    model_name: str = "large-v3"          # "medium" | "small"
    device: str = "cuda"                  # "cpu" ak treba
    compute_type: str = "int8_float16"    # "float16" | "int8"

    # Transkripcia (interné chunkovanie faster-whisper)
    language: str = "sk"
    vad_filter: bool = True
    beam_size: int = 5
    chunk_length: int = 8                  # sekundy, znížené kvôli RAM špičkám
    no_speech_threshold: float = 0.6

    # Dĺžky/segmentácia
    use_external_segment: bool = True      # ak je ffmpeg, segmentuj externe
    segment_seconds: int = 300             # 5 min bloky
    external_segment_min_duration_s: int = 420  # file dlhší než X → segmentuj

    # Podporované prípony
    supported_exts: Tuple[str, ...] = (
        ".mp4", ".mkv", ".mov", ".webm", ".avi",
        ".mp3", ".m4a", ".wav", ".flac", ".ogg",
    )


CFG = Config()
# =====================


# ==========
# TIMING API
# ==========
class Timer:
    def __init__(self, label: str):
        self.label = label
        self._start = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        print(Fore.CYAN + f"[⏱] {self.label} …")
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self._start
        print(Fore.LIGHTBLUE_EX + f"[✓] {self.label} finished in {self.elapsed:.2f}s")


# =====================
# Pomocné util funkcie
# =====================
def resolve_ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg as iio  # type: ignore
        return iio.get_ffmpeg_exe()
    except Exception:
        return None


FFMPEG = resolve_ffmpeg()


def list_media(folder: Path, exts: Iterable[str]) -> List[Path]:
    if not folder.exists():
        print(Fore.RED + f"Input folder '{folder}' neexistuje.")
        raise SystemExit(1)

    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)
    if not files:
        ext_str = ", ".join(sorted(e.lstrip(".") for e in set(exts)))
        print(Fore.RED + f"Žiadne podporované súbory ({ext_str}) v '{folder}'.")
        raise SystemExit(1)

    stats: dict[str, int] = {}
    for f in files:
        ext = f.suffix.lower()
        stats[ext] = stats.get(ext, 0) + 1
    for ext, cnt in sorted(stats.items()):
        print(Fore.YELLOW + f"Found {cnt} {ext} file(s).")

    return files


def media_duration_seconds(path: Path) -> Optional[float]:
    """Získa dĺžku média cez PyAV (bez ffmpeg.exe). Zlyhanie → None."""
    try:
        import av  # PyAV
        with av.open(str(path)) as c:
            if c.duration is not None:
                # PyAV uvádza v mikrosekundách
                return c.duration / 1_000_000.0
            # fallback – prehľadaj audio stream
            for s in c.streams.audio:
                if s.duration and s.time_base:
                    return float(s.duration * s.time_base)
    except Exception:
        return None
    return None


def external_segment(src: Path, out_dir: Path, seconds: int) -> List[Path]:
    """
    Externé rozsekanie pomocou ffmpeg (ak je k dispozícii).
    Vráti zoznam segmentov; ak ffmpeg nie je, vráti [src] (bez segmentácie).
    """
    if not FFMPEG or not CFG.use_external_segment:
        return [src]

    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f"{src.stem}_seg_%03d{src.suffix.lower()}")
    # copy stream → bez recompress, nízka záťaž
    import subprocess

    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-y",
        "-i", str(src),
        "-f", "segment",
        "-segment_time", str(seconds),
        "-reset_timestamps", "1",
        "-c", "copy",
        pattern,
    ]
    try:
        with Timer(f"External segment (ffmpeg) {src.name} → {seconds}s chunks"):
            subprocess.run(cmd, check=True)
        segs = sorted(out_dir.glob(f"{src.stem}_seg_*.{src.suffix.lstrip('.').lower()}"))
        return segs or [src]
    except Exception as exc:
        print(Fore.RED + f"Segmentácia zlyhala, pokračujem bez nej: {exc}")
        return [src]


# =====================
# Transkripcia
# =====================
def transcribe_one(model: WhisperModel, src: Path, dst: Path) -> float:
    """
    Transkribuje jeden vstup. Ak je dlhý a dostupný ffmpeg, rozseká na segmenty.
    Inak zmenší interné chunkovanie a ide priamo.
    """
    # Rozhodni, či segmentovať
    dur = media_duration_seconds(src)
    should_segment = (
        CFG.use_external_segment
        and FFMPEG is not None
        and dur is not None
        and dur >= CFG.external_segment_min_duration_s
    )

    total_elapsed = 0.0
    dst.parent.mkdir(parents=True, exist_ok=True)

    if should_segment:
        seg_dir = CFG.output_dir / f"{src.stem}_segs"
        segments = external_segment(src, seg_dir, CFG.segment_seconds)
        with dst.open("w", encoding="utf-8") as out:
            for i, seg in enumerate(segments, 1):
                with Timer(f"Transcribe segment {i}/{len(segments)}: {seg.name}") as t:
                    iterator, _info = model.transcribe(
                        str(seg),
                        language=CFG.language,
                        vad_filter=CFG.vad_filter,
                        beam_size=CFG.beam_size,
                        chunk_length=CFG.chunk_length,   # stále malé
                        no_speech_threshold=CFG.no_speech_threshold,
                    )
                    text = "".join(s.text for s in iterator).strip()
                    out.write(text + "\n\n")
                total_elapsed += t.elapsed
        # cleanup segmentov
        for s in segments:
            try:
                s.unlink(missing_ok=True)
            except Exception:
                pass
        try:
            seg_dir.rmdir()
        except Exception:
            pass
    else:
        with Timer(f"Transcribe {src.name}") as t:
            iterator, _info = model.transcribe(
                str(src),
                language=CFG.language,
                vad_filter=CFG.vad_filter,
                beam_size=CFG.beam_size,
                chunk_length=CFG.chunk_length,       # malé chunkovanie, bráni 4–5 GiB allokáciám
                no_speech_threshold=CFG.no_speech_threshold,
            )
            text = "".join(s.text for s in iterator).strip()
        total_elapsed += t.elapsed
        with Timer(f"Save transcript → {dst.name}") as t_save:
            dst.write_text(text + "\n", encoding="utf-8")
        total_elapsed += t_save.elapsed

    return total_elapsed


def main() -> None:
    CFG.output_dir.mkdir(parents=True, exist_ok=True)

    with Timer("List media"):
        files = list_media(CFG.input_dir, CFG.supported_exts)

    print(Fore.CYAN + f"Loading faster-whisper '{CFG.model_name}' on {CFG.device} ({CFG.compute_type})")
    with Timer("Model load"):
        model = WhisperModel(CFG.model_name, device=CFG.device, compute_type=CFG.compute_type)
    print(Fore.GREEN + "✓ Model ready.")

    grand_total = 0.0
    for idx, src in enumerate(files, 1):
        out = CFG.output_dir / f"transcript_{src.stem}.txt"
        print(Style.BRIGHT + f"\n[{idx}/{len(files)}] {src.name}")
        grand_total += transcribe_one(model, src, out)

    print(Style.BRIGHT + Fore.GREEN + f"\n✅ All done. Total processing time: {grand_total:.2f}s")


if __name__ == "__main__":
    main()
