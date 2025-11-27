# Faster-Whisper Transcription Pipeline

This project is a GPU-accelerated transcription pipeline built around **faster-whisper** and CUDA.
It processes `.mkv` (or other media) files from a fixed input directory and generates text transcripts.

The repository contains:

- Python scripts for batch transcription.
- A reproducible Python environment (`requirements.txt`).
- A helper script (`cuda_env.sh`) to make CUDA runtime libraries (cuBLAS, cuDNN) visible.
- Optional documentation about the GPU/driver stack under `docs/`.

---

## 1. Features

- Uses **faster-whisper** for fast, high-quality transcription.
- Runs on NVIDIA GPU via CUDA.
- Processes video/audio files from a source directory and writes transcripts to an output directory.
- Can be automated and run remotely over SSH.

---

## 2. Requirements

### Hardware

- NVIDIA GPU with sufficient VRAM for the chosen Whisper model.
- Stable power and cooling (transcription can be compute-intensive).

### Software (Linux)

- Recent Linux distro (e.g. Ubuntu 22.04/24.04).
- NVIDIA driver installed and working (`nvidia-smi` shows your GPU).
- Python 3.10+ (project currently uses Python 3.12 in the venv).
- Git (for cloning and version control).

### Python libraries

All Python dependencies are listed in `requirements.txt`. They include, among others:

- `faster-whisper`
- `ctranslate2`
- `nvidia-cublas-cu12`
- `nvidia-cudnn-cu12`

---

## 3. Project structure

Example layout:

```text
.
├─ src/                      # transcription and helper scripts (optional)
├─ video_files/              # input videos (ignored by git)
├─ transcripts_out/          # output transcripts (ignored by git or committed selectively)
├─ .venv/                    # Python virtual environment (not committed)
├─ requirements.txt          # frozen Python dependencies
├─ cuda_env.sh               # environment variables for CUDA runtime libs
├─ README.md
├─ .gitignore
└─ docs/
   ├─ cuda-packages.txt      # optional: dpkg -l | grep cuda/nvidia
   └─ nvidia-smi.txt         # optional: nvidia-smi snapshot
