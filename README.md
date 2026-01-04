# Faster-Whisper Transcription Pipeline

GPU-accelerated transcription pipeline built around **faster-whisper** and CUDA.

It processes `.mkv` (or other media) files from a fixed input directory and generates text transcripts. The project is designed so you can recreate the whole environment after an OS reinstall using `requirements.txt` and a minimal CUDA setup script.

---

## 1. Overview

Main ideas:

- Use **faster-whisper** on NVIDIA GPU for fast, high-quality transcription.
- Keep all Python dependencies in a virtual environment and freeze them into `requirements.txt`.
- Store minimal CUDA runtime config (`cuda_env.sh`) so the project remains reproducible.
- Track everything in a GitHub repository.

---

## 2. Project structure

Example layout (adapt names to your actual project):

```text
.
├─ video_files/              # input videos (ignored by git)
├─ transcripts_out/          # output transcripts (ignored by git or committed selectively)
├─ .venv/                    # Python virtual environment (not committed)
├─ requirements.txt          # frozen Python dependencies
├─ cuda_env.sh               # env vars for CUDA runtime libs
├─ README.md
├─ .gitignore
└─ docs/
   ├─ cuda-packages.txt      # optional: dpkg -l | grep cuda/nvidia
   └─ nvidia-smi.txt         # optional: nvidia-smi snapshot
```

---

## 3. Prerequisites

### Hardware

- NVIDIA GPU with enough VRAM for the chosen Whisper model.
- Stable power and cooling.

### Software (Linux)

- Recent Linux distro (e.g. Ubuntu 22.04/24.04).
- NVIDIA driver installed and working:
  ```bash
  nvidia-smi
  ```
- Python 3.10+ (project currently uses Python 3.12 in the venv).
- Git (for cloning and version control).

---

## 4. Initial setup (first time)

Clone the repository and prepare the environment:

```bash
git clone https://github.com/s0nnyc/transcripts.git
cd transcripts

# Check your python version
python3 --version

# Create venv (update python version based on previous command)
sudo apt install python3.12-venv
python3 -m venv .venv
source .venv/bin/activate

# Install all Python deps
pip install -r requirements.txt
```

If you do not have a `requirements.txt` yet and you are starting from scratch:

```bash
pip install faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12
pip freeze > requirements.txt
```

Then commit `requirements.txt` to the repo:

```bash
git add requirements.txt
git commit -m "Add requirements.txt"
git push
```

---

## 5. CUDA library environment

The project uses **pip-installed CUDA runtime libraries** (cuBLAS, cuDNN) instead of relying purely on the system CUDA toolkit.

The helper script `cuda_env.sh` sets `LD_LIBRARY_PATH` to point at the libraries inside the venv. Example contents (paths must match your machine):

```bash
# cuda_env.sh
# CUDA / NVIDIA runtime libraries for this project
# Activate venv first: source .venv/bin/activate
# Replace ABSOLUTE PATH with terminal output from this:
#   python - << 'EOF'
#   from importlib import resources
#   print("CUBLAS_LIB=", resources.files("nvidia.cublas") / "lib")
#   print("CUDNN_LIB=", resources.files("nvidia.cudnn") / "lib")
#   EOF

# IMPORTANT: mind the space before /home so you don't end up like this
# export CUBLAS_LIB= "/home/your_pc/...
# export CUDNN_LIB= "/home/your_pc/..

export CUBLAS_LIB="/ABS/PATH/TO/.venv/lib/python3.12/site-packages/nvidia/cublas/lib"
export CUDNN_LIB="/ABS/PATH/TO/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="$CUBLAS_LIB:$CUDNN_LIB:$LD_LIBRARY_PATH"
```

Usage:

```bash
source .venv/bin/activate
source cuda_env.sh
python your_script.py
```

If you change Python version or recreate the venv in a different location, update `CUBLAS_LIB` and `CUDNN_LIB` inside `cuda_env.sh`.

---

## 6. Running the transcription

Typical usage pattern (adapt script / paths to your code):

```bash
source .venv/bin/activate
source cuda_env.sh

python src/transcribe.py   --input-dir video_files   --output-dir transcripts_out   --model medium   --language en
```

Notes:

- Ensure `video_files/` contains `.mkv` or other supported media files.
- The transcription script should be written to skip corrupted files and continue with the rest.
- Adjust CLI arguments and paths to match your actual script.

---

## 7. Freezing dependencies for reproducibility

Whenever you add or upgrade Python libraries in the venv:

```bash
source .venv/bin/activate
pip install <new-packages>
pip freeze > requirements.txt
```

Commit the updated `requirements.txt` so future setups match:

```bash
git add requirements.txt
git commit -m "Update Python dependencies"
git push
```

On a new machine or after OS reinstall, you can rebuild the environment with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
source cuda_env.sh
```

This restores the same Python environment you had when `requirements.txt` was generated.

---

## 8. Git usage (including Git Bash and commits)

### 8.1 Initialize and first commit (if repo is not yet created)

From the project root:

```bash
git init
git add .
git commit -m "Initial commit: faster-whisper transcripts project"
```

Create a new empty repo on GitHub (no README, no .gitignore) and add it as remote:

```bash
git branch -M main
git remote add origin https://github.com/s0nnyc/transcripts.git
git push -u origin main
```

### 8.2 Working with the repo using Git Bash (Windows or Linux)

From Git Bash or any shell:

```bash
# Clone the repo
git clone https://github.com/s0nnyc/transcripts.git
cd transcripts

# Check current status
git status

# Stage changes
git add <file1> <file2>
# or everything
git add .

# Commit with a message
git commit -m "Describe what you changed"

# Push to GitHub
git push

# Pull latest changes
git pull
```

### 8.3 .gitignore

A typical `.gitignore` for this project:

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd

# Virtual environments
.venv/
venv/
env/

# Editor / IDE
.vscode/
.idea/
*.iml

# OS
.DS_Store
Thumbs.db

# Project-specific
video_files/
transcripts_out/
*.mkv
*.mp4
*.wav
*.log

# CUDA cache (if any)
.nv/
```

Do not commit input media, output transcripts (unless you explicitly want some), or the venv.

---

## 9. Recreating the environment after OS reinstall

After reinstalling Linux (or on a new machine):

```bash
# 1) Clone repo
git clone https://github.com/s0nnyc/transcriber.git
cd transcriber

# 2) Create venv
python3 -m venv .venv
source .venv/bin/activate

# 3) Install Python deps
pip install -r requirements.txt

# 4) Adjust CUDA env script if needed
nano cuda_env.sh          # fix paths if Python/venv path changed
source cuda_env.sh

# 5) Run your script
python src/transcribe.py ...
```

Optionally use `docs/nvidia-smi.txt` and `docs/cuda-packages.txt` as reference for which GPU driver and CUDA-related packages worked on the old system.

---

## 10. Notes and troubleshooting

- If you see errors like:
  - `Unable to load any of {libcudnn_ops.so...}`
  - `Library libcublas.so.12 is not found or cannot be loaded`

  check that:

  ```bash
  source .venv/bin/activate
  pip show nvidia-cublas-cu12 nvidia-cudnn-cu12
  ```

  and confirm `cuda_env.sh` points to the correct `lib` directories, and you ran:

  ```bash
  source .venv/bin/activate
  source cuda_env.sh
  ```

- If `torch.cuda.is_available()` (or equivalent) is `False`:
  - Verify NVIDIA driver installation with `nvidia-smi`.
  - Ensure you are not in an environment without GPU passthrough (e.g. WSL without GPU support).
  - Check for conflicting CUDA installs or broken drivers.

- If a particular `.mkv` file fails with “invalid data”:
  - The container is likely corrupted.
  - Remove or repair the file (e.g. with `ffmpeg`) and make your transcription script skip bad inputs instead of crashing.

Keep `docs/nvidia-smi.txt` and `docs/cuda-packages.txt` updated if you change drivers or CUDA versions so you always know which combination previously worked.
