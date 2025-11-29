# CUDA / NVIDIA runtime libraries for this project
# Activate venv first: source .venv/bin/activate
# Replace ABSOLUTE PATH with terminal output from this:
#   python - << 'EOF'
#   from importlib import resources
#   print("CUBLAS_LIB=", resources.files("nvidia.cublas") / "lib")
#   print("CUDNN_LIB=", resources.files("nvidia.cudnn") / "lib")
#   EOF

export CUBLAS_LIB=/home/altaira/python_projects/transcriber/.venv/lib/python3.12/site-packages/nvidia/cublas/lib
export CUDNN_LIB=/home/altaira/python_projects/transcriber/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH="$CUBLAS_LIB:$CUDNN_LIB:$LD_LIBRARY_PATH"
