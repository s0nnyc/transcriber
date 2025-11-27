from importlib import resources
print("CUBLAS_LIB=", resources.files("nvidia.cublas") / "lib")
print("CUDNN_LIB=", resources.files("nvidia.cudnn") / "lib")