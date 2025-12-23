import torch

# 查看 PyTorch 版本
print("PyTorch 版本:", torch.__version__)

# 检查 CUDA 是否可用
print("CUDA 可用:", torch.cuda.is_available())

# 查看 CUDA 版本（PyTorch 编译时使用的 CUDA 版本）
print("PyTorch 使用的 CUDA 版本:", torch.version.cuda)

# 查看 GPU 数量和名称
if torch.cuda.is_available():
    print("GPU 数量:", torch.cuda.device_count())
    print("当前 GPU:", torch.cuda.get_device_name(0))
else:
    print("未检测到可用的 CUDA 设备")