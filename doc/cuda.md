# 什么是cuda?
CUDA（Compute Unified Device Architecture）是NVidia公司开发的一个用于GPU计算的并行计算平台。CUDA是GPU编程的扩展，CUDA的编程模型基于C/C++，CUDA的编程模型与OpenCL类似，CUDA的编程模型与OpenMP类似。CUDA的编程模型与OpenMP类似。CUDA的编程模型与OpenMP类似。CUDA的编程模型与OpenMP类似。CUDA的编程模型与OpenMP类似。CUDA的编程模型与OpenMP类似。

通俗的说：
CUDA 是 NVIDIA 提供的一个“让程序员能用 GPU 做通用计算”的技术平台。

- GPU 有成千上万个小型计算核心，特别适合并行计算（比如同时处理百万个数字）

- CPU 只有几个到几十个核心，擅长逻辑控制，但不适合大规模重复计算


# 查看机器是否支持cuda
- cmd  输入 nvidia-smi
- nvidia官网 https://developer.nvidia.com/cuda-gpus

# 安装

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

