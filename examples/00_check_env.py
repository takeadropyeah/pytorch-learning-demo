"""00 · 环境自检

先跑这个文件，确认 PyTorch 装对了、显卡能被用上。
全部输出正常，你的环境就准备好了。
"""

import platform
import sys
import time

import torch

print("=" * 60)
print(f"Python      : {sys.version.split()[0]}")
print(f"系统         : {platform.system()} {platform.release()}")
print(f"解释器       : {sys.executable}")
print(f"PyTorch     : {torch.__version__}")
print(f"编译用 CUDA  : {torch.version.cuda}")
print("=" * 60)

cuda_ok = torch.cuda.is_available()
print(f"CUDA 可用    : {cuda_ok}")

if cuda_ok:
    print(f"显卡         : {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"显存         : {props.total_memory / 1024**3:.1f} GB")
    print(f"算力         : sm_{props.major}{props.minor}")
else:
    print("[!] 只能跑 CPU。如果这是意外，检查是否装成了 CPU 版 torch。")

# ---------------------------------------------------------------- 实机算一下
print("\n--- 矩阵乘法实测（4096×4096，取 5 次平均）---")


def benchmark(device: str) -> float:
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)

    if device == "cuda":
        torch.cuda.synchronize()  # GPU 是异步的，先等它排空

    t0 = time.perf_counter()
    for _ in range(5):
        _ = a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 5


t_cpu = benchmark("cpu")
print(f"CPU 单次      : {t_cpu * 1000:8.2f} ms")

if cuda_ok:
    t_gpu = benchmark("cuda")
    print(f"GPU 单次      : {t_gpu * 1000:8.2f} ms")
    print(f"加速比        : {t_cpu / t_gpu:8.1f}×")

print("\n[OK] 环境就绪" if cuda_ok else "\n[OK] 环境可用（CPU 模式）")
