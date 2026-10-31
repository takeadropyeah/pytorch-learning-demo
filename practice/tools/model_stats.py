r"""
A1：统计每个模型的 参数量 / MACs / 单张推理延迟

为什么要有这个脚本
------------------
`results.csv` 前 11 列全是「训练配置」，只有 `test_acc` 一个结果。
有了成本三列，才能画出「精度-成本曲线」——那是所有效率类工作的基本盘。

三个指标的口径（很重要，别和别人的表格对不上）
------------------------------------------------
- params    : 参数量，`sum(p.numel() for p in model.parameters())`
- macs_g    : 乘加次数（G MACs），不是 FLOPs。1 MAC = 2 FLOPs
- infer_ms  : 单张（bs=1）前向延迟，毫秒，中位数

为什么用 MACs 而不是 FLOPs
    论文里的 chart 常写 FLOPs，但 thop/fvcore 这类工具算出来的其实是 MACs。
    直接标 GMACs 最不容易和别人差 2 倍。

为什么不用 pip install thop
    thop 干的事就是「注册 forward hook + 套公式累加」，一共几十行。
    这里自己实现一份，零依赖，也顺便看清它到底怎么数。
    （如果以后遇到本脚本不认识的算子，再换 thop / fvcore。）

用法
----
    python .\tools\model_stats.py              # 打印表格
    python .\tools\model_stats.py --fill       # 回填 results.csv 的三列
    python .\tools\model_stats.py --json p.json
"""

import sys
import time
import json
import argparse
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

BASE = Path(__file__).resolve().parent.parent   # practice/
sys.path.insert(0, str(BASE))


# --------------------------------------------------------------------------- #
# 1. 把三个脚本里的模型类「按文件路径」导入
#    文件名是 03_xxx.py，不能 `import 03_xxx`（标识符不能以数字开头），
#    所以用 importlib 按路径加载。三个脚本都有 __main__ 保护，导入不会触发训练。
# --------------------------------------------------------------------------- #
def load_module(filename, alias):
    path = BASE / filename
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}")
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod_cnn = load_module("03_cnn_cifar.py", "mod_cnn")
_mod_res = load_module("resnet_model.py", "mod_resnet")
_mod_lenet = load_module("06_lenet.py", "mod_lenet")


# --------------------------------------------------------------------------- #
# 2. 模型清单：csv 里的名字 -> (构造函数, 输入形状)
#    输入形状必须写真实的，因为 MACs 是「结构 × 输入尺寸」的产物。
#    CIFAR 是 32×32×3，MNIST 是 28×28×1，差一个都不是同一个数。
# --------------------------------------------------------------------------- #
MODEL_SPECS = {
    "SmallCNN":     (lambda: _mod_cnn.SmallCNN(),            (1, 3, 32, 32)),
    "ResNet18":     (lambda: _mod_res.ResNet18(plain=False), (1, 3, 32, 32)),
    "Plain18":      (lambda: _mod_res.ResNet18(plain=True),  (1, 3, 32, 32)),
    "ResNet34":     (lambda: _mod_res.build_resnet("34", plain=False), (1, 3, 32, 32)),
    "Plain34":      (lambda: _mod_res.build_resnet("34", plain=True),  (1, 3, 32, 32)),
    "LeNet5-MNIST": (lambda: _mod_lenet.LeNet5(),            (1, 1, 28, 28)),
}


# --------------------------------------------------------------------------- #
# 3. 测量
# --------------------------------------------------------------------------- #
def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_macs(model, shape):
    """返回 GMACs。用 forward hook 累加，模型和 dummy 都放 CPU。

    公式（和 thop 一致）：
        Conv2d : 输出元素数 × (in_channels / groups) × k_h × k_w
        Linear : in_features × 输出元素数（每对输入输出一次乘加）
        Norm   : 输出元素数（逐元素乘 γ 加 β，量级很小）
        Pool/激活/Flatten : 0（只有比较和取数，没有乘加）
    """
    model = model.to("cpu").eval()
    x = torch.randn(*shape)
    macs = 0

    def hook(module, inputs, output):
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            k = module.kernel_size[0] * module.kernel_size[1]
            macs += output.numel() * (module.in_channels // module.groups) * k
        elif isinstance(module, nn.Linear):
            macs += module.in_features * output.numel()
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
            macs += output.numel()

    handles = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
            handles.append(m.register_forward_hook(hook))

    with torch.inference_mode():
        model(x)

    for h in handles:
        h.remove()

    return macs / 1e9


def bench_latency(model, shape, device, iters=50, warmup=10):
    """返回单张前向延迟的中位数（毫秒）。"""
    model = model.to(device).eval()
    x = torch.randn(*shape, device=device)
    is_cuda = device.type == "cuda"

    with torch.inference_mode():
        for _ in range(warmup):          # 预热：让 cuDNN 选好算法、显存池起来
            model(x)
        if is_cuda:
            torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            if is_cuda:
                torch.cuda.synchronize()  # GPU 是异步的，必须夹住才测得准
            t0 = time.perf_counter()
            model(x)
            if is_cuda:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]       # 中位数：避开偶尔被系统抢占的离群值


def fmt_macs(g):
    """小模型用 M 显示，否则全是 0.000G 看不清。"""
    return f"{g * 1000:7.2f}M" if g < 0.1 else f"{g:7.3f}G"


def measure_all(device):
    rows = {}
    for name, (factory, shape) in MODEL_SPECS.items():
        p = count_params(factory())
        g = count_macs(factory(), shape)
        ms = bench_latency(factory(), shape, device)
        rows[name] = {"params": p, "macs_g": round(g, 6), "infer_ms": round(ms, 3)}
        print(f"  [{name:<13}] params={p:>12,}  MACs={fmt_macs(g)}  "
              f"infer={ms:>6.2f}ms", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# 4. 回填 results.csv
# --------------------------------------------------------------------------- #
def fill_results_csv(rows, csv_path):
    import csv

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        old_cols = reader.fieldnames
        records = list(reader)

    cost_cols = ["params", "macs_g", "infer_ms"]
    new_cols = [c for c in old_cols if c not in cost_cols] + cost_cols

    for r in records:
        m = rows.get(r["model"])
        if m is None:
            raise KeyError(f"results.csv 里有未知模型 {r['model']!r}，"
                           f"请在 MODEL_SPECS 里登记")
        r.update({k: m[k] for k in cost_cols})

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_cols)
        writer.writeheader()
        writer.writerows(records)

    print(f"\n已回填 {len(records)} 行 -> {csv_path}")


def main():
    ap = argparse.ArgumentParser(description="统计模型的 params / MACs / 推理延迟")
    ap.add_argument("--fill", action="store_true", help="回填 results.csv")
    ap.add_argument("--json", type=str, default="", help="把结果写成 json")
    ap.add_argument("--cpu-only", action="store_true", help="不测 GPU 延迟")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu_only or not torch.cuda.is_available() else "cuda")
    print(f"设备: {device}   (MACs 口径: 1 MAC = 2 FLOPs)\n")

    rows = measure_all(device)

    # 写成本表：训练脚本跑完后会自动读这个文件补三列
    costs_path = BASE / "tools" / "costs.json"
    costs_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n成本表已写入 {costs_path.relative_to(BASE)}（训练脚本会自动读取）")

    print(f"\n{'模型':<14}{'params':>13}{'MACs':>10}{'延迟(ms)':>10}   {'参数倍数':>9}")
    base = rows["SmallCNN"]["params"]
    for name, m in rows.items():
        print(f"{name:<14}{m['params']:>13,}{fmt_macs(m['macs_g'])}"
              f"{m['infer_ms']:>10.2f}   {m['params']/base:>8.1f}x")

    print("\n注意：bs=1 时 GPU 根本没吃满，延迟主要由 kernel 启动开销决定。"
          "\n      单张推理的地板大约 0.3~0.5ms，低于这个量级的差异都是噪声。")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n已写入 {args.json}")

    if args.fill:
        fill_results_csv(rows, BASE / "results.csv")


if __name__ == "__main__":
    main()
