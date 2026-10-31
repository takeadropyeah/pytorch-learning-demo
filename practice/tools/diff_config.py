"""对比两个 checkpoint 里保存的训练配置，逐项标出差异。

为什么需要这个工具（真实事故，2026-10-28）：
    判断"两次实验只差一个变量"不能靠记忆。
    那天拿 #14 当基线去跑 P5（加 Normalize），以为只改了一件事，
    实际 #14 是 `--erase 0` 跑的、P5 用了默认的 `erase=0.5`，
    于是"加了 Normalize"和"把 RandomErasing 加回来"同时发生。
    结论里的 +0.72 点是假的，真实只有 +0.27。

    这个脚本就是把"我以为"换成"我看到"。

用法（在 practice 目录下）：
    python tools\\diff_config.py ablation_resnet18 p5_norm
    python tools\\diff_config.py ablation_resnet18.pth p5_norm.pth
    python tools\\diff_config.py --list        # 列出所有存档

参数可以写 tag 片段（自动在 checkpoints 目录里模糊匹配），也可以写完整路径。
"""

import sys
from pathlib import Path

import torch

# Windows 控制台默认是 GBK，中文和特殊符号会直接抛 UnicodeEncodeError，
# 所以先把 stdout 强制成 UTF-8，替换掉无法编码的字符而不是崩掉。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, AttributeError):
    pass

CKPT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"


def resolve(name):
    """把 tag 片段或路径解析成一个 checkpoint 文件。"""
    p = Path(name)
    if p.suffix == ".pth" and p.exists():
        return p
    if p.exists() and p.is_file():
        return p

    hits = sorted(CKPT_DIR.glob(f"*{name}*.pth"))
    if not hits:
        raise SystemExit(f"找不到匹配 {name!r} 的存档，可用 --list 查看")
    if len(hits) > 1:
        print(f"{name!r} 匹配到多个存档，请写得更具体：", file=sys.stderr)
        for h in hits:
            print(f"    {h.name}", file=sys.stderr)
        raise SystemExit(2)
    return hits[0]


def load_meta(path):
    """只取 config 和少量元信息。weights_only=False 是必须的：
    存档里有 optimizer 状态，纯张量白名单加载会失败。"""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # 老版本 torch 没有 weights_only 参数
        ckpt = torch.load(path, map_location="cpu")
    return {
        "config": ckpt.get("config", {}) or {},
        "epoch": ckpt.get("epoch"),
        "best_acc": ckpt.get("best_acc"),
    }


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0

    if argv[0] == "--list":
        for p in sorted(CKPT_DIR.glob("*.pth")):
            print(f"  {p.name}")
        return 0

    if len(argv) != 2:
        raise SystemExit("需要两个参数：diff_config.py <tagA> <tagB>")

    pa, pb = resolve(argv[0]), resolve(argv[1])
    ma, mb = load_meta(pa), load_meta(pb)
    ca, cb = ma["config"], mb["config"]

    # 这两项不是实验变量：tag 只是文件名，resume 只影响从哪轮开始，不改训练行为。
    IGNORE = {"tag", "resume"}

    keys = sorted(set(ca) | set(cb))

    print(f"A = {pa.name}")
    print(f"B = {pb.name}")
    print()
    print(f"{'配置项':<16}{'A':<24}{'B':<24}")
    print("-" * 72)

    real, version_only = [], []
    for k in keys:
        if k in IGNORE:
            continue
        va, vb = ca.get(k), cb.get(k)
        sa = fmt(va) if k in ca else "(旧脚本无此项)"
        sb = fmt(vb) if k in cb else "(旧脚本无此项)"

        if k not in ca or k not in cb:
            version_only.append(k)
            mark = "  ~ 版本差异"
        elif va != vb:
            real.append(k)
            mark = "  <== 不同"
        else:
            mark = ""
        print(f"{k:<16}{sa:<24}{sb:<24}{mark}")

    print("-" * 72)
    if not real:
        print("实验变量完全一致 —— 这两次实验的差异不来自超参。")
    else:
        print(f"实验变量 {len(real)} 项不同：{', '.join(real)}")
        if len(real) > 1:
            print()
            print("  [!] 有多个变量同时改变了，这不是一次干净的单变量对照。")
            print(f"      要么补一次只改一个的实验，要么在结论里写明是"
                  f"「{'+'.join(real)} 的联合效果」，")
            print("      不要归因给其中任意一项。")

    if version_only:
        print()
        print(f"（另有 {len(version_only)} 项是脚本版本差异，不是人为改动的变量："
              f"{', '.join(version_only)}）")
        print("  旧脚本没这个开关时取的是当时的默认行为，要确认默认值是否与新版一致。")

    # 存档元信息：同一配置复现时，这三项能看出"训练有没有真跑完"
    print()
    for label, m in (("A", ma), ("B", mb)):
        ep = m["epoch"]
        best = m["best_acc"]
        best_s = f"{best * 100:.2f}%" if isinstance(best, (int, float)) else "-"
        print(f"{label} 存档：epoch={ep}  best_acc={best_s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
