"""LoRA 的 ΔW 到底用了几个方向 —— 奇异值谱分析

    python tools/lora_rank_spectrum.py
    python tools/lora_rank_spectrum.py --plot

为什么做这个：之前 `tools/demo_lowrank.py` 只证明了「训练好的 W 几乎满秩」，
**从来没有量过任何训练出来的 ΔW**。笔记里那句「只有 ΔW 低秩」是推理，不是测量。

但要说清楚一件事：ΔW = B@A 的秩**按构造就 ≤ r**，所以「低秩」是废话。
真正值得问的是第二个问题：

    **在那 r 个方向里，能量是怎么分布的？**

- 如果谱衰减很快 → 模型只用了少数几个方向，r 里有很多是浪费的
  → 这正好解释 r 扫描的收益递减（r=4 → 1.6015，r=8 → 1.5962，r=16 → 1.5888）
- 如果谱接近平坦 → 模型想把能量铺满整个 r 维子空间，r 是真瓶颈

顺带能回答：r 从 4 涨到 16 时，新增的 12 个方向是不是几乎全零？

数据源：`checkpoints/` 下训练好的 LoRA 权重（ViT/Flowers102 和 GPT-2/莎士比亚），
不需要重新训练，纯读盘分析。
"""
import argparse
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
CKPT = _REPO / "practice" / "checkpoints"


# ============================================================================
# 读盘：两种格式
# ============================================================================
def _pairs_from_sd(sd: dict) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """从 state_dict 里抠出所有 (lora_A, lora_B) 对。

    要同时兼容两种命名：
      自己写的   ...mlp.0.lora_A           （AttributeError 里的裸张量名）
      peft 的    ...c_attn.lora_A.weight   （多一层 .weight）
    """
    out = []
    for k in sd:
        for suf_a, suf_b in (("lora_A", "lora_B"), ("lora_A.weight", "lora_B.weight")):
            if k.endswith(suf_a):
                stem = k[: -len(suf_a)]
                kb = stem + suf_b
                if kb in sd:
                    out.append((stem, sd[k].float(), sd[kb].float()))
                break
    return out


def load_checkpoint(path: Path):
    """返回 (名称, r, [(stem, A, B), ...])。r 从张量形状推断，不信配置里的数字。"""
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        sd = load_file(str(path))
    else:
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        sd = ck.get("state_dict") or ck.get("model") or ck
        sd = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}

    pairs = _pairs_from_sd(sd)
    if not pairs:
        return None
    # A 的形状是 (r, in)，所以 r = A.shape[0]
    r = pairs[0][1].shape[0]
    return path.parent.name if path.name == "adapter_model.safetensors" else path.stem, r, pairs


# ============================================================================
# 谱分析
# ============================================================================
def spectrum(pairs, max_rank: int = 16):
    """把所有模块的奇异值按秩位置对齐后求均值。

    归一化做法：每个模块先除以它自己的 σ_1（否则大层的谱会盖住小层），
    再按「第 k 个 / 第 1 个」跨模块平均。
    """
    acc = torch.zeros(max_rank)
    cnt = torch.zeros(max_rank)
    eff90, eff99, ratios, numranks = [], [], [], []
    for _, A, B in pairs:
        r = A.shape[0]
        # B@A 的形状是 (out, in)（比如 3072×768），svdvals 会给 min(out,in)=768 个值，
        # 但真正的秩最多只有 r，第 r+1 个往后全是浮点噪声。
        # 所以下面所有统计都必须在**前 r 个**里做，不能拿 s[-1]。
        M = (B @ A).double()
        s = torch.linalg.svdvals(M)
        if s.numel() == 0 or s[0] == 0:
            continue
        n = min(r, max_rank)
        acc[:n] += (s[:n] / s[0])
        cnt[:n] += 1
        energy = torch.cumsum(s**2, 0) / (s**2).sum()
        eff90.append(int((energy < 0.90).sum()) + 1)
        eff99.append(int((energy < 0.99).sum()) + 1)
        ratios.append(float(s[r - 1] / s[0]))          # ← 第 r 个，不是最后一个
        numranks.append(int((s > 1e-6 * s[0]).sum()))
    mean = (acc / cnt.clamp(min=1)).tolist()
    return {
        "spectrum": mean,
        "eff90": sum(eff90) / len(eff90),
        "eff99": sum(eff99) / len(eff99),
        "tail_ratio": sum(ratios) / len(ratios),
        "numrank": sum(numranks) / len(numranks),
        "n_modules": len(eff90),
    }


def find_all(pattern: str) -> list[Path]:
    found = list(CKPT.glob(pattern))
    found += list(CKPT.glob(f"*/{pattern}"))
    return sorted(p for p in found if p.is_file())


# ============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="LoRA ΔW 的奇异值谱")
    ap.add_argument("--plot", action="store_true", help="同时出图到 practice/lora_rank_spectrum.png")
    a = ap.parse_args()

    print("=" * 78)
    print("LoRA 的 ΔW = B@A 用了几个方向")
    print("=" * 78)

    groups = [
        ("ViT-B/16 · Flowers102", [p for p in find_all("*.pth") if "lora" in p.name]),
        ("GPT-2 small · 莎士比亚", [p for p in find_all("*lora*/adapter_model.safetensors")]),
    ]

    rows = []
    for title, paths in groups:
        if not paths:
            continue
        print(f"\n【{title}】")
        print(f"{'权重':34s} {'r':>2s} {'模块':>4s} "
              f"{'90%能量':>7s} {'99%能量':>7s} {'σ_r/σ_1':>8s} {'数值秩':>6s}")
        print("-" * 78)
        for p in paths:
            got = load_checkpoint(p)
            if not got:
                continue
            name, r, pairs = got
            st = spectrum(pairs)
            rows.append({"group": title, "name": name, "r": r, **st})
            print(f"{name[:34]:34s} {r:2d} {st['n_modules']:4d} "
                  f"{st['eff90']:7.2f} {st['eff99']:7.2f} "
                  f"{st['tail_ratio']:8.4f} {st['numrank']:6.1f}")

    if not rows:
        print("没找到任何 LoRA 权重")
        return 1

    print("\n" + "=" * 78)
    print("归一化谱（每个模块先除以自己的 σ_1，再跨模块平均）")
    print("=" * 78)
    print(f"{'权重':34s} " + " ".join(f"k={k}" .rjust(6) for k in range(1, 9)))
    for d in rows:
        sp = d["spectrum"][:8]
        print(f"{d['name'][:34]:34s} " + " ".join(f"{v:6.3f}" for v in sp))

    print("\n怎么读：")
    print("  · 90%能量/99%能量 是「要保留前几维才够」——单位是维，上限是 r")
    print("  · σ_r/σ_1 是最后一个奇异值占第一个的比例，越接近 0 说明尾巴越没用")
    print("  · 如果 r=16 的 90%能量维数 ≈ r=4 的，说明多给的 12 维基本被浪费")

    if a.plot:
        _plot(rows)
    return 0


def _short(name: str, r: int, group: str) -> str:
    """把冗长的 checkpoint 名压成一句能进图例的话。"""
    tag = "ViT" if "vit" in name else ("GPT2" if "gpt2" in name else "?")
    if "smoke" in name:
        extra = " 冒烟(1轮3步)"
    elif "15ep" in name:
        extra = " 15轮"
    elif "attnffn" in name:
        extra = " attn+FFN"
    elif "attn" in name:
        extra = " attn"
    elif "ffn" in name or "lora_r8_lora_r8" in name:
        extra = " FFN"
    else:
        extra = ""
    return f"{tag} r={r}{extra}"


def _plot(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    ax = axes[0]
    for d in rows:
        sp = d["spectrum"]
        ax.plot(range(1, len(sp) + 1), sp, marker="o", ms=3, alpha=0.85,
                label=_short(d["name"], d["r"], d["group"]))
    ax.set_xlabel("奇异值序号 k（按大小排序）")
    ax.set_ylabel(r"$\sigma_k / \sigma_1$")
    ax.set_title("训练后的 ΔW 谱：能量集中在少数方向\n（冒烟那条最平——训练不足时还没有集中）")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    styles = {"ViT-B/16 · Flowers102": "-", "GPT-2 small · 莎士比亚": "--"}
    for group, ls in styles.items():
        sub = [d for d in rows if d["group"] == group and "smoke" not in d["name"]]
        if not sub:
            continue
        rs = sorted({d["r"] for d in sub})
        for metric, mk, lab in (("eff90", "o", "90%"), ("eff99", "s", "99%")):
            ys = []
            for r in rs:
                vals = [d[metric] for d in sub if d["r"] == r]
                ys.append(sum(vals) / len(vals))
            ax.plot(rs, ys, ls, marker=mk, ms=6,
                    label=f"{group.split(' ')[0]} · {lab} 能量")
    rs_all = sorted({d["r"] for d in rows})
    ax.plot(rs_all, rs_all, "k:", lw=1.2, label="y = r（完全用满）")
    ax.set_xlabel("LoRA 的 r")
    ax.set_ylabel("所需维数")
    ax.set_title("r 翻倍时新增的方向被用上了吗\n（90% 那条几乎水平 = 收益递减）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = _REPO / "practice" / "lora_rank_spectrum.png"
    fig.savefig(out, dpi=150)
    print(f"\n图已存 → {out}")


if __name__ == "__main__":
    sys.exit(main())
