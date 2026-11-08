"""把 ViT-B/16 微调的显存拆成四块：参数 / 激活 / 梯度 / 优化器状态。

动机
----
`results_flowers.csv` 里只有一行「显存峰值 5.10 GB」。这个数字没法回答任何问题——
想知道「量化能省多少」「LoRA 省的是哪一块」「batch 翻倍会不会炸」，必须先把它拆开。

四块的反应完全不同：

    ┌──────────────┬──────────┬─────────────────────────────┐
    │ 参数         │ 0.34 GB  │ 4-bit 能压到 1/8            │
    │ 激活值       │ ~3.7 GB  │ 量化压不动，只有混合精度减半 │
    │ 梯度         │ 0.34 GB  │ 冻结就不存在                │
    │ 优化器状态   │ 0.69 GB  │ 冻结就不存在                │
    └──────────────┴──────────┴─────────────────────────────┘

测法
----
PyTorch 的 allocator 记账是分阶段累加的，所以每做一步量一次差值就能拆开：

    起点              -> 参数
    forward 之后峰值  -> 参数 + 激活
    backward 之后     -> 参数 + 梯度      （激活已被释放）
    optim.step 之后   -> 参数 + 梯度 + 优化器状态

注意第一个 batch 会额外带上 cuDNN/attention 的一次性 workspace 和 LRU 缓存，
所以先跑一个「热身 batch」再重置峰值统计。

用法
----
    python tools\\vram_breakdown.py                 # 3 组：fp32 / autocast bf16 / LoRA
    python tools\\vram_breakdown.py --batch-size 16
"""

import argparse
import gc
import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "practice"))

MB = 1024 ** 2
GB = 1024 ** 3


def load_base13():
    """复用 13_vit_finetune.py 的 load_data / build_model，避免复制粘贴。"""
    import importlib.util

    path = _REPO / "practice" / "13_vit_finetune.py"
    spec = importlib.util.spec_from_file_location("base13", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["base13"] = mod
    spec.loader.exec_module(mod)
    return mod


def gb(n):
    return f"{n / GB:6.2f} GB"


class _null:
    """空上下文，让 amp / 非 amp 两路共用同一份代码。"""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def measure(tag, model, batch, device, *, amp=False):
    """跑一步完整的 forward + backward + step，分四个断点记账。"""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)

    params = torch.cuda.memory_allocated(device)

    # ---- 热身：把一次性 workspace 与 allocator 缓存撑开，不计入结论
    with torch.no_grad():
        if amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(batch)
        else:
            model(batch)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    params = torch.cuda.memory_allocated(device)

    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else _null()

    # ---- 1. forward
    with ctx:
        logits = model(batch)
        loss = logits.float().mean()
    peak_fwd = torch.cuda.max_memory_allocated(device)
    act = peak_fwd - params

    # ---- 2. backward
    loss.backward()
    torch.cuda.synchronize()
    after_bwd = torch.cuda.memory_allocated(device)
    grads = after_bwd - params

    # ---- 3. optimizer step（AdamW 第一次 step 才分配两个动量态）
    trainable = [p for p in model.parameters() if p.requires_grad]
    if trainable:
        opt = torch.optim.AdamW(trainable, lr=1e-4)
        opt.step()
        torch.cuda.synchronize()
        after_step = torch.cuda.memory_allocated(device)
        opt_state = after_step - after_bwd
    else:                                    # 全冻结：没有梯度也没有优化器态
        after_step = after_bwd
        opt_state = 0

    total_peak = torch.cuda.max_memory_allocated(device)
    n_train = sum(p.numel() for p in trainable)

    print(f"\n【{tag}】 参数量 {sum(p.numel() for p in model.parameters()):,}"
          f" | 可训练 {n_train:,}")
    print(f"  参数         {gb(params):>10s}   {params / MB:8.1f} MB")
    print(f"  激活值(前向) {gb(act):>10s}   {act / MB:8.1f} MB")
    print(f"  梯度         {gb(grads):>10s}   {grads / MB:8.1f} MB")
    print(f"  优化器状态   {gb(opt_state):>10s}   {opt_state / MB:8.1f} MB")
    print(f"  {'-' * 34}")
    print(f"  实测峰值     {gb(total_peak):>10s}   {total_peak / MB:8.1f} MB")
    print(f"  训练结束后常驻 {gb(params + grads + opt_state):>10s}"
          f"   {(params + grads + opt_state) / MB:8.1f} MB")

    del logits, loss, model, batch
    if trainable:
        del opt
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "tag": tag,
        "params": params,
        "act": act,
        "grads": grads,
        "opt_state": opt_state,
        "total": total_peak,
        "resident": params + grads + opt_state,
    }


def freeze_all_but_head(model):
    """只留分类头可训练。"""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.heads.head.parameters():
        p.requires_grad = True
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--only", default="all",
                    choices=["all", "full", "amp", "head", "lora"],
                    help="只跑一个场景（用于逐个进程隔离测量）")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA —— 这套账在 CPU 上没有意义")

    device = torch.device(args.device)
    base13 = load_base13()

    print("=" * 74)
    print(f"ViT-B/16 微调显存拆解   batch={args.batch_size}   device={device}"
          f"   only={args.only}")
    print("=" * 74)

    batch = torch.randn(args.batch_size, 3, 224, 224, device=device)

    def scenario_full():
        return measure("全量微调 · fp32", base13.build_model(False, False, device),
                       batch, device)

    def scenario_amp():
        return measure("全量微调 · autocast bf16",
                       base13.build_model(False, False, device), batch, device,
                       amp=True)

    def scenario_head():
        return measure("只训分类头",
                       freeze_all_but_head(base13.build_model(False, False, device)),
                       batch, device)

    def scenario_lora():
        import importlib.util as iu

        spec = iu.spec_from_file_location("loramod", _REPO / "practice" / "lora.py")
        lm = iu.module_from_spec(spec)
        sys.modules["loramod"] = lm
        spec.loader.exec_module(lm)

        model = base13.build_model(False, False, device)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.heads.head.parameters():
            p.requires_grad = True
        n_swap = 0
        for name, mod in list(model.named_modules()):
            parts = name.split(".")
            if (isinstance(mod, nn.Linear) and ".mlp." in f".{name}."
                    and parts[-1] in ("0", "3")):
                parent = model
                for q in parts[:-1]:
                    parent = getattr(parent, q)
                setattr(parent, parts[-1], lm.LoRALinear(mod, r=8, alpha=16))
                n_swap += 1
        print(f"  （注入 {n_swap} 个 LoRALinear）")
        return measure("LoRA 落 FFN (r=8)", model.to(device), batch, device)

    table = {"full": scenario_full, "amp": scenario_amp,
             "head": scenario_head, "lora": scenario_lora}

    if args.only == "all":
        print("\n[提示] 逐个场景在同一进程里跑会互相污染 allocator，")
        print("       精确数字请用 --only <场景> 分进程跑。\n")
        rows = [fn() for fn in table.values()]
        _summary(rows)
    else:
        _summary([table[args.only]()])


def _summary(rows):
    print("\n" + "=" * 74)
    print("汇总（单位 GB）")
    print("=" * 74)
    print(f"{'场景':26s} {'参数':>7s} {'激活':>7s} {'梯度':>7s} {'优化器':>7s}"
          f" {'峰值':>7s} {'常驻':>7s}")
    print("-" * 74)
    for r in rows:
        print(f"{r['tag']:26s} {r['params']/GB:7.2f} {r['act']/GB:7.2f} "
              f"{r['grads']/GB:7.2f} {r['opt_state']/GB:7.2f} "
              f"{r['total']/GB:7.2f} {r['resident']/GB:7.2f}")

    full = rows[0]
    print("\n" + "=" * 74)
    print("怎么读这张表")
    print("=" * 74)
    print("· 峰值 ≈ 参数 + 激活（前向那一刻），而不是四项相加 —— 反向过程中激活逐层释放，")
    print("  梯度是边算边填的，两者时间上重叠，所以四项之和会大于实际峰值。")
    print(f"· 全量微调的峰值 {full['total']/GB:.2f} GB 里，激活占 "
          f"{full['act']/full['total']*100:.0f}%；常驻（训练结束仍在的）只有 "
          f"{full['resident']/GB:.2f} GB。")
    print(f"· 冻结权重砍掉梯度 {full['grads']/GB:.2f} + 优化器 {full['opt_state']/GB:.2f}"
          f" GB，但激活一分不省（backbone 仍要前向存图）—— 这是 LoRA 的 3.70 GB。")
    print(f"· 4-bit 只压参数那一列：{full['params']/GB:.2f} GB -> "
          f"{full['params']/8/GB:.2f} GB。")
    print("· QLoRA 的算术（7B 模型上 8GB 卡）：")
    print("    4-bit 权重 3.5 GB + LoRA 梯度/状态 ~0.1 GB + 激活（小 batch + 梯度累积）")
    print("  前三列靠量化，第四列靠重算/梯度累积/减小 batch —— 两个旋钮不能互换。")


if __name__ == "__main__":
    main()
