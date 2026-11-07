"""台阶 4：用 peft 做 LoRA 微调 GPT-2，把 BPC 从零样本的 1.929 往下压。

这个脚本要回答三个问题
    1. LoRA 微调能把 BPC 降多少？        —— 对照 16_gpt2_bpc.py 测出的 1.929
    2. 它花了多少参数 / 多少显存？        —— 对照全量微调（台阶 6）
    3. 「预训练」和「目标域数据」各值多少？—— 补齐 2x2 表的右下格

2x2 归因表（本脚本填最后一格）
    ┌──────────────┬──────────────────┬────────────────────────┐
    │              │ 无预训练          │ 有预训练                │
    ├──────────────┼──────────────────┼────────────────────────┤
    │ 无目标域训练  │ 均匀 5.05        │ GPT-2 零样本 1.929      │
    │              │ unigram 4.78     │ （16_gpt2_bpc.py）      │
    │              │ bigram 3.58      │                        │
    ├──────────────┼──────────────────┼────────────────────────┤
    │ 有目标域训练  │ mini GPT 2.492   │ ← 本脚本                │
    │              │ （11_mini_gpt）   │                        │
    └──────────────┴──────────────────┴────────────────────────┘

**为什么用 peft 而不是接着手写**

手写 LoRA 的价值在于「知道里面每一个矩阵是什么」（lora.py 已经做过了）。
但 GPT-2 的权重不是 nn.Linear 而是 transformers 自己的 Conv1D：

    nn.Linear : weight (out, in)，forward 里 x @ W.T
    Conv1D    : weight (in, out)，forward 里 x @ W      ← GPT-2 用的是这个

15_gpt2_smoke.py 已经实测确认过这一点。所以「手写」那条路要先把矩阵方向
翻过来——那是台阶 5 的事，不是台阶 4 的。台阶 4 先用成熟的库拿到可信的数，
再用它当标尺去验台阶 5 的手写实现。

**target_modules 的匹配规则（这里最容易错）**

peft 匹配的是 `参数名.endswith("." + target)`，不是精确相等。
所以写 "c_proj" 会同时命中 attn.c_proj 和 mlp.c_proj（每个 block 两个），
而 GPT-2 的每个 block 里 c_proj 出现两次：

    transformer.h.0.attn.c_attn     (768, 2304)  QKV 融合
    transformer.h.0.attn.c_proj     (768,  768)  ← 注意力输出
    transformer.h.0.mlp.c_fc        (768, 3072)
    transformer.h.0.mlp.c_proj      (3072, 768)  ← 这是 MLP 的，不是注意力的

想只打注意力（论文 §4.2 的 Wq/Wk/Wv/Wo），必须写 "attn.c_proj"。
本脚本 --targets attn 就是这么写的，并且用公式交叉验证参数个数。

用法：
    python practice/17_gpt2_lora.py --max-steps 3        # 冒烟，约 30 秒
    python practice/17_gpt2_lora.py                      # 默认 r=8 attention，5 轮
    python practice/17_gpt2_lora.py --targets all --lora-r 16
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / ".cache" / "hf" / "gpt2"
CORPUS = ROOT / "data" / "shakespeare.txt"
CKPT_DIR = ROOT / "practice" / "checkpoints"
CSV_PATH = ROOT / "practice" / "results_gpt2.csv"

# 和 11_mini_gpt.py / 16_gpt2_bpc.py 完全相同的划分
MINI_GPT_VAL_START = 1_003_854
LN2 = math.log(2.0)

# GPT-2 每个 block 里四个 Conv1D 的 (in_features, out_features)
# 来自 15_gpt2_smoke.py 的实测：weight 形状是 (in, out)
CONV1D_SPECS = {
    "attn.c_attn": (768, 2304),
    "attn.c_proj": (768, 768),
    "mlp.c_fc": (768, 3072),
    "mlp.c_proj": (3072, 768),
}

TARGET_PRESETS = {
    # 论文 §4.2：只加 Wq/Wk/Wv 和 Wo 的 LoRA（c_attn 是 QKV 融合，算一个）
    "attn": ["c_attn", "attn.c_proj"],
    # 注意力 + MLP，4 个矩阵全加
    "all": ["c_attn", "attn.c_proj", "c_fc", "mlp.c_proj"],
}


def expected_lora_params(preset: str, r: int, n_layer: int) -> int:
    """按 (in + out) * r 手算 LoRA 参数个数，用来交叉验证 peft 的实际结果。"""
    targets = TARGET_PRESETS[preset]
    per_layer = 0
    for name, (fan_in, fan_out) in CONV1D_SPECS.items():
        if any(name.endswith(t) for t in targets):
            per_layer += r * (fan_in + fan_out)
    return per_layer * n_layer


def lr_at(step: int, total: int, warmup: int, base_lr: float, min_ratio: float = 0.1) -> float:
    """线性 warmup + 余弦退火到 min_ratio * base_lr。"""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cos)


def get_batch(ids: torch.Tensor, block: int, batch: int, device: str, gen: torch.Generator):
    """随机切 windows；y 是 x 右移一位（自回归目标）。"""
    ix = torch.randint(len(ids) - block - 1, (batch,), generator=gen)
    x = torch.stack([ids[i : i + block] for i in ix]).to(device)
    y = torch.stack([ids[i + 1 : i + block + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def eval_bpc(
    model,
    ids: torch.Tensor,
    n_chars: int,
    block: int,
    batch: int,
    device: str,
    max_windows: int = 0,
    tag: str = "",
) -> tuple[float, float, int]:
    """把 16_gpt2_bpc.py 的算法原样搬过来，保证两次测的是同一个东西。

    返回 (BPC, nats_per_token, 预测位置数)。
    BPC 用的是「按 token 数换算」那个口径：
        BPC = (nats/token) / ln2 / (字符/token)
    """
    model.eval()
    L = min(block, model.config.n_positions)
    starts = list(range(0, len(ids) - L - 1, L))
    if max_windows:
        starts = starts[:max_windows]

    total_nll = 0.0
    total_tokens = 0
    t0 = time.time()
    for bi in range(0, len(starts), batch):
        chunk = starts[bi : bi + batch]
        xs = torch.stack([ids[s : s + L] for s in chunk]).to(device)
        ys = torch.stack([ids[s + 1 : s + L + 1] for s in chunk]).to(device)
        logits = model(input_ids=xs).logits
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), ys.reshape(-1), reduction="sum"
        )
        total_nll += float(nll)
        total_tokens += ys.numel()

    nats_per_token = total_nll / max(total_tokens, 1)
    chars_per_token = n_chars / len(ids)
    bpc = nats_per_token / LN2 / chars_per_token
    print(
        f"    {tag}BPC {bpc:.4f}  nats/token {nats_per_token:.4f}  "
        f"{total_tokens:,} 位置  {time.time() - t0:.1f}s",
        flush=True,
    )
    model.train()
    return bpc, nats_per_token, total_tokens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=0, help="0 = 用 2*r")
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--targets", choices=sorted(TARGET_PRESETS), default="attn")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-steps", type=int, default=0, help="只跑 N 步，0=不限制（冒烟用）")
    ap.add_argument("--eval-windows", type=int, default=0, help="评测只用前 N 窗口，0=全跑")
    ap.add_argument("--no-eval-base", action="store_true", help="跳过零样本基线评测")
    ap.add_argument("--gen", action="store_true", help="训练后生成一段看风格变化")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lora_alpha = args.lora_alpha or 2 * args.lora_r
    tag = args.tag or f"lora_r{args.lora_r}_{args.targets}"

    print("=" * 70)
    print(f"台阶 4：peft LoRA 微调 GPT-2   tag={tag}")
    print("=" * 70)

    # ---------- 1. 数据 ----------
    tok = GPT2TokenizerFast.from_pretrained(MODEL_DIR, local_files_only=True)
    text = CORPUS.read_text(encoding="utf-8")
    train_text = text[:MINI_GPT_VAL_START]
    val_text = text[MINI_GPT_VAL_START:]
    train_ids = torch.tensor(tok(train_text)["input_ids"], dtype=torch.long)
    val_ids = torch.tensor(tok(val_text)["input_ids"], dtype=torch.long)
    print(f"  训练 {len(train_text):,} 字符 → {len(train_ids):,} tokens")
    print(f"  验证 {len(val_text):,} 字符 → {len(val_ids):,} tokens（与 mini GPT 同一段，不参与训练）")

    # ---------- 2. 基座 + 包 LoRA ----------
    base = GPT2LMHeadModel.from_pretrained(MODEL_DIR, local_files_only=True).to(device)
    total_params = sum(p.numel() for p in base.parameters())

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=TARGET_PRESETS[args.targets],
        bias="none",  # 论文：训 bias 没有收益
    )
    model = get_peft_model(base, lora_cfg).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = expected_lora_params(args.targets, args.lora_r, base.config.n_layer)
    print(f"  总参数     {total_params:,}")
    print(f"  可训练     {trainable:,}  ({trainable / total_params:.4%})")
    print(f"  公式预期   {expected:,}   r×(in+out)×{base.config.n_layer} 层")
    if trainable != expected:
        print(f"  ⚠ 不一致（差 {trainable - expected:,}）——target_modules 可能多打或少打了模块")
    else:
        print("  ✅ 与手算公式一致")

    # 逐层确认打到了哪些 Conv1D，避免 "c_proj" 误伤 mlp.c_proj
    # peft 的包装模块类型是 peft.tuners.lora.layer.Linear / Conv2d，
    # 判据用 base_layer + lora_A 两个属性比类名可靠
    hit: dict[str, int] = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "base_layer") and hasattr(mod, "lora_A"):
            short = ".".join(name.split(".")[3:])
            hit[short] = hit.get(short, 0) + 1
    n_hit = sum(hit.values())
    print(f"  命中模块   {n_hit} 个 → {hit}")
    if n_hit != len(TARGET_PRESETS[args.targets]) * base.config.n_layer:
        print(f"  ⚠ 期望 {len(TARGET_PRESETS[args.targets]) * base.config.n_layer} 个模块")

    # ---------- 3. 零样本基线 ----------
    base_bpc = None
    if not args.no_eval_base:
        print("\n[基线] 微调之前的 BPC（应与 16_gpt2_bpc.py 的 1.9288 一致）")
        with model.disable_adapter():
            base_bpc, _, _ = eval_bpc(
                model, val_ids, len(val_text), args.block, args.batch, device,
                args.eval_windows, tag="零样本 ",
            )

    # ---------- 4. 训练 ----------
    tokens_per_step = args.batch * args.block * args.grad_accum
    steps_per_epoch = max(1, len(train_ids) // tokens_per_step)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    print(f"\n[训练] {args.epochs} 轮 × {steps_per_epoch} 步 = {total_steps} 步"
          f"（每步 {tokens_per_step:,} token，lr={args.lr}，warmup={args.warmup}）")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    gen = torch.Generator().manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model.train()
    t0 = time.time()
    step = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        n_run = 0
        opt.zero_grad(set_to_none=True)
        for it in range(steps_per_epoch):
            if args.max_steps and step >= args.max_steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, args.warmup, args.lr)

            x, y = get_batch(train_ids, args.block, args.batch, device, gen)
            logits = model(input_ids=x).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            (loss / args.grad_accum).backward()

            if (it + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            running += float(loss.detach())
            n_run += 1
            step += 1
            if step % 20 == 0 or step == total_steps:
                el = time.time() - t0
                # 注意：轮首 running 会重置，所以要除以真实步数而不是写死 20，
                # 否则轮首第一次打印会出现 1.33 / 0.18 这种假数字
                print(f"    ep{epoch} step {step:4}/{total_steps}  "
                      f"train {running / max(n_run, 1):.4f}  "
                      f"lr {opt.param_groups[0]['lr']:.2e}  {el:.0f}s", flush=True)
                running = 0.0
                n_run = 0

        done = args.max_steps and step >= args.max_steps
        bpc, _, _ = eval_bpc(
            model, val_ids, len(val_text), args.block, args.batch, device,
            args.eval_windows, tag=f"ep{epoch} ",
        )
        history.append({"epoch": epoch, "step": step, "val_bpc": bpc})
        if done:
            break

    train_secs = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    # ---------- 5. 收尾 ----------
    final_bpc = history[-1]["val_bpc"]
    print("\n" + "=" * 70)
    print("结果")
    print("=" * 70)
    print(f"  可训练参数     {trainable:,}  ({trainable / total_params:.4%})")
    print(f"  训练步数       {step}   {train_secs:.0f} s ({train_secs / max(step, 1):.3f} s/步)")
    print(f"  显存峰值       {peak:.2f} GB")
    if base_bpc is not None:
        print(f"  零样本 BPC     {base_bpc:.4f}")
        print(f"  微调后 BPC     {final_bpc:.4f}   ({final_bpc - base_bpc:+.4f})")

    print("\n  2x2 归因表（最后一行是本脚本填的）")
    print(f"  {'':14}{'无预训练':>12}{'有预训练':>14}")
    print(f"  {'无目标域训练':12}{'3.58 (bigram)':>12}{'1.929 (GPT-2)':>14}")
    cell = f"{final_bpc:.3f} (LoRA)" if base_bpc is not None else "?"
    print(f"  {'有目标域训练':12}{'2.492 (mini)':>12}{cell:>14}")

    # ---------- 6. 存档 ----------
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    adapter_dir = CKPT_DIR / f"gpt2_lora_{tag}"
    model.save_pretrained(adapter_dir)
    print(f"\n  adapter → {adapter_dir}")

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not CSV_PATH.exists()
    with CSV_PATH.open("a", encoding="utf-8") as f:
        if new_file:
            f.write("tag,r,alpha,targets,epochs,lr,trainable,total,val_bpc,base_bpc,peak_gb,secs\n")
        f.write(
            f"{tag},{args.lora_r},{lora_alpha},{args.targets},{args.epochs},{args.lr},"
            f"{trainable},{total_params},{final_bpc:.4f},"
            f"{'' if base_bpc is None else f'{base_bpc:.4f}'},"
            f"{peak:.2f},{train_secs:.1f}\n"
        )
    print(f"  台账 → {CSV_PATH}")

    if args.gen:
        print("\n" + "=" * 70 + "\n微调后的生成（prompt = 'ROMEO:\\n'）\n" + "=" * 70)
        prompt = "ROMEO:\n"
        enc = tok(prompt, return_tensors="pt").to(device)
        model.eval()
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=80, do_sample=True, temperature=0.8,
                top_k=40, pad_token_id=tok.eos_token_id,
            )
        print(tok.decode(out[0], skip_special_tokens=True))

    return 0


if __name__ == "__main__":
    sys.exit(main())
