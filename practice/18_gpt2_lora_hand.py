"""台阶 5：手写 LoRA 注入 GPT-2 的 Conv1D，和 peft 对账。

台阶 4 用 peft 拿到了 1.5962 BPC。但这个数字有一个隐患：
**你看不出来它算得对不对**。peft 内部做了什么（矩阵方向、缩放、dropout 位置）
全在库里，如果它和论文的式(3)有偏差，你也不会知道。

所以台阶 5 要把同一个东西再写一遍，然后**逼两条路径给出同一个答案**。
这是唯一能证明「台阶 4 的数字可信」的方法，也是唯一能看见「库替你做了什么」的方法。

三个验收，一个比一个强
    V1 参数个数一样      弱：只能证明数量对，不证明值对
    V2 前向输出逐位相等   强：把 peft 训好的权重搬进手写版，logits 必须一模一样
    V3 merge 前后一致     证明 merge/unmerge 的矩阵方向对

**核心差异只有一处**

nn.Linear 和 transformers 的 Conv1D 存的权重方向相反：

    nn.Linear : weight (out, in)，forward 里  y = x @ W.T
    Conv1D    : weight (in, out)，forward 里  y = x @ W

设 LoRA 的 A: (r, in)、B: (out, r)：

    nn.Linear 的增量    delta = x @ A.T @ B.T       形状 (…, in)@(in,r)@(r,out) = (…, out) ✓
    Conv1D   的增量    delta = x @ A.T @ B.T       形状 (…, nx)@(nx,r)@(r,out) = (…, nf)  ✓

**同一个表达式两边都对**——因为 A、B 的形状是按「数据流」定义的，与底层矩阵怎么存无关。

真正的差别在 merge（把增量并进基座权重）：

    nn.Linear  W += scaling * (B @ A)         W 是 (out, in)，B@A 是 (out,r)@(r,in) = (out,in) ✓
    Conv1D     W += scaling * (B @ A).T       W 是 (in, out)，必须转置                  ✓

peft 把这件事叫 `fan_in_fan_out`。台阶 4 跑的时候它自动打了一条警告：

    UserWarning: fan_in_fan_out is set to False but the target module is `Conv1D`.
    Setting fan_in_fan_out to True.

**那条警告就是这行转置。** 本脚本把它显式写出来，并且用 V3 验证。

用法：
    python practice/18_gpt2_lora_hand.py --check-only   # 只做三个验收，几秒
    python practice/18_gpt2_lora_hand.py                # 验收 + 训练，与踏台 4 对照
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import pathlib
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.pytorch_utils import Conv1D

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / ".cache" / "hf" / "gpt2"
CORPUS = ROOT / "data" / "shakespeare.txt"

TARGET_PRESETS = {
    "attn": ["c_attn", "attn.c_proj"],
    "all": ["c_attn", "attn.c_proj", "c_fc", "mlp.c_proj"],
}


def _load(name: str, filename: str):
    """按路径加载兄弟脚本。sys.modules[name] 必须在 exec_module 之前登记，
    否则模块里的类拿不到 __module__ 归属，多进程 DataLoader 会 PicklingError。
    （这个坑在 14_lora_vit.py 上踩过一次。）"""
    spec = importlib.util.spec_from_file_location(name, pathlib.Path(__file__).parent / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base17 = _load("base17", "17_gpt2_lora.py")


# ---------------------------------------------------------------- 手写 LoRA


class LoRAConv1D(nn.Module):
    """包住 transformers 的 Conv1D，加低秩增量。

    和 lora.py 的 LoRALinear 唯一的区别在 merge 里多一个 .T。
    """

    def __init__(self, base: Conv1D, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / r
        self.merged = False

        nx, nf = base.nx, base.nf          # (in, out)，注意顺序
        self.lora_A = nn.Parameter(torch.empty(r, nx))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(nf, r))
        self.lora_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)                          # (…, nf)
        if self.merged:
            return out
        # x:(…,nx) @ A.T:(nx,r) @ B.T:(r,nf) → (…,nf)
        delta = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return out + delta * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        if self.merged:
            return
        # base.weight 是 (nx, nf)，而 (B@A) 是 (nf, nx) —— 必须转置
        self.base.weight += self.scaling * (self.lora_B @ self.lora_A).T
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        self.base.weight -= self.scaling * (self.lora_B @ self.lora_A).T
        self.merged = False


def inject_lora_gpt2(model: nn.Module, targets: list[str], r: int, alpha: int, dropout: float) -> int:
    """把模型里所有匹配 targets 的 Conv1D 换成 LoRAConv1D，返回替换个数。"""
    count = 0

    def rec(parent: nn.Module, prefix: str) -> None:
        nonlocal count
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, Conv1D) and any(full.endswith("." + t) for t in targets):
                setattr(parent, name, LoRAConv1D(child, r, alpha, dropout))
                count += 1
            else:
                # 替换后不要再往下递归，否则会钻进新包的 LoRAConv1D
                rec(child, full)

    rec(model, "")
    return count


def freeze_all_but_lora(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for mod in model.modules():
        if isinstance(mod, LoRAConv1D):
            mod.lora_A.requires_grad = True
            mod.lora_B.requires_grad = True


def lora_params(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------- 权重搬运


def copy_peft_into_hand(hand: nn.Module, peft_model: nn.Module) -> int:
    """把 peft 的 lora_A/lora_B 原样搬进手写版。

    peft 的参数名：base_model.model.transformer.h.0.attn.c_attn.lora_A.default.weight
                    ^^^^^^^^^^^^^^^^^^ peft 自己套的两层外壳，要剥掉
    手写的参数名：transformer.h.0.attn.c_attn.lora_A
    """
    hmap = dict(hand.named_parameters())
    n = 0
    prefix = "base_model.model."
    with torch.no_grad():
        for pname, ptensor in peft_model.named_parameters():
            if ".lora_A." not in pname and ".lora_B." not in pname:
                continue
            if not pname.startswith(prefix):
                raise KeyError(f"peft 参数名没有预期前缀：{pname}")
            pname = pname[len(prefix):]
            stem, rest = pname.split(".lora_")
            which = rest.split(".")[0]              # 'A' 或 'B'
            hname = f"{stem}.lora_{which}"
            if hname not in hmap:
                raise KeyError(f"手写版找不到 {hname}（peft 名 {pname}）")
            hmap[hname].copy_(ptensor)
            n += 1
    return n


# ---------------------------------------------------------------- 三个验收


def v1_params(hand, peft_model) -> bool:
    n_hand = sum(p.numel() for p in lora_params(hand))
    n_peft = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    ok = n_hand == n_peft
    print(f"  V1 参数个数   手写 {n_hand:,}   peft {n_peft:,}   {'✅' if ok else '❌'}")
    return ok


def v2_forward(hand, peft_model, tokens: torch.Tensor, device: str) -> bool:
    """同一份权重、同一个输入，logits 必须逐位相等。"""
    hand.eval()
    peft_model.eval()
    with torch.no_grad():
        a = hand(tokens.to(device)).logits
        b = peft_model(tokens.to(device)).logits
    diff = float((a - b).abs().max())
    # 两边算的是同一个数学式子但浮点累加顺序不同，允许 1e-6 量级的差
    ok = diff < 1e-6
    scale = float(b.abs().mean())
    print(f"  V2 前向输出   最大绝对差 {diff:.3e}（logits 平均绝对值 {scale:.3f}）  "
          f"{'✅' if ok else '❌'}")
    return ok


def v3_merge(hand, tokens: torch.Tensor, device: str) -> bool:
    """merge 之后，增量已经并进 base.weight，前向结果必须不变。"""
    hand.eval()
    before_weight = None
    target = None
    for mod in hand.modules():
        if isinstance(mod, LoRAConv1D):
            target = mod
            break
    if target is None:
        print("  V3 merge      ❌ 没有找到 LoRAConv1D")
        return False

    with torch.no_grad():
        before_weight = target.base.weight.detach().clone()
        out_unmerged = hand(tokens.to(device)).logits

    # 全模型的 merge/unmerge（用子集模块验证权重形状也对）
    n = 0
    for mod in hand.modules():
        if isinstance(mod, LoRAConv1D):
            mod.merge()
            n += 1
    with torch.no_grad():
        out_merged = hand(tokens.to(device)).logits
        weight_delta = float((target.base.weight - before_weight).abs().max())

    for mod in hand.modules():
        if isinstance(mod, LoRAConv1D):
            mod.unmerge()
    with torch.no_grad():
        out_restored = hand(tokens.to(device)).logits

    d_merge = float((out_merged - out_unmerged).abs().max())
    d_restore = float((out_restored - out_unmerged).abs().max())
    ok = d_merge < 1e-4 and d_restore < 1e-4
    print(f"  V3 merge      {n} 个模块  weight 变化 {weight_delta:.3e}  "
          f"前向差 {d_merge:.3e}  unmerge 还原差 {d_restore:.3e}  {'✅' if ok else '❌'}")
    return ok


# ---------------------------------------------------------------- 训练


def train_hand(hand, train_ids, val_ids, n_val_chars, args, device):
    """和 17_gpt2_lora.py 完全相同的训练协议，方便直接对比 BPC。"""
    params = lora_params(hand)
    tokens_per_step = args.batch * args.block * args.grad_accum
    steps_per_epoch = max(1, len(train_ids) // tokens_per_step)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    gen = torch.Generator().manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    hand.train()
    t0 = time.time()
    step = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        running, n_run = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for it in range(steps_per_epoch):
            if args.max_steps and step >= args.max_steps:
                break
            for g in opt.param_groups:
                g["lr"] = base17.lr_at(step, total_steps, args.warmup, args.lr)

            x, y = base17.get_batch(train_ids, args.block, args.batch, device, gen)
            logits = hand(input_ids=x).logits
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
                print(f"    ep{epoch} step {step:4}/{total_steps}  "
                      f"train {running / max(n_run, 1):.4f}  "
                      f"lr {opt.param_groups[0]['lr']:.2e}  {time.time() - t0:.0f}s", flush=True)
                running, n_run = 0.0, 0

        bpc, _, _ = base17.eval_bpc(
            hand, val_ids, n_val_chars, args.block, args.batch, device,
            args.eval_windows, tag=f"ep{epoch} ",
        )
        history.append(bpc)
        if args.max_steps and step >= args.max_steps:
            break
        hand.train()

    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    return history[-1], step, time.time() - t0, peak


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=0, help="0 = 2*r")
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--targets", choices=sorted(TARGET_PRESETS), default="attn")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--eval-windows", type=int, default=0)
    ap.add_argument("--check-only", action="store_true", help="只做 V1/V2/V3，不训练")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alpha = args.lora_alpha or 2 * args.lora_r
    targets = TARGET_PRESETS[args.targets]
    tag = args.tag or f"hand_r{args.lora_r}_{args.targets}"

    print("=" * 70)
    print(f"台阶 5：手写 LoRA vs peft   tag={tag}")
    print("=" * 70)

    # ---------- 两份引擎，同一份基座 ----------
    base_hand = GPT2LMHeadModel.from_pretrained(MODEL_DIR, local_files_only=True)
    base_peft = GPT2LMHeadModel.from_pretrained(MODEL_DIR, local_files_only=True)

    n_inj = inject_lora_gpt2(base_hand, targets, args.lora_r, alpha, args.lora_dropout)
    freeze_all_but_lora(base_hand)
    base_hand.to(device)

    peft_model = get_peft_model(
        base_peft,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=alpha,
            lora_dropout=args.lora_dropout,
            target_modules=targets,
            bias="none",
        ),
    ).to(device)

    print(f"  手写注入 {n_inj} 个 Conv1D   r={args.lora_r}  alpha={alpha}  "
          f"dropout={args.lora_dropout}")

    # ---------- 三个验收 ----------
    print("\n[验收]")
    tok = GPT2TokenizerFast.from_pretrained(MODEL_DIR, local_files_only=True)
    sample = torch.tensor([tok("ROMEO:\nI take thee")["input_ids"]], dtype=torch.long)

    ok1 = v1_params(base_hand, peft_model)

    # peft 把 B 初始化成 0，此时 B=0 让两边都退化成原模型，直接比没有信息量。
    # 所以先把 peft 的 A、B 填成非零随机数，让两条路径都真正在工作状态下比较。
    torch.manual_seed(args.seed + 1)
    with torch.no_grad():
        for pname, p in peft_model.named_parameters():
            if ".lora_A." in pname or ".lora_B." in pname:
                p.normal_(0, 0.02)
    n_copy = copy_peft_into_hand(base_hand, peft_model)
    print(f"  权重搬运   把 peft 的 {n_copy} 个张量拷进手写版之后才比较")

    ok2 = v2_forward(base_hand, peft_model, sample, device)
    ok3 = v3_merge(base_hand, sample, device)

    if not (ok1 and ok2 and ok3):
        print("\n  ❗ 验收没过，后面的训练结果没有意义，先修实现")
        return 1

    if args.check_only:
        print("\n  --check-only：验收通过，跳过训练")
        return 0

    # ---------- 训练（和台阶 4 同协议） ----------
    text = CORPUS.read_text(encoding="utf-8")
    val_text = text[base17.MINI_GPT_VAL_START:]
    train_ids = torch.tensor(tok(text[: base17.MINI_GPT_VAL_START])["input_ids"], dtype=torch.long)
    val_ids = torch.tensor(tok(val_text)["input_ids"], dtype=torch.long)
    print(f"\n[训练] 与 17_gpt2_lora.py 同协议（同数据、同 lr、同 warmup、同 seed）")

    bpc, steps, secs, peak = train_hand(
        base_hand, train_ids, val_ids, len(val_text), args, device
    )

    print("\n" + "=" * 70)
    print("结果")
    print("=" * 70)
    print(f"  手写 LoRA 可训练  {sum(p.numel() for p in lora_params(base_hand)):,}")
    print(f"  训练              {steps} 步 / {secs:.0f} s")
    print(f"  显存峰值          {peak:.2f} GB")
    print(f"  手写 BPC          {bpc:.4f}")
    print(f"  peft BPC          1.5962   （台阶 4，r=8 attn，同一协议）")
    print(f"  差值              {bpc - 1.5962:+.4f}")
    print(f"\n  说明：差值不严格为 0 是正常的——dropout 的随机数消耗顺序不同，"
          f"且浮点累加顺序不同。")
    print(f"  真正严格的是 V2：同一组权重下前向输出逐位相等。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
