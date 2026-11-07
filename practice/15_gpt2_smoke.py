"""台阶 1：把 GPT-2 small 装进手里，不训练，只确认它长什么样。

这一步要钉死四件事，后面所有设计都建立在这四条上：

1. 参数量真的是 124,439,808（和 mini GPT 的 809,856 差 153.6 倍）
2. GPT-2 不用 nn.Linear，用的是 Conv1D，weight 形状是 (in, out)——
   和 `lora.py` 里按 (out, in) 写的 LoRALinear 正好相反
3. `lm_head` 与 `wte` 共享权重 → Shakespeare 词表没变，**没有新分类头要训**
4. 一个 BPE token 平均等于多少字符（台阶 2/3 换算 BPC 的分母）

用法：
    python practice/15_gpt2_smoke.py
"""

from __future__ import annotations

import pathlib
import sys
import time

import torch
import transformers
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / ".cache" / "hf" / "gpt2"
CORPUS = ROOT / "data" / "shakespeare.txt"

EXPECTED_PARAMS = 124_439_808


def main() -> int:
    print("=" * 68)
    print("台阶 1：GPT-2 small 冒烟（不训练）")
    print("=" * 68)
    print(f"  transformers {transformers.__version__}")
    print(f"  torch        {torch.__version__}   cuda={torch.cuda.is_available()}")

    if not MODEL_DIR.exists():
        print(f"\n模型目录不存在：{MODEL_DIR}")
        print("先跑：python practice/tools/fetch_gpt2.py")
        return 1

    # ---------- 1. 加载 ----------
    print(f"\n[1/4] 从本地加载 {MODEL_DIR}")
    t0 = time.time()
    tok = GPT2TokenizerFast.from_pretrained(MODEL_DIR, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR, local_files_only=True)
    model.eval()
    print(f"      用时 {time.time() - t0:.1f} s（全程离线，没有任何网络请求）")

    total = sum(p.numel() for p in model.parameters())
    cfg = model.config
    print(f"\n[2/4] 规模")
    print(f"      总参数        {total:,}")
    print(f"      预期          {EXPECTED_PARAMS:,}   {'✅ 一致' if total == EXPECTED_PARAMS else '❌ 不一致'}")
    print(f"      层数          {cfg.n_layer}")
    print(f"      n_embd        {cfg.n_embd}     n_head {cfg.n_head}     n_positions {cfg.n_positions}")
    print(f"      词表          {cfg.vocab_size:,}")
    print(f"      激活函数      {cfg.activation_function}")
    print(f"      层归一化 eps  {cfg.layer_norm_epsilon}")
    print(f"      权重绑定      wte == lm_head ? "
          f"{model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()}")

    # 按模块类型拆参数——看清 85% 的参数到底在哪
    embed = model.transformer.wte.weight.numel() + model.transformer.wpe.weight.numel()
    per_layer_attn = sum(p.numel() for p in model.transformer.h[0].attn.parameters())
    per_layer_mlp = sum(p.numel() for p in model.transformer.h[0].mlp.parameters())
    per_layer_ln = sum(p.numel() for p in model.transformer.h[0].ln_1.parameters()) \
                 + sum(p.numel() for p in model.transformer.h[0].ln_2.parameters())
    print(f"\n      嵌入(wte+wpe)   {embed:>12,}  ({embed / total:5.1%})"
          f"   ← wte {model.transformer.wte.weight.numel():,} + wpe {model.transformer.wpe.weight.numel():,}")
    print(f"      每层 attention  {per_layer_attn:>12,}")
    print(f"      每层 MLP        {per_layer_mlp:>12,}")
    print(f"      每层 LayerNorm  {per_layer_ln:>12,}")
    print(f"      12 层合计       {(per_layer_attn + per_layer_mlp + per_layer_ln) * cfg.n_layer:>12,}"
          f"  ({(per_layer_attn + per_layer_mlp + per_layer_ln) * cfg.n_layer / total:5.1%})")

    # ---------- 2. Conv1D 是什么 ----------
    print(f"\n[3/4] 第一层的四个线性映射（注意类型和形状）")
    blk = model.transformer.h[0]
    rows = [
        ("attn.c_attn", blk.attn.c_attn, "QKV 融合，3×768=2304"),
        ("attn.c_proj", blk.attn.c_proj, "多头输出合并"),
        ("mlp.c_fc", blk.mlp.c_fc, "FFN 升维 768→3072"),
        ("mlp.c_proj", blk.mlp.c_proj, "FFN 降维 3072→768"),
    ]
    for name, mod, note in rows:
        w = mod.weight
        print(f"      {name:14} {type(mod).__name__:8} weight {tuple(w.shape)!s:14}"
              f"  {w.numel():>9,}   {note}")

    n_conv1d = sum(1 for m in model.modules() if type(m).__name__ == "Conv1D")
    n_linear = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    print(f"\n      全模型 Conv1D {n_conv1d} 个，nn.Linear {n_linear} 个（lm_head 是 Linear）")
    print(f"      → 可以包 LoRA 的对象：{n_conv1d} 个 Conv1D（每层 4 个 × 12 层）")

    print(f"\n      对照 nn.Linear：weight 是 (out, in)，forward 里 x @ W.T")
    print(f"      Conv1D：        weight 是 (in, out)，forward 里 x @ W   ← 不做转置")
    print(f"      所以 lora_B @ lora_A 要长得像 (in, out)，和 lora.py 的写法相反")

    # ---------- 3. 分词 ----------
    print(f"\n[4/4] 分词：一个 token 等于多少字符")
    if not CORPUS.exists():
        print(f"      语料不存在：{CORPUS}")
        return 1
    text = CORPUS.read_text(encoding="utf-8")
    sample = text[:20000]
    ids = tok(sample)["input_ids"]
    cpt = len(sample) / len(ids)
    print(f"      语料总计      {len(text):,} 字符")
    print(f"      抽样          {len(sample):,} 字符 → {len(ids):,} tokens")
    print(f"      字符/token    {cpt:.3f}    ← BPC 换算的分母")
    print(f"      全语料约      {len(text) / cpt:,.0f} tokens")

    head = text[:120]
    print(f"\n      样例：{head!r}")
    print(f"      → {len(tok(head)['input_ids'])} tokens: {tok(head)['input_ids'][:20]}...")
    print(f"      → 解码回来：{tok.decode(tok(head)['input_ids'])[:60]!r}")

    # ---------- 4. 一次前向 ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    batch = torch.tensor([ids[:64]], device=device)
    print(f"\n一次前向（不训练）：输入 {tuple(batch.shape)} → ", end="", flush=True)
    t0 = time.time()
    with torch.no_grad():
        out = model(batch)
    dt = (time.time() - t0) * 1000
    print(f"logits {tuple(out.logits.shape)}   {dt:.0f} ms")

    nxt = out.logits[0, -1].argmax().item()
    print(f"      看前 64 个 token 后的下一个词预测："
          f"{tok.decode([nxt])!r}（只是 argmax，不代表合理）")
    if device == "cuda":
        print(f"      显存峰值 {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

    # ---------- 收尾：和 mini GPT 的对照 ----------
    print(f"\n{'=' * 68}")
    print("和 mini GPT 的对照")
    print(f"{'':16}{'GPT-2 small':>16}{'mini GPT':>14}{'倍数':>10}")
    print(f"  {'参数':14}{total:>16,}{809_856:>14,}{total / 809_856:>9.1f}×")
    print(f"  {'词表':14}{cfg.vocab_size:>16,}{65:>14,}{cfg.vocab_size / 65:>9.0f}×")
    print(f"  {'上下文':14}{cfg.n_positions:>16,}{64:>14,}{cfg.n_positions / 64:>9.0f}×")
    print(f"\n  单位不同：GPT-2 的 loss 是 nats/token，mini GPT 是 nats/字符。")
    print(f"  要比较必须先换成 BPC = nats ÷ ln2 ÷ {cpt:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
