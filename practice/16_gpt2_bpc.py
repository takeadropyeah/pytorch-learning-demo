"""台阶 2+3：把 GPT-2 的零样本能力换算成 BPC，和 mini GPT 放在同一张表里。

**为什么必须换单位**

    mini GPT：char-level，词表 65      → loss 1.7271 的单位是 nats/**字符**
    GPT-2：   BPE，词表 50,257         → loss 的单位是 nats/**token**
    一个 GPT-2 token 平均 3.307 个字符（台阶 1 实测）

直接比这两个数会得出完全相反的结论。统一到「压缩这段文字需要多少比特/字符」：

    BPC = 总 nats / ln2 / 原文字符数

这个定义是对的，因为它直接回答「这张纸上每个字符值多少比特」，
和用 char-level 还是 BPE 无关。

**划分必须和 mini GPT 一致**，否则不是同一份数据：
    mini GPT 的划分是前 1,003,854 字符训练、后 111,540 字符验证（90/10 按字符切）。
    这里验证集用**完全相同的后 111,540 字符**。

用法：
    python practice/16_gpt2_bpc.py                    # 全量验证集
    python practice/16_gpt2_bpc.py --max-windows 40   # 快速试
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / ".cache" / "hf" / "gpt2"
CORPUS = ROOT / "data" / "shakespeare.txt"

# 和 11_mini_gpt.py 完全相同的划分（按字符切 90/10）
MINI_GPT_VAL_START = 1_003_854
LN2 = math.log(2.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=512, help="窗口长度（≤1024）")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-windows", type=int, default=0, help="只跑前 N 个窗口，0=全跑")
    ap.add_argument("--gen", action="store_true", help="额外做一次零样本生成")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 68)
    print("台阶 3：GPT-2 零样本 BPC")
    print("=" * 68)

    tok = GPT2TokenizerFast.from_pretrained(MODEL_DIR, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR, local_files_only=True).to(device).eval()
    print(f"  {sum(p.numel() for p in model.parameters()):,} 参数   device={device}")

    text = CORPUS.read_text(encoding="utf-8")
    val_text = text[MINI_GPT_VAL_START:]
    print(f"  语料 {len(text):,} 字符；验证集 = 后 {len(val_text):,} 字符（与 mini GPT 同一段）")

    ids = tok(val_text)["input_ids"]
    print(f"  验证集 {len(ids):,} tokens")

    L = min(args.block, model.config.n_positions)
    # 窗口不重叠，每个窗口预测 L 个 token（第 0 个没有前文，丢掉）
    starts = list(range(0, len(ids) - L - 1, L))
    if args.max_windows:
        starts = starts[: args.max_windows]
    print(f"  窗口 {len(starts)} 个 × {L} token/窗口 = 覆盖 {len(starts) * L:,} token"
          f"（{len(starts) * L / len(ids):.1%}）")
    print(f"  预计前向 {math.ceil(len(starts) / args.batch)} 批\n")

    total_nll = 0.0
    total_tokens = 0
    t0 = time.time()

    with torch.no_grad():
        for bi in range(0, len(starts), args.batch):
            chunk = starts[bi : bi + args.batch]
            xs = torch.tensor(
                [ids[s : s + L] for s in chunk], dtype=torch.long, device=device
            )
            ys = torch.tensor(
                [ids[s + 1 : s + L + 1] for s in chunk], dtype=torch.long, device=device
            )
            logits = model(xs).logits                      # (B, L, V)
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), ys.reshape(-1), reduction="sum"
            )
            total_nll += float(nll)
            total_tokens += ys.numel()

            done = min(bi + args.batch, len(starts))
            if (bi // args.batch) % 20 == 0 or done == len(starts):
                el = time.time() - t0
                cur = total_nll / max(total_tokens, 1)
                print(f"    {done:4}/{len(starts)} 窗口  {el:6.1f}s  "
                      f"累计 nats/token {cur:.4f}  BPC {cur / LN2 / (len(val_text) / len(ids)):.4f}",
                      flush=True)

    secs = time.time() - t0
    nats_per_token = total_nll / total_tokens
    chars_per_token = len(val_text) / len(ids)

    # BPC 的两种算法：按 token 数换算，和按字符总数换算。
    # 覆盖率不足 100% 时两者会差一点，这里两个都报，让你看见差在哪。
    bpc_by_token = nats_per_token / LN2 / chars_per_token
    covered_chars = total_tokens * chars_per_token
    bpc_total = total_nll / LN2 / covered_chars
    ppl = math.exp(nats_per_token)

    uniform = math.log(model.config.vocab_size)

    print(f"\n{'=' * 68}")
    print(f"结果（{secs:.1f} s，{total_tokens:,} 个预测位置）")
    print(f"{'=' * 68}")
    print(f"  平均 nats/token      {nats_per_token:.4f}")
    print(f"  困惑度 PPL           {ppl:.2f}")
    print(f"  字符/token           {chars_per_token:.3f}")
    print(f"  BPC                  {bpc_by_token:.4f}   （按 token 数换算）")
    print(f"  BPC                  {bpc_total:.4f}   （按字符总数换算，更严）")
    print(f"  均匀分布上界 ln(V)    {uniform:.4f} nats/token = {uniform / LN2 / chars_per_token:.4f} BPC")

    print(f"\n{'=' * 68}")
    print("同一张表：GPT-2 零样本 vs mini GPT 从零训练")
    print(f"{'=' * 68}")
    print(f"  {'模型':22}{'参数':>14}{'语料':>10}{'BPC':>9}")
    print(f"  {'GPT-2 small 零样本':20}{124_439_808:>14,}{'1.1 MB':>10}{bpc_by_token:>9.3f}")
    print(f"  {'mini GPT 从零训练':20}{809_856:>14,}{'1.1 MB':>10}{2.492:>9.3f}")
    print(f"\n  GPT-2 好 {2.492 / bpc_by_token:.2f} 倍，代价是 {124_439_808 / 809_856:.0f} 倍参数。")
    print(f"  但 GPT-2 没见过这份文本（WebText 里莎士比亚占比很低），"
          f"它赢的是**预训练**而不是**这份数据**。")

    if args.gen:
        print(f"\n{'=' * 68}\n零样本生成（没在这个语料上训练过）\n{'=' * 68}")
        prompt = "ROMEO:\n"
        enc = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=80, do_sample=True, temperature=0.8,
                top_k=40, pad_token_id=tok.eos_token_id,
            )
        print(tok.decode(out[0], skip_special_tokens=True))

    return 0


if __name__ == "__main__":
    sys.exit(main())
