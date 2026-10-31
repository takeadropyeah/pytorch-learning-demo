r"""
demo_attention.py —— 从"加权平均"一步步长成 attention

不解释概念，只把每一步的**数字**打出来，让你看到它是怎么变过去的。

关键：第 1~4 步里完全没有神经网络，只有「点积」和「softmax」。
第 5 步才加上可学习的投影矩阵。**attention 的主体不是网络，是一套权重算法。**

运行：
    python .\tools\demo_attention.py
"""

import math
import torch

torch.set_printoptions(precision=3, sci_mode=False)

LINE = "=" * 78


def head(title):
    print("\n" + LINE)
    print(title)
    print(LINE)


# =========================================================================== #
# 0. 原材料：4 个 token，每个是一个 4 维向量
#    想象这是一句话里的 4 个词，每个词被编码成了 4 个数字
#    「狗」故意做成和「猫」很像 —— 后面要用它演示"相似度决定权重"
# =========================================================================== #
X = torch.tensor([
    [1.00, 0.00, 0.00, 0.00],   # token 0：猫
    [0.00, 1.00, 0.00, 0.00],   # token 1：追
    [0.90, 0.15, 0.00, 0.00],   # token 2：狗（和"猫"很像）
    [0.00, 0.00, 1.00, 0.00],   # token 3：球
])
NAMES = ["猫", "追", "狗", "球"]

head("0. 原材料：4 个 token，每个 4 维")
print("   token 名字   " + "  ".join(f"{n:>6s}" for n in NAMES))
for i, row in enumerate(X):
    print(f"   {i} {NAMES[i]:>4s}   " + "  ".join(f"{float(v):6.2f}" for v in row))
print("\n   注意：「狗」=[0.90, 0.15, 0, 0]，和「猫」=[1, 0, 0, 0] 几乎同方向。")


# =========================================================================== #
# 1. 你已经会的东西：加权平均
# =========================================================================== #
head("1. 最朴素的：手写一组权重，做加权平均")

w_manual = torch.tensor([0.1, 0.1, 0.7, 0.1])
print(f"   权重 w   = {[round(float(v), 2) for v in w_manual]}   和 = {float(w_manual.sum()):.3f}")
print(f"   输出 out = {[round(float(v), 3) for v in (w_manual @ X)]}")
print("\n   这一行就是 output。它 = 0.1×猫 + 0.1×追 + 0.7×狗 + 0.1×球。")
print("   问题只有一个：**这组权重是人手写的，模型学不到它。**")


# =========================================================================== #
# 2. 让权重由数据自己决定：看谁"像"谁
# =========================================================================== #
head("2. 权重从哪来？用点积算「像不像」")

i_query = 0
query = X[i_query]
print(f"   查询 q = token {i_query}（{NAMES[i_query]}）的向量 = {query.tolist()}")
print("\n   和每个 token 做点积（点积 = 方向一致的程度的度量）：")
scores = X @ query
order = torch.argsort(scores, descending=True)
for j in order.tolist():
    bar = "#" * max(1, int(round(float(scores[j]) * 20)))
    print(f"     q·{NAMES[j]:<2s}(token {j}) = {float(scores[j]):6.3f}  {bar}")
print(f"\n   最高的两个是「{NAMES[int(order[0])]}」和「{NAMES[int(order[1])]}」——")
print("   因为它们的向量和「猫」方向最接近。**点积就是相似度。**")


# =========================================================================== #
# 3. 分数要变成"权重"：softmax
# =========================================================================== #
head("3. 把分数变成权重：softmax（保证非负 + 和为 1）")

w = torch.softmax(scores, dim=0)
print(f"   原始分数     = {[round(float(s), 3) for s in scores]}")
print(f"   softmax 之后 = {[round(float(v), 3) for v in w]}   和 = {float(w.sum()):.3f}")
print("\n   softmax 干两件事：①取指数（负数变正数，大的变得更大）②除以和（归一化）。")
print(f"   ↑ 这组数就是 attention 里的「注意力权重」：{NAMES[i_query]} 把 "
      f"{float(w.max()) * 100:.0f}% 的注意力给了「{NAMES[int(w.argmax())]}」。")


# =========================================================================== #
# 4. 用这组权重加权平均 —— attention 主体已经完成
# =========================================================================== #
head("4. 用这组权重加权平均：attention 的主体完成了（还没用到神经网络）")

out_simple = w @ X
terms = " + ".join(f"{float(w[j]):.2f}×{NAMES[j]}" for j in range(4))
print(f"   out = {terms}")
print(f"       = {[round(float(v), 3) for v in out_simple]}")
print(f"\n   ↑ 输出里占了 {float(w[2]) * 100:.0f}% 的「狗」。")
print("     而「球」只贡献了 "
      f"{float(w[3]) * 100:.0f}%——因为它和「猫」不像。")
print("     **权重是算出来的，不是人写的。** 只要 X 能学，权重就跟着学好。")


# =========================================================================== #
# 5. 但还不够：需要学「怎么问」「怎么被匹配」「给出什么」
# =========================================================================== #
head("5. 加三个可学习投影：Q = X@Wq，K = X@Wk，V = X@Wv")

d = 4
torch.manual_seed(3)
Wq = torch.eye(d) + 0.7 * torch.randn(d, d)
Wk = torch.eye(d) + 0.7 * torch.randn(d, d)
Wv = torch.eye(d) + 0.7 * torch.randn(d, d)

Q, K, V = X @ Wq, X @ Wk, X @ Wv
print("   三个角色必须分开：")
print("     Q (Query) —— 我在问什么          ← 「猫」想找什么")
print("     K (Key)   —— 我能被什么问题匹配上 ← 「狗」身上挂的标签")
print("     V (Value) —— 如果我被选中，给出什么内容 ← 「狗」想贡献的信息")
print("\n   若 Q=K=V=X，问法/标签/内容被同一组数绑死，模型学不动。")

scores2 = Q @ K.T / math.sqrt(d)
w2 = torch.softmax(scores2, dim=-1)
out2 = w2 @ V

print("\n   注意力权重矩阵（每行 = 一个 token 对全部 4 个 token 的关注度）：")
print("           " + "  ".join(f"{n:>6s}" for n in NAMES) + "     行和")
for i in range(4):
    row = "  ".join(f"{float(v):6.3f}" for v in w2[i])
    print(f"   {NAMES[i]:>4s}    {row}   {float(w2[i].sum()):.3f}")

print("\n   输出 out（4×4，和输入同形状）：")
print(out2)


# =========================================================================== #
# 6. 结果拆开看：out 的每一行 = V 的加权平均
# =========================================================================== #
head("6. 验证：输出的每一行，真的就是 V 的加权平均")

i = 2
manual = sum(float(w2[i, j]) * V[j] for j in range(4))
print(f"   以 token {i}（{NAMES[i]}）为例：")
print(f"     用权重手算 = {[round(float(v), 3) for v in manual]}")
print(f"     矩阵算出来 = {[round(float(v), 3) for v in out2[i]]}")
print(f"     是否一致   = {torch.allclose(manual, out2[i], atol=1e-6)}")
print("\n   ↑ 这一行验证了整件事：")
print("     **attention = 用算出来的权重，对 V 做加权平均。**")


# =========================================================================== #
# 7. 形状约定（必须背下来）
# =========================================================================== #
head("7. 形状约定")

B, L, dk = 2, 5, 8
q = torch.randn(B, L, dk)
k = torch.randn(B, L, dk)
v = torch.randn(B, L, dk)
s = q @ k.transpose(-2, -1) / math.sqrt(dk)
a = torch.softmax(s, dim=-1)

print(f"   q, k, v        {tuple(q.shape)}      (B=批, L=序列长度, d=每维)")
print(f"   q @ k.transpose(-2,-1)   {tuple(s.shape)}      ← 注意力分数，鼓出 L×L")
print(f"   softmax(dim=-1)          {tuple(a.shape)}      ← 权重，每行和=1")
print(f"   再 @ v                   {tuple((a @ v).shape)}      ← 回到 (B,L,d)")
print("\n   口诀：**进去是 (B,L,d)，出来还是 (B,L,d)，中间鼓一个 (B,L,L)。**")
print("   那个 L×L 就是「每个位置对每个位置的关注度」，对角线是「看自己」。")
print("\n   为什么 softmax 在最后一维？最后一维 = 候选集合的方向，")
print("   沿它归一化 = 「第 i 个位置把注意力按比例分给所有位置」。")


# =========================================================================== #
# 8. 最后：为什么必须除以 sqrt(d)
# =========================================================================== #
head("8. 为什么除以 sqrt(d)")

torch.manual_seed(0)
for dk in (8, 64, 512):
    qq = torch.randn(1, dk)
    kk = torch.randn(1000, dk)
    ss = kk @ qq.squeeze()
    p_raw = torch.softmax(ss, 0)
    p_scaled = torch.softmax(ss / math.sqrt(dk), 0)
    print(f"   d={dk:4d}  q·k 的 std={float(ss.std()):6.2f} (理论√d={math.sqrt(dk):5.2f})"
          f"   不缩放最大权重={float(p_raw.max()):.3f}"
          f"   除以√d后={float(p_scaled.max()):.3f}")

print("\n   ↑ 维度越高，点积的绝对值越大 → softmax 越接近 one-hot")
print("     → 梯度 ≈ p(1-p) → 接近 0 → 大部分位置学不动。")
print("     除以 √d 把分布拉回正常范围。**这是数值稳定性，不是魔法。**")
