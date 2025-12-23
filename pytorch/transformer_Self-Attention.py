import torch
import torch.nn.functional as F

torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 手动定义3个词的向量（维度 d_model = 4）
# 实际中这些来自 nn.Embedding 或预训练模型
x = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],   # "The"
    [0.0, 1.0, 0.0, 0.0],   # "animal"
    [0.0, 0.0, 1.0, 0.0]    # "it"
])  # shape: [seq_len=3, d_model=4]

d_model = 4
d_k = d_model  # 简化：不降维

# 随机但固定的权重（为了可复现，也可以设为单位矩阵）
torch.manual_seed(42)
W_q = torch.randn(d_model, d_k)
W_k = torch.randn(d_model, d_k)
W_v = torch.randn(d_model, d_k)

# 计算 Q, K, V
Q = x @ W_q   # [3, 4]
K = x @ W_k   # [3, 4]
V = x @ W_v   # [3, 4]

print("Q:\n", Q)
print("K:\n", K)
print("V:\n", V)

# 计算 QK^T
scores = Q @ K.T  # [3, 3] —— 每个词对其他词的原始分数

# 缩放（除以 sqrt(d_k)）
scores = scores / (d_k ** 0.5)

# Softmax 得到注意力权重
attn_weights = F.softmax(scores, dim=-1)  # 对每一行做 softmax

print("\n注意力权重 (attn_weights):")
print(attn_weights)

# 输出 = attn_weights @ V
output = attn_weights @ V  # [3, 4]

print("\n输出表示 (output):")
print(output)

# 特别看 "it" 的新表示（第3行）
print("\n'it' 的新表示（融合了上下文）:")
print(output[2])
