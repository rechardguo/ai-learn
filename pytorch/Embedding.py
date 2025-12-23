import torch
import torch.nn as nn

vocab_size = 10000        # 词表大小
d_model = 768                 # embedding 维度

embedding_layer = nn.Embedding(vocab_size, d_model)

print("Embedding 层参数:")
for name, param in embedding_layer.named_parameters():
    print(f"{name}: {param.data.shape}")

# 输入：一批 token ID（每个在 0 ~ 9999 之间）
input_ids = torch.tensor([[123, 456, 789], [101, 205, 309]])  # shape: [2, 3]

# 输出：每个 token 被替换为其 embedding 向量
embeddings = embedding_layer(input_ids)  # shape: [2, 3, 768]

print("输入 token IDs:")
print(input_ids)
print("对应的 Embeddings:")
print(embeddings)
print("Embeddings 形状:", embeddings.shape)  # 应该是 [2, 3, 5]


