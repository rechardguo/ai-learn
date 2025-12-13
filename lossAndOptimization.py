# 理解损失函数和优化器

import torch
import torch.nn as nn
import torch.optim as optim

# 1. 定义模型、损失函数、优化器
model = nn.Linear(1, 1)                # 简单线性模型 y = wx + b
criterion = nn.MSELoss()               # 回归任务用 MSE
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 2. 准备数据
x = torch.tensor([[2.0]])              # 输入 x=2
y_true = torch.tensor([[5.0]])         # 真实 y=5（假设 y = 2x + 1）

# 梯度 Gradient
# 3. 训练一步
optimizer.zero_grad()                  # 🔑 清空上次梯度（重要！）

y_pred = model(x)                      # 前向传播 → 比如得到 y=3.0
loss = criterion(y_pred, y_true)       # 计算 loss = (5-3)^2 = 4

loss.backward()                        # 🔑 反向传播：计算 d_loss/d_w, d_loss/d_b
print("梯度:", model.weight.grad)      # 比如 tensor([[-4.]])

optimizer.step()                       # 🔑 更新参数：w = w - lr * grad
print("更新后权重:", model.weight)     # 比如从 1.0 → 1.04