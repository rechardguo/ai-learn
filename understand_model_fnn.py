# 模型拟合一条真实直线，比如 y=3x+2
import torch

# torch.nn 是 PyTorch 中用于构建神经网络的核心模块，全称是 torch.neural_networks
# 包含
#   网络层（Layers）	nn.Linear, nn.Conv2d, nn.LSTM	构建模型的基本单元
#   激活函数（Activations）	nn.ReLU, nn.Sigmoid, nn.Tanh	引入非线性
#   损失函数（Losses）	nn.CrossEntropyLoss, nn.MSELoss	衡量预测误差
#   容器（Containers）	nn.Sequential, nn.ModuleList	组合多个层
#   基础类	nn.Module	所有自定义模型的基类
import torch.nn as nn
# 生成数据
x = torch.randn(100, 1)
y_true = 3 * x + 2 + 0.1 * torch.randn(100, 1)  # 加点噪声

# 训练
model = nn.Linear(1, 1)  # y=wx+b
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(300):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y_true)
    loss.backward()
    optimizer.step()

print("学到的 w:", model.weight.item())  # 应接近 3
print("学到的 b:", model.bias.item())    # 应接近 2

# 保存模型（推荐保存 state_dict）
torch.save(model.state_dict(), "linear_model.pth")
print("模型已保存！")


# 重新定义模型结构（必须一致！）
model = nn.Linear(1, 1)

# 加载权重
model.load_state_dict(torch.load("linear_model.pth"))
model.eval()  # 设置为评估模式（对 Linear 无影响，但好习惯）

# 预测新数据
new_x = torch.tensor([[4.0], [5.0], [6.0]])  # 批量预测
with torch.no_grad():
    predictions = model(new_x)

print("输入 x:", new_x.flatten())
print("预测 y:", predictions.flatten())
# 应输出 ≈ [14, 17, 20]