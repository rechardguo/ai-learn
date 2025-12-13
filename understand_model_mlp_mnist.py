import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==============================
# 1. 定义 MLP 模型
# ==============================

# MNIST 图像是 28×28 像素 → 共 784 个数值
model = nn.Sequential(
    nn.Linear(28 * 28, 128),  # 输入784 → 隐藏层128
    nn.ReLU(),
    nn.Linear(128, 10)        # 输出10类（0~9）
)

# ==============================
# 2. 设置设备（自动使用 GPU 如果可用）
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# ==============================
# 3. 加载 MNIST 数据集
# ==============================
transform = transforms.ToTensor()  # 自动将 PIL 图像转为 [0,1] 的 Tensor

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ==============================
# 4. 定义损失函数和优化器
# ==============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==============================
# 5. 训练函数
# ==============================
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28).to(device)  # 拉平: [64, 1, 28, 28] → [64, 784]
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# ==============================
# 6. 测试函数
# ==============================
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(-1, 28 * 28).to(device)
            target = target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

# ==============================
# 7. 开始训练！
# ==============================
for epoch in range(1, 6):  # 只训练 5 轮（足够达到 ~97% 准确率）
    train(epoch)
    test()