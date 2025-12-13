# 目标
- 学习目标：理解核心原理
- 神经网络如何用 PyTorch 构建？
- Transformer 是怎么工作的？
- LLM（大语言模型）的基本结构是什么？
- 微调（Fine-tuning）和推理（Inference）流程是怎样的？


# 安装pytorch

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 什么是pytorch

PyTorch 是一个开源的机器学习库，主要用于进行计算机视觉（CV）、自然语言处理（NLP）、语音识别等领域的研究和开发。

PyTorch由 Facebook 的人工智能研究团队开发，并在机器学习和深度学习社区中广泛使用。

PyTorch 以其灵活性和易用性而闻名，特别适合于深度学习研究和开发

## 学习前提

- 编程基础：熟悉至少一种编程语言，尤其是 Python，因为 PyTorch 主要是用 Python 编写的。

- 数学基础：了解线性代数、概率论和统计学、微积分等基础数学知识，这些是理解和实现机器学习算法的基石。

- 机器学习基础：了解机器学习的基本概念，如监督学习、无监督学习、强化学习、模型评估指标（准确率、召回率、F1分数等）。

- 深度学习基础：熟悉神经网络的基本概念，包括前馈神经网络、卷积神经网络（CNN）、循环神经网络（RNN）、长短期记忆网络（LSTM）等。

- 计算机视觉和自然语言处理基础：如果你打算在这些领域应用 PyTorch，了解相关的背景知识会很有帮助。

- Linux/Unix 基础：虽然不是必需的，但了解 Linux/Unix 操作系统的基础知识可以帮助你更有效地使用命令行工具和脚本，特别是在数据预处理和模型训练中。

- 英语阅读能力：由于许多文档、教程和社区讨论都是用英语进行的，具备一定的英语阅读能力将有助于你更好地学习和解决问题


# 实现深度学习最基础的完整训练流程，从而真正理解神经网络如何工作
- MLP （多层感知机，Multilayer Perceptron）,MLP 是一种前馈神经网络,用于解决
- - 分类（Classification）	手写数字识别（MNIST）、垃圾邮件判断、图像类别预测
- - 回归（Regression）	房价预测、温度预测、股票趋势拟合

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
# 理解：前向传播、损失函数、反向传播、优化器
```

- MNIST

MNIST 是一个包含 7 万张手写数字（0~9）灰度图像的数据集，每张图都是 28×28 像素

- CNN
- RNN
- Transformer 


# 
