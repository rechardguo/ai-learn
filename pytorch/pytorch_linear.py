# 理解 pytorch里的线性层 nn.Linear

import torch
import torch.nn as nn


layer = nn.Linear(1,1) # y=wx+b ,参数w 和 b
print("layer.weight:",layer.weight)
print("layer.bias:",layer.bias)

print("线性层参数:")
for name, param in layer.named_parameters():
    print(f"{name}: {param.data}")


layer2 = nn.Linear(2,1) # y=w1x1+w2x2+b ,参数w1,w2 和 b
print("layer2参数:")
for name, param in layer2.named_parameters():
    print(f"{name}: {param.data}")

layer3 = nn.Linear(2,3) # y=[y1,y2,y3], 每个yi都是y=wi1*x1+wi2*x2+bi ,参数wi1,wi2 和 bi  
print("layer3参数:")
for name, param in layer3.named_parameters():
    print(f"{name}: {param.data}")  

