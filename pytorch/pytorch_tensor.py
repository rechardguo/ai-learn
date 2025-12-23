import torch
# 理解张量，Shape，Dimensionality，DType

# 创建一个一维张量，包含四个元素，一维数组
x = torch.tensor([[-2.0, -0.5, 0.3, 1.8],[1,1,1,1]])
print("张量形状:", x.shape)
print("输入张量:", x)
print("张量维度:", x.dim())
print(x.shape[0])


print("-----------------------------------")
image = torch.randn(3,2) 
print("图像张量形状:", image.shape)
print("图像张量:", image)
print("图像维度",image.dim())


print("-----------------------------------")
torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = torch.rand(3,4,2)
print("随机张量:", tensor)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")