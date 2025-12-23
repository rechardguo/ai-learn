import numpy as np
a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.add(a,b)
print(c)

W = np.array([[1, 2, 1],
              [3, 4,  3],[3, 4,3]])      # shape (3, 3)
x = np.array([5,6,7])      # shape (3, 1)

y = W @ x   # 或 np.dot(W, x)
print(y)

print(W.T)



# 创建 A ∈ R^(3×4)
A = np.random.randn(3, 4)   # 随机生成 3x4 矩阵
print("A shape:", A.shape)
print(A)

# 创建 B ∈ R^(4×2)
B = np.random.randn(4, 2)   # 随机生成 4x2 矩阵
print("B shape:", B.shape)
print(B)

# 计算 AB
C = A @ B   # 或者 np.dot(A, B)
print("AB shape:", C.shape)
print(C)



# 创建一个 2x3 的全 0 张量
a = torch.zeros(2, 3)
print(a)

# 创建一个 2x3 的全 1 张量
b = torch.ones(2, 3)
print(b)

# 创建一个 2x3 的随机数张量
c = torch.randn(2, 3)
print(c)

# 从 NumPy 数组创建张量
import numpy as np
numpy_array = np.array([[1, 2], [3, 4]])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(tensor_from_numpy)

# 在指定设备（CPU/GPU）上创建张量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = torch.randn(2, 3, device=device)
print(d)