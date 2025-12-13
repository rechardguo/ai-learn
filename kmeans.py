# 导入所需库
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据（这里为了方便演示，使用随机生成的数据）
np.random.seed(42)
X = np.random.rand(100, 2) * 100  # 假设数据为100个样本，每个样本有两个属性

# 使用K-Means进行聚类分析
kmeans = KMeans(n_clusters=5, random_state=42)  # 假设我们要划分成5个簇
y_kmeans = kmeans.fit_predict(X)

# 可视化结果
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')  # 绘制点

# 绘制簇中心
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='x')
plt.title('Customer Segments based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()