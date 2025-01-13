import matplotlib.pyplot as plt
import numpy as np

n = 10

# 创建一个 n x n 的二维numpy数组
data_1d_0_1 = np.linspace(start=0, stop=1, num=n ** 2)
a = np.reshape(data_1d_0_1, (n, n))
print(a)
plt.figure(figsize=(12, 4.5))

# 展示使用viridis颜色映射的图像，同样没有进行颜色的混合
plt.subplot(121)
plt.imshow(a, cmap='viridis', interpolation='nearest')
plt.yticks([])
plt.xticks(range(n))
# Viridis映射，无混合
plt.title('Viridis color map, no blending', y=1.02, fontsize=12)
plt.colorbar()

# 展示使用viridis颜色映射的图像，并且使用了双立方插值方法进行颜色混合
plt.subplot(122)
plt.imshow(a, cmap='viridis', interpolation='bicubic')
plt.yticks([])
plt.xticks(range(n))
# Viridis 映射，双立方混合
plt.title('Viridis color map, bicubic blending', y=1.02, fontsize=12)
plt.colorbar()
plt.show()
