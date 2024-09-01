import numpy as np

# 生成理想条件下相似度矩阵sims_ideal.npy
full = np.full((100, 500), -1)
for i in range(1000):
    full[i, 5 * i:5 * i + 5] = 1
print()
np.save('./sims_ideal.npy', full)
