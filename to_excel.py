import numpy as np
import pandas as pd
# 将npy文件转化为excel文件，以便导入origin作图
# 加载.npy文件
data_ideal = np.load('./sims_ideal.npy')
data = np.load('./sims.npy')
data_d_i = data_ideal[:100, :500]
data_d = data[:100, :500]

# 将NumPy数组转换为DataFrame
df1 = pd.DataFrame(data_d_i)
df = pd.DataFrame(data_d)
# 保存DataFrame为Excel文件
df1.to_excel('output_reduction_ideal.xlsx', index=False)
df.to_excel('output_reduction.xlsx', index=False)
