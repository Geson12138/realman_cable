import numpy as np

approaching_numPoints = 500 # 前进阶段的点数
trans_length = 0.05  # 平移距离
delta_steps = trans_length / approaching_numPoints # 前半段沿x轴平移的步长
print(delta_steps)