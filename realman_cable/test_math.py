import numpy as np

point_c = np.array([0,0,2]) # 蓝色的点 c
point_d = np.array([1,-1,2]) # 两个角点的中点 d
vector_cd = point_d - point_c # cd

# 用arctan2计算与x轴的夹角，范围[0, 360)
theta_deg = np.degrees(np.arctan2(vector_cd[1], vector_cd[0]))
if theta_deg < 0:
    theta_deg += 360

print("cd与x轴夹角:", theta_deg)