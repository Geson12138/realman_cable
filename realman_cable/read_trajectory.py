import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============================末端位姿变化曲线==============================
# # 将数据转换为 NumPy 数组
# with open('./saved_data/end_pose_data_desired.json', 'r') as f:
#     end_pose_data_desired = json.load(f)
# end_pose_data_desired = np.array([np.array(item) for item in end_pose_data_desired])  # 形状 (n_samples, 6)
# print(f"num of end_pose_desired: {end_pose_data_desired.shape[0]}")
# with open('./saved_data/end_pose_data_real.json', 'r') as f:
#     end_pose_data_real = json.load(f)
# # 将数据转换为 NumPy 数组
# end_pose_data_real = np.array([np.array(item) for item in end_pose_data_real])  # 形状 (n_samples, 6)
# print(f"num of end_pose_real: {end_pose_data_real.shape[0]}")
# # 创建一个窗口
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(end_pose_data_desired[:, 0], end_pose_data_desired[:, 1], end_pose_data_desired[:, 2], label='Desired End Effector Trajectory', color='b')
# ax.plot(end_pose_data_real[:, 0], end_pose_data_real[:, 1], end_pose_data_real[:, 2], label='Real End Effector Trajectory', color='r')
# ax.set_xlabel('X (m)')
# ax.set_ylabel('Y (m)')
# ax.set_zlabel('Z (m)')
# ax.set_title('End Effector 3D Trajectory')
# ax.legend()
# plt.tight_layout()
# plt.savefig('./saved_data/end_pose_3d.png')
# ==============================末端位姿变化曲线==============================


# ==============================期望关节变化曲线==============================
# with open('./saved_data/trajectory_data.json', 'r') as f:
#     trajectory_data_desired = json.load(f)
# # 将数据转换为 NumPy 数组
# trajectory_array_desired = np.array([np.array(item) for item in trajectory_data_desired])  # 形状 (n_samples, 6)
# print(f"num of joint_desired: {trajectory_array_desired.shape[0]}")
# # 创建一个窗口，包含6个子图
# fig, axes = plt.subplots(7, 1, figsize=(12, 18))
# joint_labels = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Joint 7']

# # 获取样本数量
# n_samples = trajectory_array_desired.shape[0]

# # 绘制每个关节的变化曲线
# for i in range(7):
#     axes[i].plot(np.arange(n_samples), trajectory_array_desired[:, i], label=f'Desired {joint_labels[i]}' ,color='blue')
#     axes[i].set_ylabel('Angle (rad)')
#     axes[i].legend()
#     axes[i].grid(True)
# axes[5].set_xlabel('Sample')
# plt.tight_layout()
# plt.savefig('./saved_data/trajectory_array_desired.png')
# ==============================期望关节变化曲线==============================

# ==============================实际关节变化曲线==============================
# with open('./saved_data/trajectory_data_real.json', 'r') as f:
#     trajectory_data_real = json.load(f)
# # 将数据转换为 NumPy 数组
# trajectory_array_real = np.array([np.array(item) for item in trajectory_data_real])  # 形状 (n_samples, 6)
# # 获取样本数量
# print(f"num of joint_real: {trajectory_array_real.shape[0]}")
# n_samples = trajectory_array_real.shape[0]

# # 创建一个窗口，包含6个子图
# fig, axes = plt.subplots(7, 1, figsize=(12, 18))
# joint_labels = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Joint 7']

# # 绘制每个关节的变化曲线
# for i in range(7):
#     axes[i].plot(np.arange(n_samples), trajectory_array_real[:, i], label=f'Real {joint_labels[i]}', color='red')
#     # axes[i].plot(np.arange(n_samples), trajectory_array[:, i], label=f'Desired {joint_labels[i]}' ,color='blue')
#     axes[i].set_ylabel('Angle (rad)')
#     axes[i].legend()
#     axes[i].grid(True)
# axes[5].set_xlabel('Sample')
# plt.tight_layout()
# plt.savefig('./saved_data/trajectory_array_real.png')
# ==============================实际关节变化曲线==============================


# ==============================实际关节速度曲线==============================
# with open('./saved_data/trajectory_velocity_real.json', 'r') as f:
#     trajectory_velocity_real = json.load(f)
# # 将数据转换为 NumPy 数组
# trajectory_velocity_real = np.array([np.array(item) for item in trajectory_velocity_real])  # 形状 (n_samples, 6)
# n_samples = trajectory_velocity_real.shape[0]
# print(trajectory_velocity_real.shape[0])

# # 创建一个窗口，包含6个子图
# fig, axes = plt.subplots(6, 1, figsize=(12, 18))
# joint_labels = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6']

# # 绘制每个关节的变化曲线
# for i in range(6):
#     axes[i].plot(np.arange(n_samples), trajectory_velocity_real[:, i], label=f'Velocities {joint_labels[i]}', color='green')
#     axes[i].set_ylabel('Angle (rad)')
#     axes[i].legend()
#     axes[i].grid(True)
# axes[5].set_xlabel('Sample')
# plt.tight_layout()
# plt.savefig('./saved_data/trajectory_velocity_real.png')
# ==============================实际关节速度曲线==============================


# ==============================实际关节加速度曲线==============================
# trajectory_acc_real = np.diff(trajectory_velocity_real, axis=0) / 0.04
# n_samples = trajectory_acc_real.shape[0]
# print(trajectory_acc_real.shape[0])

# # 创建一个窗口，包含6个子图
# fig, axes = plt.subplots(6, 1, figsize=(12, 18))
# joint_labels = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6']

# # 绘制每个关节的变化曲线
# for i in range(6):
#     axes[i].plot(np.arange(n_samples), trajectory_acc_real[:, i], label=f'acceleration {joint_labels[i]}', color='orange')
#     axes[i].set_ylabel('Angle (rad)')
#     axes[i].legend()
#     axes[i].grid(True)
# axes[5].set_xlabel('Sample')
# plt.tight_layout()
# plt.savefig('./saved_data/trajectory_acc_real.png')
# ==============================实际关节加速度曲线==============================

# ==============================末端坐标系下接触力曲线==============================
with open('./saved_data/force_data_inTcp.json','r') as f:
    force_data_inTcp = json.load(f)
# 将数据转换为 NumPy 数组
force_data_inTcp = np.array([np.array(item) for item in force_data_inTcp])  # 形状 (n_samples, 6)
print(f"num of force_data_inTcp: {force_data_inTcp.shape[0]}")
# 创建一个窗口，包含6个子图
fig, axes = plt.subplots(6, 1, figsize=(12, 18))
force_labels = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']

# 获取样本数量
n_samples = force_data_inTcp.shape[0]

# 绘制每个关节的变化曲线
for i in range(6):
    axes[i].plot(np.arange(n_samples), force_data_inTcp[:, i], label=f'Force_inTcp {force_labels[i]}', color='magenta')
    axes[i].set_ylabel('Force (N)')
    axes[i].legend()
    axes[i].grid(True)
axes[5].set_xlabel('Sample')
plt.tight_layout()
plt.savefig('./saved_data/force_data_inTcp.png')
# ==============================末端坐标系下接触力曲线==============================

# ==============================基座坐标系下接触力曲线==============================
# with open('./saved_data/force_data_inBase.json','r') as f:
#     force_data_inBase = json.load(f)
# # 将数据转换为 NumPy 数组
# force_data_inBase = np.array([np.array(item) for item in force_data_inBase])  # 形状 (n_samples, 6)
# # 创建一个窗口，包含6个子图
# print(f"num of force_data_inBase: {force_data_inBase.shape[0]}")
# fig, axes = plt.subplots(6, 1, figsize=(12, 18))
# force_labels = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']

# # 获取样本数量
# n_samples = force_data_inBase.shape[0]

# # 绘制每个关节的变化曲线
# for i in range(6):
#     axes[i].plot(np.arange(n_samples), force_data_inBase[:, i], label=f'Force_inBase {force_labels[i]}', color='purple')
#     axes[i].set_ylabel('Force (N)')
#     axes[i].legend()
#     axes[i].grid(True)
# axes[5].set_xlabel('Sample')
# plt.tight_layout()
# plt.savefig('./saved_data/force_data_inBase.png')
# ==============================基座坐标系下接触力曲线==============================

plt.show()