"""
眼在手外 用采集到的图片信息和机械臂位姿信息计算 相机坐标系相对于机械臂基座标的 旋转矩阵和平移向量
A2^{-1}*A1*X=X*B2*B1^{−1}
"""

import os.path
import cv2
import numpy as np

np.set_printoptions(precision=8,suppress=True)

images_path = './collected_images.txt' #手眼标定采集的标定版图片所在路径
poses_path = './collected_poses.txt'  #采集标定板图片时对应的机械臂末端的位姿 从 第一行到最后一行 需要和采集的标定板的图片顺序进行对应

def euler_angles_to_rotation_matrix(rx, ry, rz):
    # 计算旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])

    R = Rz@Ry@Rx  # 先绕 z轴旋转 再绕y轴旋转  最后绕x轴旋转
    return R


def pose_to_homogeneous_matrix(pose):
    x, y, z, rx, ry, rz = pose
    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z]).reshape(3, 1)

    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t[:, 0]

    return H

def inverse_transformation_matrix(T):
    R = T[:3, :3]
    t = T[:3, 3]

    # 计算旋转矩阵的逆矩阵
    R_inv = R.T

    # 计算平移向量的逆矩阵
    t_inv = -np.dot(R_inv, t)

    # 构建逆变换矩阵
    T_inv = np.identity(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv

    return T_inv

def func():

    print("---打开文本文件，读取机器人末端在基坐标系下的位姿---")
    # 打开文本文件
    with open(f'{poses_path}', "r",encoding="utf-8") as f:
        # 读取文件中的所有行
        lines = f.readlines()

    N = len(lines)

    # 遍历每一行数据
    lines = [float(i)  for line in lines for i in line.split(',')]

    matrices = []

    for i in range(0,len(lines),6):
        matrices.append(inverse_transformation_matrix(pose_to_homogeneous_matrix(lines[i:i+6])))
    
    # Assuming matrices is populated as before
    matrices = np.array(matrices)  # Convert matrices to a NumPy array
    # print(matrices.shape)

    R_tool = []
    t_tool = []
    for i in range(matrices.shape[0]):  # Iterate over the first dimension
        # For each 4x4 matrix, extract the rotation matrix and translation vector
        R_tool.append(matrices[i, 0:3, 0:3])  # Rotation matrix (3x3)
        t_tool.append(matrices[i, 0:3, 3].reshape(-1, 1))  # Translation vector (3x1)

    print("---打开文本文件，读取标定板在相机坐标系下的位姿---")
    # 打开文本文件
    with open(f'{images_path}', "r",encoding="utf-8") as f:
        # 读取文件中的所有行
        lines = f.readlines()
    # 定义一个空列表，用于存储结果

    # 遍历每一行数据
    lines = [float(i) for line in lines for i in line.split(',')]

    lines = np.array(lines).reshape(-1,6)
    # print(lines)

    rvecs = []
    tvecs = []

    for i in range(int(N)):
        rvecs.append(lines[i,0:3])
        tvecs.append(lines[i,3:6])

    R, t = cv2.calibrateHandEye(R_tool, t_tool, rvecs, tvecs, cv2.CALIB_HAND_EYE_TSAI)

    # Create a 4x4 identity matrix
    transformation_matrix = np.eye(4)

    # Place R and t into the transformation matrix
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = t.flatten()  # Ensure t is a flat array

    return transformation_matrix

# 旋转矩阵

if __name__ == '__main__':

    tf_matrix = func()
    print(f'transformation_matrix: \n{tf_matrix}')
