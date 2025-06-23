import numpy as np
import matplotlib.pyplot as plt
from pykin.utils import transform_utils as transform_utils


'''
FUNCTION: 位姿相加函数，计算两个位姿的和，位置直接相加，姿态用四元数相加
input: pose1(x1,y1,z1,rx1,ry1,rz1), pose2(x2,y2,z2,rx2,ry2,rz2)
output: pose(x1+x2,y1+y2,z1+z2,rx,ry,rz) = pose1 + pose2
q1 + q2 = q2 * q1
'''
def pose_add(pose1, pose2):

    # 位置直接相加
    pose1 = np.array(pose1); pose2 = np.array(pose2)
    pose = pose1.copy()
    pose[:3] = pose1[:3] + pose2[:3]

    # 姿态用四元数相加
    quater1 = transform_utils.get_quaternion_from_rpy(pose1[3:])
    quater2 = transform_utils.get_quaternion_from_rpy(pose2[3:])
    quater3 = transform_utils.quaternion_multiply(quater2, quater1) # 顺序不能反，先施加quater1再施加quater2,所以quater1在前面
    pose[3:] = transform_utils.get_rpy_from_quaternion(quater3)

    return pose

'''
FUNCTION: 位姿相减函数，计算两个位姿的差，位置直接相减，姿态用四元数相减
input: pose1(x1,y1,z1,rx1,ry1,rz1), pose2(x2,y2,z2,rx2,ry2,rz2)
output: pose(x1-x2,y1-y2,z1-z2,rx,ry,rz) = pose1 - pose2
q1 - q2 = q1 * q2'
'''
def pose_sub(pos1, pos2):

    # 位置直接相减
    pose1 = np.array(pos1); pose2 = np.array(pos2)
    pose = pose1.copy()
    pose[:3] = pose1[:3] - pose2[:3]

    # 姿态用四元数相减
    quater1 = transform_utils.get_quaternion_from_rpy(pose1[3:])
    temp_quater2 = transform_utils.get_quaternion_from_rpy(pose2[3:])
    quater2 = temp_quater2.copy(); quater2[1:] = -quater2[1:] # 逆四元数
    quater3 = transform_utils.quaternion_multiply(quater1, quater2) # 顺序不能反，从quater2旋转到quater1,所以quater2在后面
    pose[3:] = transform_utils.get_rpy_from_quaternion(quater3)

    return pose

class AdmittanceController:
    def __init__(self, mass, stiffness, damping, dt):
        """
        初始化导纳控制器，考虑惯性、刚度和阻尼之间的关系。
        :param mass: 惯性参数列表 [m_x, m_y, m_z, m_roll, m_pitch, m_yaw]
        :param stiffness: 刚度参数列表 [k_x, k_y, k_z, k_roll, k_pitch, k_yaw]
        :param damping_ratio: 阻尼比 ξ
        :param dt: 控制周期（单位：秒）
        """
        self.M = np.diag(mass)  # 惯性矩阵
        self.K = np.diag(stiffness)  # 刚度矩阵
        self.B = np.diag(damping)  # 阻尼矩阵 
        self.dt = dt  # 控制周期

        # 状态初始化
        self.eef_pose = np.zeros(6)  # 当前位姿
        self.eef_vel = np.zeros(6)  # 当前速度

        self.velocity_error = np.zeros(6)  # 速度误差
        self.pose_error = np.zeros(6) # 位姿误差

    def update(self, des_eef_pose, des_eef_vel, measured_force):
        """
        更新导纳控制器的输出
        :param des_eef_pose: 期望六维位置 [x, y, z, roll, pitch, yaw]
        :param des_eef_vel: 期望六维速度 [vx, vy, vz, wx, wy, wz]
        :param measured_force: 实际测量的六维力/力矩向量 [Fx, Fy, Fz, Tx, Ty, Tz]
        :return: 末端执行器的新的期望位姿 [x, y, z, roll, pitch, yaw]
        """

        # 计算位姿误差：位姿误差 = 实际位姿 - 期望位姿
        pose_error = pose_sub(self.eef_pose, des_eef_pose)

        # 计算加速度 a = M^(-1) * (F - B * v - K * x)
        acceleration = np.linalg.inv(self.M) @ (
            measured_force 
            - self.B @ self.velocity_error 
            - self.K @ pose_error 
        )

        # print(f"Acceleration: {acceleration}")

        # 利用加速度更新期望速度误差
        self.velocity_error += acceleration * self.dt
        # print(f"Velocity Error: {self.velocity_error}")

        # 使用期望速度误差更新期望位姿误差：期望位姿误差 = 期望速度误差 * dt
        self.pose_error += self.velocity_error * self.dt
        # print(f"Pose Error: {self.pose_error}")

        # 利用期望位姿误差更新当前位置：当前位置 = 期望位置 + 期望位姿误差
        self.position = pose_add(des_eef_pose, self.pose_error)

        return self.position

# 示例用法
if __name__ == "__main__":
    
    # 定义质量、刚度和阻尼比
    mass = [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]  # 惯性参数
    stiffness = [50.0, 50.0, 50.0, 10.0, 10.0, 10.0]  # 刚度参数
    damping = [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]  # 阻尼参数
    dt = 0.01  # 控制周期（单位：秒）

    # 初始化控制器
    controller = AdmittanceController(mass, stiffness, damping, dt)

    # 期望六维位置和速度
    des_eef_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    des_eef_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # 模拟控制循环
    positions = []
    total_force = []
    total_steps = 500
    half_steps = total_steps // 2
    for step in range(total_steps):
        if step < half_steps:
            z = 10.0 * (step / half_steps)  # 从0升到5
        else:
            z = 10.0 * (1 - (step - half_steps) / half_steps)  # 从5降到0
        measured_force = np.array([0.0, 0.0, 0.0, 0.0, 0.0, z])  # 实际测量的力值
        total_force.append(measured_force.copy())

        # 通过导纳控制器计算新的位置
        updated_position = controller.update(des_eef_pose, des_eef_vel, measured_force)
        positions.append(updated_position.copy())

    # 打印位置调整量
    print(f"Step {step}, Updated Position: {updated_position}")

    plt.figure()
    plt.plot(range(total_steps), [p[5] for p in positions], label="Yaw Position")
    plt.plot(range(total_steps), [f[5] for f in total_force], label="Z Force")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

