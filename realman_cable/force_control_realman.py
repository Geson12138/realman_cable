import os
import sys
import ikpy.inverse_kinematics
import numpy as np
import matplotlib.pyplot as plt
import time
import socket
import threading
import json
from pykin.robots.single_arm import SingleArm
from pykin.utils import transform_utils as transform_utils
from collections import deque
from mpl_toolkits.mplot3d import Axes3D
from admittance_controller import AdmittanceController
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline
import rospy
from srv._ChakouXYZ import *
from std_msgs.msg import Float64MultiArray
import queue
import concurrent.futures
from spatialmath import SE3
from spatialmath import SO3
import logging
from logging.handlers import RotatingFileHandler
from lib.robotic_arm import *
from typing import List,Tuple
from include.predefine_pose import *
from read_forcedata import ForceSensorRS485
from scipy.spatial.transform import Rotation as Ro
from geometry_msgs.msg import PoseArray
from srv._KinInverse import *


# 创建一个logger
logger = logging.getLogger('force_control')

def logger_init():
    
    logger.setLevel(logging.INFO) # Log等级总开关
    # 清除上一次的logger
    if logger.hasHandlers():
        logger.handlers.clear()

    if not os.path.exists('./logfiles'): # 创建log目录
        os.mkdir('./logfiles')

    logfile = './logfiles/force_control.log' # 创建一个handler，用于写入日志文件
    fh = RotatingFileHandler(logfile, mode='a', maxBytes=1024*1024*50, backupCount=30) # 以append模式打开日志文件
    fh.setLevel(logging.INFO) # 输出到file的log等级的开关

    ch = logging.StreamHandler() # 再创建一个handler，用于输出到控制台
    ch.setLevel(logging.INFO) # 输出到console的log等级的开关

    formatter = logging.Formatter("%(asctime)s [%(thread)u] %(levelname)s: %(message)s") # 定义handler的输出格式
    fh.setFormatter(formatter) # 为文件输出设定格式
    ch.setFormatter(formatter) # 控制台输出设定格式

    logger.addHandler(fh) # 设置文件输出到logger
    logger.addHandler(ch) # 设置控制台输出到logger


# 全局变量存储udp上报数据
latest_data = {
    'joint_position': [0.0] * 7,  # 7个关节角度
    'pose': [0.0] * 6,            # 前3个是position(x,y,z)，后3个是euler(rx,ry,rz)
    'zero_force': [0.0] * 6       # 传感器坐标系下系统收到的外力
}

# 全局变量储存插口pose
headcam_pose = PoseArray()
armcam_ri_pose = PoseArray()
recent_peg_positions = []
recent_hole_positions = []
peg_avg_position = np.zeros((3,1))
hole_avg_position = np.zeros((3,1))

def headcam_pose_callback(msg):

    global headcam_pose
    headcam_pose = msg
    # print(headcam_pose.poses[5].position.x)

def armcam_ri_pose_callback(msg):
    """
    armcam_ri_pose.poses[0] 是插口 pose
    armcam_ri_pose.poses[1] 是插头 pose
    """    
    global armcam_ri_pose
    armcam_ri_pose = msg

    global recent_peg_positions,recent_hole_positions,peg_avg_position,hole_avg_position

    peg_pos = msg.poses[1].position # 获取插头的(x, y, z)
    peg_position = np.array([peg_pos.x, peg_pos.y, peg_pos.z])
    hole_pos = msg.poses[0].position # 获取插口的(x, y, z)
    hole_position = np.array([hole_pos.x, hole_pos.y, hole_pos.z])

    # 保存到列表
    recent_peg_positions.append(peg_position)
    recent_hole_positions.append(hole_position)

    if len(recent_peg_positions) > 20:
        recent_peg_positions.pop(0)  # 移除最早的

    if len(recent_hole_positions) > 20:
        recent_hole_positions.pop(0)  # 移除最早的    

    # 计算平均值
    peg_avg_position = np.mean(recent_peg_positions, axis=0)
    hole_avg_position = np.mean(recent_hole_positions, axis=0)

    # logger.info(armcam_ri_pose.poses[0].position.x)

def pose_force_publisher():

    global raw_force_data, latest_data

    pub1 = rospy.Publisher('/right_arm_pose', Float64MultiArray, queue_size=10)
    pub2 = rospy.Publisher('/right_arm_force', Float64MultiArray, queue_size=10)
    rate = rospy.Rate(10)  # 10Hz发布频率
    
    while pub_pose_force_running:

        pose_msg = Float64MultiArray()
        pose_msg.data = latest_data['pose']
        
        force_msg = Float64MultiArray()
        force_msg.data = raw_force_data
        
        # 发布消息
        pub1.publish(pose_msg)
        pub2.publish(force_msg)
        
        rate.sleep()

def arm_status(data):

    latest_data['joint_position'] = np.array(data.joint_status.joint_position) # 角度制
    cur_ee_pos = np.array([data.waypoint.position.x, data.waypoint.position.y, data.waypoint.position.z])
    cur_ee_ori = np.array([data.waypoint.euler.rx, data.waypoint.euler.ry, data.waypoint.euler.rz])
    latest_data['pose'] = np.concatenate((cur_ee_pos, cur_ee_ori))
    latest_data['zero_force'] = np.array(data.force_sensor.zero_force)


def udp_arm_state() -> Tuple[List[float], List[float]]:
    """
    获取当前机械臂信息
    """
    # logger.info(f"joint: {[round(float(j), 2) for j in latest_data['joint_position'].copy()]}")
    # logger.info(f"pose: {[round(float(j), 3) for j in latest_data['pose'].copy()]}")
    # logger.info(f"zero_force: {[round(float(j), 3) for j in latest_data['zero_force'].copy()]}")
    return latest_data['joint_position'].copy(), latest_data['pose'].copy(), latest_data['zero_force'].copy()


def read_forcedata(sensor):
    """
    读取力传感器数据的线程函数
    """
    # 启动数据流并测量频率
    sensor.save_forcedata()

raw_force_data = np.zeros(6) # fx fy fz Mx My Mz 原始力传感器数据
calib_force_data = np.zeros(6) # 标定后的力数据
force_zero_point = np.zeros(6) # 力传感器数据清零
read_force_data_running = True # 读取数据线程运行标志
G_force = np.zeros(6) # Gx Gy Gz Mgx Mgy Mgz 负载重力在力传感器坐标系下的分量

def calib_force_data_func(robot):
    '''
    校准力传感器数据, 输出负载辨识和标定后的纯外力数据
    '''

    global calib_force_data,read_force_data_running, raw_force_data, force_zero_point

    # 读取 calibration_result.JSON 文件得到标定结果
    file_name = './calibrate_forceSensor/calibration_result.json'
    with open(file_name, 'r') as json_file:
        json_data = json.load(json_file)
        gravity_bias = np.array(json_data['gravity_bias'])
        mass = np.array(json_data['mass'][0])
        force_zero = np.array(json_data['force_zero'])

    # # 初始化一个队列用于存储前5次的力传感器数据
    # length_force_data_queue = 20
    # force_data_queue = deque(maxlen=length_force_data_queue)

    # # 定义滤波器参数
    # cutoff = 3  # 截止频率 (Hz)
    # order = 2    # 滤波器阶数
    # fs = 200  # 根据实际情况设置采样频率 (Hz)
    # b, a = butter(order, cutoff / (0.5 * fs), btype='low', analog=False) # 设计低通Butterworth滤波器

    alpha = 0.9 # 滑动窗口加权滤波
    smoothed_force_data = np.zeros(6) # 平滑后的力传感器数据
    time_count = 0

    start_time = time.time()
    
    while read_force_data_running:
        
        if raw_force_data is not None:

            # st_time = time.perf_counter() # 记录开始时间
            
            raw_force_data = np.array(sensor.forcedata).copy()

            # # 将当前的力传感器数据添加到队列中
            # force_data_queue.append(raw_force_data)
            # # 如果队列中的数据少于5次，则继续等待
            # if len(force_data_queue) < length_force_data_queue:
            #     continue
            # # 将 force_data_queue 转换为 NumPy 数组
            # force_data_array = np.array(force_data_queue)
            # # 应用滤波器
            # filtered_signal = filtfilt(b, a, force_data_array, axis=0)
            # new_force_data = filtered_signal[-1]

            smoothed_force_data = alpha * raw_force_data  + (1-alpha) * smoothed_force_data

            '''
            获取机械臂当前的力传感器坐标系的末端姿态, 需要根据不同的机械臂型号和通信协议获取
            '''
            joint_r, _, _ = udp_arm_state()
            joints_r_rad = [j/180*np.pi for j in joint_r]
            r_ee = pykin_fsensor.forward_kin(joints_r_rad)
            r_ee_rot = r_ee[pykin_fsensor.eef_name].rot
            # logger.info(f"r_ee_rot: {r_ee_rot}") # 打印末端姿态四元数

            '''
            计算传感器坐标系下的外部力数据 = 原始力数据 - 力传感器零点 - 负载重力分量
            '''
            rotation_matrix = np.array(transform_utils.get_matrix_from_quaternion(r_ee_rot))
            inv_rotation_matrix = np.linalg.inv(rotation_matrix) # 3*3
            temp_gravity_vector = np.array([0, 0, -1]).reshape(3, 1) # 3*1
            gravity_vector = np.dot(inv_rotation_matrix, temp_gravity_vector) # 3*1
            G_force[:3] = np.transpose(mass * gravity_vector) # 3*1 Gx Gy Gz
            G_force[3] = G_force[2] * gravity_bias[1] - G_force[1] * gravity_bias[2] # Mgx = Gz × y − Gy × z 
            G_force[4] = G_force[0] * gravity_bias[2] - G_force[2] * gravity_bias[0] # Mgy = Gx × z − Gz × x
            G_force[5] = G_force[1] * gravity_bias[0] - G_force[0] * gravity_bias[1] # Mgz = Gy × x − Gx × y

            # 标定后的传感器坐标系下的力数据 = 原始力传感器数据 - 力传感器零点 - 负载重力分量 - 手动清零分量
            calib_force_data = smoothed_force_data - force_zero - G_force - force_zero_point # 6*1
            force2tcp = SE3.Trans(0.0, 0.0, -0.199) * SE3.RPY(0, 0, 1.57) # 如何从末端工具坐标系转换到力传感器坐标系，注意不是另一种
            contact_force_data_inTcp = trans_force_data(force2tcp.A, calib_force_data) # 6*1
            # logger.info(f'Tcp坐标系下的受力:{contact_force_data_inTcp}')
            
            # 保存数据
            force_data_inSensor_list.append(calib_force_data) # 末端受到的外部力
            time_list.append(time.time() - start_time) # 时间
            # logger.info(f"calib_force_data: {calib_force_data}") # 打印标定后的力数据
            # logger.info(f"消耗时间: {time.perf_counter() - st_time:.4f}秒")

            time.sleep(0.001) 
        
        else :

            logger.info("force_data is None, 没有收到原始力传感器数据")
            time.sleep(0.005)

def force_data_zero():
    '''
    力传感器数据清零
    '''
    global force_zero_point, calib_force_data
    force_zero_point = calib_force_data.copy()

def pose_add(pose1, pose2):
    '''
    位姿相加函数，计算两个位姿的和，位置直接相加，姿态用四元数相加: q1 + q2 = q2 * q1
    param pose1: 位姿1 [x1,y1,z1,rx1,ry1,rz1]
    param pose2: 位姿2 [x2,y2,z2,rx2,ry2,rz2]
    return: 位姿 [x1+x2,y1+y2,z1+z2,rx,ry,rz] = pose1 + pose2
    '''
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


def pose_sub(pos1, pos2):
    '''
    位姿相减函数，计算两个位姿的差，位置直接相减，姿态用四元数相减: q1 - q2 = q2' * q1
    param pos1: 位姿1 [x1,y1,z1,rx1,ry1,rz1]
    param pos2: 位姿2 [x2,y2,z2,rx2,ry2,rz2]
    return: 位姿 [x1-x2,y1-y2,z1-z2,rx,ry,rz] = pos1 - pos2 
    '''
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

# def robot_inverse_kinematic(refer_joints,target_pos,target_ori_rpy_rad):
#     '''
#     逆运动学求解关节角驱动机械臂运动
#     :param target_pos: 末端期望位置 / m
#     :param target_ori_rpy_rad 末端期望角度rqy / degree
#     :return: 逆运动学求解的关节角度 / rad
#     '''
#     global call_count, total_time
#     call_count += 1
#     start_time = time.time()
    
#     target_rot = transform_utils.get_matrix_from_rpy(target_ori_rpy_rad)
#     target_pose = np.eye(4);target_pose[:3, :3] = target_rot; target_pose[:3, 3] = target_pos
#     r_qd = pykin_ri.inverse_kin([j/180*np.pi for j in refer_joints], target_pose, method="LM", max_iter=20)
#     arm_joint = [j/np.pi*180 for j in r_qd]
    
#     elapsed_time = time.time() - start_time
#     total_time += elapsed_time
    
#     return arm_joint

def robot_inverse_kinematic(refer_joints,target_pos,target_ori_rpy_rad):
    
    target_pose = [0.0] * 6
    target_pose[:3] = target_pos
    target_pose[3:] = target_ori_rpy_rad
    rsp = kin_inverse(refer_joints, target_pose)
    
    return rsp.joint_angles
    


def generate_trajectory_with_constraints(q_start, v_start, a_start, q_end, v_end, a_end, move_period, lookahead_time, max_velocity, max_acceleration):
    """
    生成五次多项式轨迹，确保速度和加速度不超限，并返回时间戳
    :param q_start: 初始位置 (数组)
    :param v_start: 初始速度 (数组)
    :param a_start: 初始加速度 (数组)
    :param q_end: 目标位置 (数组)
    :param v_end: 目标速度 (数组)
    :param a_end: 目标加速度 (数组)
    :param move_period: 轨迹采样时间间隔
    :param lookahead_time: 轨迹总时长
    :param max_velocity: 允许的最大速度
    :param max_acceleration: 允许的最大加速度
    :return: 时间戳、轨迹的位置信息、速度信息、加速度信息、最终轨迹时长
    """
    num_joints = len(q_start)
    
    # 动态调整轨迹时间，保证最大速度和最大加速度
    T = lookahead_time
    st_time = time.time()

    while True:
        t_vals = np.arange(0, T + move_period, move_period)  # 生成时间戳
        positions = np.zeros((len(t_vals), num_joints))
        velocities = np.zeros((len(t_vals), num_joints))
        accelerations = np.zeros((len(t_vals), num_joints))

        max_v = 0
        max_a = 0

        for j in range(num_joints):
            # 计算五次多项式系数
            A = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 2, 0, 0, 0],
                [1, T, T**2, T**3, T**4, T**5],
                [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
                [0, 0, 2, 6*T, 12*T**2, 20*T**3]
            ])
            B = np.array([q_start[j], v_start[j], a_start[j], q_end[j], v_end[j], a_end[j]])
            coeffs = np.linalg.solve(A, B)

            # 计算轨迹
            positions[:, j] = coeffs[0] + coeffs[1]*t_vals + coeffs[2]*t_vals**2 + coeffs[3]*t_vals**3 + coeffs[4]*t_vals**4 + coeffs[5]*t_vals**5
            velocities[:, j] = coeffs[1] + 2*coeffs[2]*t_vals + 3*coeffs[3]*t_vals**2 + 4*coeffs[4]*t_vals**3 + 5*coeffs[5]*t_vals**4
            accelerations[:, j] = 2*coeffs[2] + 6*coeffs[3]*t_vals + 12*coeffs[4]*t_vals**2 + 20*coeffs[5]*t_vals**3

            # 记录最大速度和加速度
            max_v = max(max_v, np.max(np.abs(velocities[:, j])))
            max_a = max(max_a, np.max(np.abs(accelerations[:, j])))
            
        
        # logger.info(f'max_v:{max_v}')
        # logger.info(f'max_a:{max_a}')
        # 判断是否满足最大速度和最大加速度的约束 或者超过了一定时间限制
        if (max_v <= max_velocity and max_a <= max_acceleration) or T >= 1.0:
            # logger.info(f"T: {T}")
            break  # 如果满足约束，停止调整
        
        T +=0.03  # 增大轨迹时间，降低速度和加速度
    
    # print(f"planning time: {time.time()-st_time}")
    
    return t_vals, positions, velocities, accelerations, T  # 返回时间戳


def trans_force_data(trans_matrix, force_data):
    '''
    将力传感器数据从A坐标系转换到B坐标系
    :param trans_matrix: 坐标系转换矩阵: 从A坐标系到B坐标系, B坐标系下A坐标系的位姿
    :param force_data: A坐标系下的力传感器数据 [Fx, Fy, Fz, Tx, Ty, Tz]
    :return: B坐标系下的力传感器数据 [Fx, Fy, Fz, Tx, Ty, Tz]
    '''
    trans = trans_matrix[:3,3]; oritation = trans_matrix[:3,:3]
    trans_inv = np.array([
        [0, -trans[2], trans[1]],
        [trans[2], 0, -trans[0]],
        [-trans[1], trans[0], 0]   
    ])
    # 伴随矩阵
    adjoint_matrix = np.zeros((6,6))
    adjoint_matrix[:3,:3] = oritation; adjoint_matrix[3:,3:] = oritation
    adjoint_matrix[:3,3:] = 0; adjoint_matrix[3:, :3] = np.dot(trans_inv, oritation)
    new_force_data = np.dot(adjoint_matrix, force_data)

    return new_force_data

def admittance_controller(robot):
    '''
    阻抗控制器, 用于机械臂外环控制, 控制周期25hz
    '''
    global num_joints, flag_enter_tcp2canbus, calib_force_data, p_end_list

    # 设置admittance control阻抗控制器参数
    mass = [300.0, 300.0, 300.0, 50.0, 50.0, 50.0]  # 定义惯性参数
    damping = [700.0, 700.0, 700.0, 120.0, 120.0, 120.0]  # 定义阻尼参数
    stiffness = [300.0, 300.0, 300.0, 50.0, 50.0, 50.0]  # 定义刚度参数
    control_period = 0.05 # 阻抗控制周期（单位：秒）要跟状态反馈时间保持一致，用于外环阻抗控制循环

    # 初始化导纳参数
    controller = AdmittanceController(mass, stiffness, damping, control_period)

    # 期望末端位置和速度
    des_eef_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    des_eef_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # 获取当前末端位姿作为起点的位置和姿态
    _, joint_r, pose_r, _, _ = robot.Get_Current_Arm_State()
    p_start_pos = np.array((pose_r[:3])) # 当前末端位置 in m
    p_start_ori = np.array((pose_r[3:])) # # 当前末端姿态 in rad
    p_end_pos = p_start_pos.copy() # 初始化期望的末端位置 in m
    p_end_ori = p_start_ori.copy() # 初始化期望的末端姿态 in rad

    # 5次多项式轨迹规划参数
    joint_vel_limit =  300 # 每个关节的最大角速度 (deg/s) 约束
    joint_acc_limit = 1000  # 计算最大角加速度 （deg/s^2）约束
    prev_joint_vel = np.zeros(num_joints) # 保存上一周期周期的关节速度
    prev_joint_acc = np.zeros(num_joints) # 保存上一周期周期的关节加速度

    # 力的初始化
    user_force_inTcp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # 用户主动施加力， Tcp坐标系下
    user_force_inBase = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # 用户主动施加力，Base坐标系下
    judge_force = 0.0 # 初始化力传感器受力, Tcp坐标系下的受力
    judge_moment = 0.0 # 初始化力传感器受力矩, Tcp坐标系下的受力
    judge_force_threshold = 0.0 # 初始化力传感器受力判断阈值, Tcp坐标系下的受力
    judge_moment_threshold = 0.0 # 初始化力传感器受力矩判断阈值, Tcp坐标系下的受力

    # input("enter to continue 前进")

    '''
    前进阶段
    '''
    approaching_numPoints = 300 # 前进阶段的点数
    trans_length = 0.15  # 前进距离
    delta_steps = trans_length / approaching_numPoints # 前进的每一步的步长
    judge_force_threshold = -3 # -N
    ee_rotateMatrix = SO3.RPY(p_start_ori[0], p_start_ori[1], p_start_ori[2])  # 生成3D旋转对象(0.0, 0.0, 0.0)
    trans_vector_inTcp = np.array([delta_steps, 0.0, 0.0]).reshape(3,1) # Tcp坐标系下的前进位移向量
    trans_vector_inBase = (ee_rotateMatrix * trans_vector_inTcp ).flatten() # 转换到Base坐标系下的前进位移向量
    force2tcp = SE3.Trans(0.0, 0.0, -0.199) * SE3.RPY(1.57, 2.356, 0.0, order='xyz') # 从末端工具坐标系转换到力传感器坐标系
    approaching_control_period = 0.050 # 透传周期50ms
    for i_0 in range(approaching_numPoints):

        logger.info("第{}个".format(i_0)+": 前进阶段")

        st_time = time.perf_counter()
        p_end_pos += trans_vector_inBase # 前半段用规划的轨迹，x向前
        p_end_list.append(np.concatenate((p_end_pos, p_end_ori), axis=0)) # 存储末端期望位姿

        q_target = np.array(robot_inverse_kinematic(joint_r, p_end_pos, p_end_ori))
        robot.Movej_CANFD(q_target,False,0) # 透传目标位置给机械臂
        all_positions_desired.append(q_target) # 保存期望关节角度

        joint_r, pose_r, _ = udp_arm_state()
        all_positions_real.append(joint_r) # 存储当前关节角
        pose_data_list.append(pose_r) # # 存储末端当前位姿

        contact_force_data_inTcp = trans_force_data(force2tcp.A, calib_force_data) # 力传感器坐标系下力转换到末端坐标系
        all_contact_force_inTcp.append(contact_force_data_inTcp) # 保存末端坐标系下的受力
        judge_force = contact_force_data_inTcp[0].copy() # 判断力传感器受力是否超过阈值
        logger.info(f'judge_force: {judge_force}')

        if judge_force <= judge_force_threshold:
            logger.info("前进阶段结束，进入搜孔阶段")
            break

        elapsed_time = time.perf_counter()-st_time
        logger.info(f"time cost: {elapsed_time}")
        if elapsed_time > approaching_control_period: # 固定透传周期
            pass
        else:
            time.sleep(approaching_control_period - elapsed_time)


    time.sleep(1) # 等待稳定
    _, pose_r, _ = udp_arm_state()
    p_end_pos = np.array((pose_r[:3])) # 更新当前末端位置 in m
    p_end_ori = np.array((pose_r[3:])) # 更新当前末端姿态 in rad

    # input("enter to continue 搜孔")

    '''
    视觉引导搜孔阶段
    '''
    searching_phase = True # 搜孔阶段标志位
    searching_numPoints = 3000 # 搜孔阶段的点数
    spiral_loops = 10  # 螺旋圈数
    gap = 0.001 # 螺距系数，控制螺旋间距
    peg_pos_inTcp = arm_cam2robotiq_right @ np.array([peg_avg_position[0],peg_avg_position[1],peg_avg_position[2],1]).reshape(4,1)
    hole_pos_inTcp = arm_cam2robotiq_right @ np.array([hole_avg_position[0],hole_avg_position[1],hole_avg_position[2],1]).reshape(4,1)
    vector_peg2hole_inTcp = np.abs(np.array(hole_pos_inTcp - peg_pos_inTcp)[:3].reshape(3, 1)) # (3,1)
    vector_peg2hole_inTcp[0] *= 0.40 # x往里走的距离减小，减小的话往里的力也会小
    vector_peg2hole_inTcp[1] *= 0.70 # y往左走的距离减小
    vector_peg2hole_inTcp[2] *= 1.05 # z往上走的距离减小
    norm = np.linalg.norm(vector_peg2hole_inTcp) # 计算模长
    if norm == 0:
        unit_vector_inTcp = np.zeros_like(vector_peg2hole_inTcp)
    else:
        unit_vector_inTcp = vector_peg2hole_inTcp / norm  # 单位向量  unit: m
    trans_vector_inTcp = unit_vector_inTcp / searching_numPoints  # 固定长度、方向与原向量平行 unit: mm
    ee_rotateMatrix = SO3.RPY(pose_r[3], pose_r[4], pose_r[5])  # 生成3D旋转对象(0.0, 0.0, 0.0)
    trans_vector_inBase = (ee_rotateMatrix * trans_vector_inTcp ).flatten() # 转换到Base坐标系下的前进位移向量
    searching_control_period = 0.050 # 透传周期50ms 
    judge_force_threshold = 10 # 搜孔阶段受力判断阈值, Tcp坐标系下的受力

    for i_1 in range(searching_numPoints):

        logger.info("第{}个".format(i_0+i_1+1)+": 搜孔阶段")

        st_time = time.perf_counter()
        p_end_pos += trans_vector_inBase # x往前走，yz做螺旋运动
        p_end_list.append(np.concatenate((p_end_pos, p_end_ori), axis=0)) # 存储末端期望位姿

        q_target = np.array(robot_inverse_kinematic(joint_r, p_end_pos, p_end_ori))
        robot.Movej_CANFD(q_target,False,0) # 透传目标位置给机械臂
        all_positions_desired.append(q_target) # 保存期望关节角度

        joint_r, pose_r, _ = udp_arm_state()
        all_positions_real.append(joint_r) # 存储当前关节角
        pose_data_list.append(pose_r) # # 存储末端当前位姿

        contact_force_data_inTcp = trans_force_data(force2tcp.A, calib_force_data) # 6*1
        all_contact_force_inTcp.append(contact_force_data_inTcp) # 保存末端坐标系下的受力
        judge_force = np.sqrt(np.sum(contact_force_data_inTcp[1]**2 + contact_force_data_inTcp[2]**2)) # 判断力传感器受力是否超过阈值
        logger.info(f'judge_force: {judge_force}')

        if judge_force >= judge_force_threshold:
            searching_phase = False # 搜孔阶段接触
            logger.info("搜孔阶段结束，进入插孔第一阶段")
            break

        elapsed_time = time.perf_counter()-st_time
        logger.info(f"time cost: {elapsed_time}")
        if elapsed_time > searching_control_period: # 固定透传周期
            pass
        else:
            time.sleep(searching_control_period - elapsed_time)

    
    _, pose_r, _ = udp_arm_state()
    p_end_pos = np.array((pose_r[:3])) # 更新当前末端位置 in m
    p_end_ori = np.array((pose_r[3:])) # 更新当前末端姿态 in rad

    # input("enter to continue 插接")

    '''
    环境吸引域插接
    '''
    # 整个控制周期的不同阶段
    insertion_numPoints = 3000  # 插接的点数
    first_phase = True # 第一阶段标志位，先往里插
    second_phase = False # 第二阶段标志位，主动偏转
    second_phase_startPoint = 0 # 第二阶段开始点
    third_phase = False # 第三阶段标志位，直插到底

    trans_params_Tcp2Base = np.zeros((6,6)) # 将阻抗参数从末端坐标系转换到基坐标系下运算

    # 进入canfd透传模式
    flag_enter_tcp2canbus = True
    logger.info("进入外环导纳控制周期，控制周期{}ms".format(control_period*1000))

    for i_2 in range(insertion_numPoints):

        st_time = time.time()
        logger.info("第{}个".format(i_0+i_1+i_2+2))

        '''
        获取当前关节状态和末端状态
        '''
        joint_r, pose_r, _ = udp_arm_state()
        cur_joint = np.array(joint_r) 
        # logger.info(f'当前关节角: {cur_joint}')
        # logger.info(f'当前末端姿态: {pose_r}')
        all_positions_real.append(cur_joint) # 存储当前关节角
        
        controller.eef_pose = pose_r.copy() # 用于导纳控制计算的6*1
        pose_data_list.append(controller.eef_pose) # # 存储末端当前位姿 6*1
        ee_pos = np.array((pose_r[:3])) # 当前末端位置 in m
        ee_ori_rpy_rad = np.array((pose_r[3:])) # # 当前末端姿态 in rad
        tcp_pose = SE3.Trans(ee_pos[0], ee_pos[1], ee_pos[2]) * SE3.RPY(ee_ori_rpy_rad[0], ee_ori_rpy_rad[1], ee_ori_rpy_rad[2]) # 4*4
        
        '''
        对不同阶段做运动规划
        ''' 
        if first_phase: # 插孔的第一阶段，先往里插

            judge_force_threshold = -3 # 插孔的第一阶段受力判断阈值, Tcp坐标系下的受力

            trans_vector_inTcp = np.array([0.05/1000, 0.0, 0.0]).reshape(3,1) # Tcp坐标系下的前进位移向量
            ee_rotateMatrix = SO3.RPY(ee_ori_rpy_rad[0], ee_ori_rpy_rad[1], ee_ori_rpy_rad[2])  # 生成3D旋转对象(0.0, 0.0, 0.0)
            trans_vector_inBase = (ee_rotateMatrix * trans_vector_inTcp).flatten() # 转换到Base坐标系下的前进位移向量
            p_end_pos = p_end_pos + trans_vector_inBase # x往前走

            # user_force_inTcp = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # 用户主动施加力，Tcp坐标系下
            logger.info("插孔的第一阶段: 先往里插")

        elif second_phase: #  插孔第二阶段，主动偏转


            # mass_params = np.diag([300.0, 300.0, 300.0, 60.0, 60.0, 60.0]) # 修改质量参数
            # damping_params = np.diag([700.0, 700.0, 700.0, 120.0, 120.0, 120.0]) # 修改阻尼参数
            # stiffness_params = np.diag([300.0, 300.0, 300.0, 0.0, 0.0, 0.0]) # 修改刚度参数
            # trans_params_Tcp2Base[:3,:3] = rotation_matrix
            # trans_params_Tcp2Base[3:,3:] = rotation_matrix
            # controller.M = trans_params_Tcp2Base @ mass_params @ trans_params_Tcp2Base.T
            # controller.B = trans_params_Tcp2Base @ damping_params @ trans_params_Tcp2Base.T
            # controller.K = trans_params_Tcp2Base @ stiffness_params @ trans_params_Tcp2Base.T

            fingertip_in_tcp = SE3.Trans(0.018, 0.0, 0.0) * SE3.RPY(0.0, 0.0, 0.0) # 指尖在末端下的变换
            tcp_in_fingertip = fingertip_in_tcp.inv() # 末端在指尖下的变换（逆变换）

            if second_phase_startforce > 0: # 如果是绕着Tcp_y轴的正向扭矩
                delta_theta = -2/180*np.pi  # 每周期旋转的弧度
                judge_moment_threshold = -0.5 # Tcp坐标系下的绕y轴的力矩My的阈值
            else:
                delta_theta =  2/180*np.pi  # 每周期旋转的弧度
                judge_moment_threshold = 0.5 # Tcp坐标系下的绕y轴的力矩My的阈值

            rotate_y_in_fingertip = SE3.RPY(0.0, delta_theta, 0.0)
            new_tcp_pose = tcp_pose * fingertip_in_tcp * rotate_y_in_fingertip * tcp_in_fingertip # 先变换到指尖坐标系，再绕y轴旋转，再变回末端坐标系
            new_pose = new_tcp_pose.A  # 如果需要numpy数组

            p_end_pos = new_pose[:3,3] 
            p_end_ori = transform_utils.get_rpy_from_matrix(new_pose[:3,:3])


        elif third_phase: # 插孔的第三阶段，直插到底

            judge_force_threshold = -20 # 插孔的第三阶段受力判断阈值, Tcp坐标系下的受力

            trans_vector_inTcp = np.array([0.05/1000, 0.0, 0.0]).reshape(3,1) # Tcp坐标系下的前进位移向量
            ee_rotateMatrix = SO3.RPY(ee_ori_rpy_rad[0], ee_ori_rpy_rad[1], ee_ori_rpy_rad[2])  # 生成3D旋转对象(0.0, 0.0, 0.0)
            trans_vector_inBase = (ee_rotateMatrix * trans_vector_inTcp ).flatten() # 转换到Base坐标系下的前进位移向量
            p_end_pos = p_end_pos + trans_vector_inBase # x往前走
        
        else:
            pass

        des_eef_pose = np.concatenate((p_end_pos, p_end_ori), axis=0) # 根据规划的路径获取旧的末端期望位姿 6*1
        p_end_list.append(des_eef_pose) # 存储末端期望位姿

        '''
        接触力在不同坐标系间转换
        '''
        # 将传感器坐标系下的力转换到末端坐标系下，转换后受力坐标系为末端坐标系
        force2tcp = SE3.Trans(0.0, 0.0, -0.199) * SE3.RPY(1.57, 2.356, 0.0, order='xyz') # 如何从末端工具坐标系转换到力传感器坐标系，注意不是另一种
        contact_force_data_inTcp = trans_force_data(force2tcp.A, calib_force_data) # 6*1

        '''
        通过接触力进行阶段判断和转移
        '''
        if first_phase: # 插孔第一阶段，先往里插

            contact_force_data_inTcp[1] *= 0.01 # 减小末端y方向的受力
            contact_force_data_inTcp[2] *= 0.01 # 减小末端z方向的受力

            judge_force = contact_force_data_inTcp[0] # Tcp坐标系下受力，为-x，朝后
            logger.info(f'judge_force: {judge_force}')
            logger.info("插孔的第一阶段: 先往里插")

            if judge_force <= judge_force_threshold: # 如果受力超过阈值，说明往里抵住了
                first_phase = False
                second_phase = True
                controller.pose_error = np.zeros(6) # 重置导纳控制器的位姿误差
                controller.velocity_error = np.zeros(6) # 重置导纳控制器的速度误差
                second_phase_startPoint = i_2 # 记录偏转阶段开始时刻
                second_phase_startforce = contact_force_data_inTcp[4] # 记录偏转阶段开始时的My
                logger.info("插孔第一阶段结束，进入第二阶段主动偏转")

        elif second_phase: # 插孔第二阶段，主动偏转

            contact_force_data_inTcp[1] *= 0.01 # 减小末端y方向的受力
            contact_force_data_inTcp[2] *= 0.01 # 减小末端z方向的受力
            contact_force_data_inTcp[3] *= 0.01 # 减小末端绕x轴的力矩
            contact_force_data_inTcp[5] *= 0.01 # 减小末端绕z轴的力矩
            user_force_inTcp[0] = -contact_force_data_inTcp[0]  # 用户主动施加力，Tcp坐标系下

            judge_momentX = contact_force_data_inTcp[3] # Tcp坐标系下的绕x轴的力矩，Mx，如果插头平行了插孔，Mx应该很小
            judge_momentY = contact_force_data_inTcp[4] # Tcp坐标系下的绕y轴的力矩，My，如果插头平行了插孔，My应该很小
            judge_momentZ = contact_force_data_inTcp[5] # Tcp坐标系下的绕z轴的力矩，Mz，如果插头平行了插孔，Mz应该很小
            logger.info(f'judge_momentX: {judge_momentX}')
            logger.info(f'judge_momentY: {judge_momentY}')
            logger.info(f'judge_momentZ: {judge_momentZ}')
            logger.info("插孔的第二阶段：偏转")

            # Mx, My, Mz均小于设定阈值，说明插头对齐了插孔
            if i_2 - second_phase_startPoint >= 30 and  judge_momentY <= judge_moment_threshold: 
                second_phase = False
                third_phase = True
                controller.pose_error = np.zeros(6) # 重置导纳控制器的位姿误差
                controller.velocity_error = np.zeros(6) # 重置导纳控制器的速度误差
                _, pose_r, _ = udp_arm_state()
                p_end_pos = np.array((pose_r[:3])) # 更新当前末端位置 in m
                p_end_ori = np.array((pose_r[3:])) # 更新当前末端姿态 in rad
                logger.info("插孔的第二阶段结束，进入第三阶段直插到底")
                
        elif third_phase: # 插孔第三阶段，直插到底

            contact_force_data_inTcp[1] *= 0.01 # 减小末端y方向的受力
            contact_force_data_inTcp[2] *= 0.01 # 减小末端z方向的受力
            contact_force_data_inTcp[3] *= 0.01 # 减小末端绕x轴的力矩
            contact_force_data_inTcp[4] *= 0.01 # 减小末端绕y轴的力矩
            contact_force_data_inTcp[5] *= 0.01 # 减小末端绕z轴的力矩
            user_force_inTcp[0] = -contact_force_data_inTcp[0] + 10 # 用户主动施加力，Tcp坐标系下
                
            judge_force = contact_force_data_inTcp[0] # Tcp坐标系下受力，为-x，朝后
            logger.info(f'judge_force: {judge_force}')
            logger.info("插孔的第三阶段: 直插到底")

            if judge_force <= judge_force_threshold: # 如果受力超过阈值，说明插孔结束
                third_phase = False
                logger.info("插孔结束")
                break
        else:
            pass
        
        compute_st_time = time.perf_counter() # 计算耗时的起点

        contact_force_data_inTcp = contact_force_data_inTcp + user_force_inTcp # 加上用户期望受力，末端坐标系下
        all_contact_force_inTcp.append(contact_force_data_inTcp) # 保存末端坐标系下的受力

        # 将末端坐标系下的力转换到基坐标系下
        contact_force_data_inBase = trans_force_data(tcp_pose.A, contact_force_data_inTcp) # 6*1
        contact_force_data = contact_force_data_inBase + user_force_inBase # 加上用户期望受力，基坐标系下
        all_contact_force_inBase.append(contact_force_data) # 保存基坐标系下的受力

        # 通过导纳控制器计算新的末端期望位姿
        updated_eef_pose = controller.update(des_eef_pose, des_eef_vel, contact_force_data)
        p_slover_pos = updated_eef_pose[:3]
        p_slover_ori = updated_eef_pose[3:]

        '''
        对导纳计算结果的末端位姿进行处理: 放松or抱紧
        '''
        if second_phase:
            # p_slover_pos = p_end_pos
            p_slover_ori = p_end_ori

        # 逆运动学计算
        q_target = np.array(robot_inverse_kinematic(cur_joint, p_slover_pos, p_slover_ori))

        # 设置期望位置、速度和加速度
        v_target = np.zeros(num_joints)
        a_target = np.zeros(num_joints)

        # 5次多项式轨迹规划生成平滑轨迹
        _, positions, velocities, _, _ = generate_trajectory_with_constraints(
            cur_joint, prev_joint_vel, prev_joint_acc, q_target, v_target, a_target, control_period, lookahead_time, joint_vel_limit, joint_acc_limit)

        # 将规划好的轨迹放进共享队列
        trajectory_queue.put(positions)

        logger.info(f"compute cost time: {time.perf_counter()-compute_st_time:.6f}")

        # 五次多项式轨迹规划
        prev_joint_acc = (velocities[1,:] - prev_joint_vel) / control_period # 保存当前周期的关节加速度
        prev_joint_vel = velocities[1,:]
        all_positions_velocity.append(prev_joint_vel) # 保存期望关节速度

        # 时间补偿, 保证控制周期为control_period
        elapsed_time = time.time()-st_time # 计算消耗时间=轨迹规划时间+逆运动学求解时间
        
        if elapsed_time > control_period:
            pass
        else:
            time.sleep(control_period - elapsed_time)
        
    logger.info("插接完成, 退出导纳控制周期")
    flag_enter_tcp2canbus = False # 退出 TCP 转 CAN 透传模式

def servo_j(robot):
    '''
    机械臂内环位置控制, 控制周期40ms
    '''
    global flag_enter_tcp2canbus

    traj_data = []; traj_index = 1; traj_len = 0

    logger.info("进入内环位置控制周期，控制周期{}ms".format(move_period*1000))

    while servo_j_running:

        st_time = time.time()

        # 获取轨迹的逻辑：当前轨迹为空 or 当前轨迹已经执行完毕 or 轨迹队列中有新的轨迹
        if len(traj_data) == 0 or traj_index >= traj_len or not trajectory_queue.empty():

            while not trajectory_queue.empty(): # 从共享队列中获取轨迹数据
                traj_data = trajectory_queue.get()
                positions = np.array(traj_data)
                traj_index = 1
                traj_len = positions.shape[0]
        
        if len(traj_data) == 0:  # 如果没有轨迹，就再等一下
            time.sleep(move_period)
            continue

        # 直接can协议透传目标位置
        if flag_enter_tcp2canbus and traj_index < traj_len:

            robot.Movej_CANFD(positions[traj_index,:],False,0) # 透传目标位置给机械臂
            all_positions_desired.append(positions[traj_index,:]) # 保存期望关节角度
            traj_index += 1 # 更新轨迹索引

        else:
            pass # 轨迹已经播放结束，等待下一个周期

        # 时间补偿, 保证内环控制周期为move_period
        elapsed_time = time.time()-st_time
        if elapsed_time > move_period:
            pass
        else:
            time.sleep(move_period - elapsed_time)
    
    logger.info("内环位置控制周期结束")

def get_target_xyz():
    # chakou_position_handle = rospy.ServiceProxy('chakou_predict', ChakouXYZ)
    # pose_out[0,3] = chakou_position_handle().point2[0]
    # pose_out[1,3] = chakou_position_handle().point2[1]
    # pose_out[2,3] = chakou_position_handle().point2[2]
    
    pose_out = np.eye(4)
    pose_out[0,3] = headcam_pose.poses[5].position.x
    pose_out[1,3] = headcam_pose.poses[5].position.y
    pose_out[2,3] = headcam_pose.poses[5].position.z
    logger.info(f"pose_out: {pose_out}")
    ################################################################
    # pose_put = head_cam2base_left @ pose_put
    pose_out = head_cam2base_right @ pose_out
    logger.info(f'base下目标位置: {pose_out[:3,3]}')
    logger.info(f'当前base下目标的旋转矩阵 : {pose_out}')

    return pose_out

def cable_plugging_pre(robot):
    
    def arm_inverse_kine(refer_joints,target_pose):

        r_qd = pykin_ri.inverse_kin([j/180*np.pi for j in refer_joints], target_pose, method="LM", max_iter=100)
        arm_joint = [j/np.pi*180 for j in r_qd]
        logger.info(f'arm_joint: {arm_joint}')
        return arm_joint
    
    # arm_joints_pre = [-42.76, -36.97, -45.06, -42.94, 106.21, -6.46, -76.57]
    # robot.Movej_Cmd(arm_joints_pre,15)
    
    # time.sleep(1)
    
    refer_joints = [47.48, -78.31, -53.92, -35.94, 135.89, 58.54, -47.65]
    # 原版
    put_Tx = np.array( [[ 0.02525999, -0.99955056, -0.0161433,   0.15012921],
                        [-0.4247701,   0.00388647, -0.90529291,  0.05207822],
                        [ 0.90494878,  0.02972488, -0.42448103, -0.03993739],
                        [ 0.,          0.,          0.,          1.        ]])
    
    # 测试机器人站的歪
    # put_Tx = np.array( [[ 0.22038112, -0.97505992, -0.02627397,  0.13184328],
    #                     [-0.33408998, -0.05014861, -0.94120614,  0.05911807],
    #                     [ 0.91641478,  0.21620193, -0.33680955, -0.07466837],
    #                     [ 0.,          0.,          0.,          1.        ]])
    put_pose= get_target_xyz()
    time.sleep(0.03)
    put_target = put_pose @ put_Tx
    arm_joints = arm_inverse_kine(refer_joints,put_target)
    robot.Movej_Cmd(arm_joints,15)
    
if __name__ == '__main__':
    
    logger_init()
    rospy.init_node('force_control_realman')
    rospy.Subscriber("/head_camera_predict_pointXYZ", PoseArray, headcam_pose_callback)
    rospy.Subscriber("/right_camera_predict_pointXYZ", PoseArray, armcam_ri_pose_callback)
    kin_inverse = rospy.ServiceProxy('/kin_inverse', KinInverse)
    # =====================机器人连接=========================
    byteIP_L ='192.168.99.17'
    byteIP_R = '192.168.99.18'
    arm_ri = Arm(75,byteIP_R)
    arm_le = Arm(75,byteIP_L)
    # 设置工具坐标系，同时需要更改pykin和udp上报force_coordinate设置
    # Arm_Tip 对应 Link7, Hand_Frame 对应 tool_hand, robotiq 对应 robotiq
    arm_le.Change_Tool_Frame('Arm_Tip')
    # arm_le.Change_Tool_Frame('Hand_Frame')
    # arm_ri.Change_Tool_Frame('Arm_Tip')
    # arm_ri.Change_Tool_Frame('Hand_Frame')
    arm_ri.Change_Tool_Frame('robotiq')
    arm_le.Change_Work_Frame('Base')
    arm_ri.Change_Work_Frame('Base')
    #关闭左臂udp主动上报, force_coordinate=2代表上报工具坐标系受力数据
    arm_le.Set_Realtime_Push(enable=False)
    arm_ri.Set_Realtime_Push(enable=True, cycle=1, force_coordinate=0)
    arm_status = RealtimePush_Callback(arm_status)
    arm_ri.Realtime_Arm_Joint_State(arm_status)

    err_code, joint_l, pose_l, _, _ = arm_le.Get_Current_Arm_State()
    err_code, joint_r, pose_r, _, _ = arm_ri.Get_Current_Arm_State()
    print('left arm joint:', [round(float(j), 2) for j in joint_l])
    print('right arm joint:', [round(float(j), 2) for j in joint_r])
    print(f'l_ee_position: {[round(float(j),3) for j in pose_l[:3]]}\n l_ee_euler: {[round(float(j),3) for j in pose_l[3:]]}')
    print(f'r_ee_position: {[round(float(j),3) for j in pose_r[:3]]}\n r_ee_euler: {[round(float(j),3) for j in pose_r[3:]]}')
    # ======================================================

    # =========================pykin=========================
    pykin_ri = SingleArm(f_name='./urdf/rm_75b_hand_gripper.urdf')
    # pykin_ri.setup_link_name('base_link', 'Link7')
    # pykin_ri.setup_link_name('base_link', 'tool_hand') 
    pykin_ri.setup_link_name('base_link', 'robotiq')
    pykin_le = SingleArm(f_name='./urdf/rm_75b_hand_gripper.urdf')
    pykin_le.setup_link_name('base_link', 'Link7')
    pykin_fsensor = SingleArm(f_name='./urdf/rm_75b_hand_gripper.urdf')
    pykin_fsensor.setup_link_name('base_link', 'f_sensor') # 力传感器坐标系
    # pykin_le.setup_link_name('base_link', 'tool_hand') 
    # ======================================================

    
    # =====================保存读取数据初始化===================
    start_time = time.time() # 记录程序开始时间
    # 记录整段轨迹的接触力信息在不同参考坐标系下
    force_data_inSensor_list = [] # 传感器坐标系下的接触力数据
    all_contact_force_inTcp = [] # 末端坐标系下的接触力数据
    all_contact_force_inBase = [] # 基座标系下的接触力数据
    joint_data_list = [] # 记录每次读取的关节角度
    pose_data_list = [] # 记录末端实际位姿
    p_end_list = [] # 记录末端期望位姿
    time_list = [] # 记录每次读取的时间
    # ======================================================

    # ===================设置 servo_j 内环位置控制参数===============
    trajectory_queue = queue.Queue() # 初始化轨迹队列
    move_period = 0.05 # 内环控制周期 
    lookahead_time = 0.3  # Lookahead time (s) 前瞻时间，用0.2s规划轨迹
    num_joints = 7  # Number of joints
    cur_joint = np.zeros(num_joints) # 当前关节角 [0, 0, 0, 0, 0, 0]
    cur_joint_vel = np.zeros(num_joints) # 当前关节角速度 [0, 0, 0, 0, 0, 0]
    cur_joint_acc = np.zeros(num_joints) # 当前关节角加速度 [0, 0, 0, 0, 0, 0]
    servo_j_running = True # servo_j 线程运行标志
    flag_enter_tcp2canbus = False # 是否进入 TCP 转 CAN 透传模式
    # 记录整段轨迹的期望关节角度和实际关节角度    
    all_positions_desired = []
    all_positions_real = []
    all_positions_velocity = []
    # ======================================================

    # ========================读取外部力数据====================
    sensor = ForceSensorRS485(port='/dev/ttyUSB1') # 创建力传感器对象
    read_forcedata_thread = threading.Thread(target=read_forcedata, args=(sensor,))
    read_forcedata_thread.start()
    # ======================================================

    # ========================校准外部力数据====================
    calib_force_data_thread = threading.Thread(target=calib_force_data_func,args=(arm_ri,))
    calib_force_data_thread.start()
    time.sleep(0.5) # 等待线程启动
    logger.info("校准外部力数据线程启动")
    # ======================================================

    # ===============rostopic pub末端位姿和力传感器数据=======
    pub_pose_force_running = True
    publisher_thread = threading.Thread(target=pose_force_publisher)
    publisher_thread.start()
    # ======================================================

    try:

        # ==========================主程序=========================
        servo_j_thread = threading.Thread(target=servo_j,args=(arm_ri,)) # 内环位置控制线程
        servo_j_thread.start()

        logger.info("等待机械臂移动到预定位置")
        # input("enter to continue")
        cable_plugging_pre(arm_ri)
        # arm_ri.Movej_Cmd([34.69, -94.58, -26.88, -15.03, -71.97, -65.1, 137.45],10)
        time.sleep(2) # 等待机械臂移动到预定位置

        force_data_zero() # 清零力传感器数据
        admittance_controller(arm_ri) # 导纳控制器

        # while True:
        #     time.sleep(0.1) # 主线程等待，保持程序运行

                
        # ========================================================
        
    except KeyboardInterrupt: # CTRL+C退出程序

        logger.info("程序被中断")

    except rospy.ROSInterruptException as e: # ROS异常

        logger.info("程序异常退出: ", e)

    finally:
        
        # ==========================清理工作=======================
        # 结束内环servoj线程
        servo_j_running = False
        servo_j_thread.join()

        # 结束发布位姿和力传感数据的线程
        pub_pose_force_running = False
        publisher_thread.join()

        # 结束读取力传感数据和机器人状态的线程
        read_force_data_running = False
        calib_force_data_thread.join()

        # 断开连接力传感器
        sensor.stop_stream()
        sensor.ser.close()

        logger.info("程序结束, 关闭socket连接和机器人连接")

        logger.info("程序结束, 保存数据")
        # ========================================================
        
        
        # ==========================数据保存=======================
        # 保存末端期望位姿
        p_end_list_converted = [data.tolist() for data in p_end_list]
        with open('./saved_data/end_pose_data_desired.json', 'w') as f:
            json.dump(p_end_list_converted, f)

        # 保存末端实际位姿
        pose_data_list_converted = [data.tolist() for data in pose_data_list]
        with open('./saved_data/end_pose_data_real.json', 'w') as f:
            json.dump(pose_data_list_converted, f)

        # 保存传感器坐标系下力数据
        force_data_inSensor_list_converted = [data.tolist() for data in force_data_inSensor_list]
        with open('./saved_data/force_data_inSensor.json', 'w') as f:
            json.dump(force_data_inSensor_list_converted, f)

        # 保存末端坐标系下力数据
        all_contact_force_inTcp_converted = [data.tolist() for data in all_contact_force_inTcp]
        with open('./saved_data/force_data_inTcp.json', 'w') as f:
            json.dump(all_contact_force_inTcp_converted, f)

        # 保存基坐标系下力数据
        all_contact_force_inBase_converted = [data.tolist() for data in all_contact_force_inBase]
        with open('./saved_data/force_data_inBase.json', 'w') as f:
            json.dump(all_contact_force_inBase_converted, f)

        # 保存期望关节角度
        all_positions_converted_desired = [pos.tolist() for pos in all_positions_desired]
        with open('./saved_data/trajectory_data.json', 'w') as f:
            json.dump(all_positions_converted_desired, f)
        
        # 保存实际关节角度
        all_positions_real_converted = [pos.tolist() for pos in all_positions_real]
        with open('./saved_data/trajectory_data_real.json', 'w') as f:
            json.dump(all_positions_real_converted, f)
        
        # 保存实际关节速度
        all_positions_velocity_converted = [pos.tolist() for pos in all_positions_velocity]
        with open('./saved_data/trajectory_velocity_real.json', 'w') as f:
            json.dump(all_positions_velocity_converted, f)
        # =========================================================

