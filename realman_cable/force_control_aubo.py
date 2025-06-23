import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import socket
import threading
import json
from lib.robotcontrol import Auboi5Robot, RobotError, RobotErrorType, RobotEventType, RobotEvent, RobotMoveTrackType, RobotCoordType, RobotIOType, RobotUserIoName
from pykin.robots.single_arm import SingleArm
from pykin.utils import transform_utils as transform_utils
from collections import deque
from mpl_toolkits.mplot3d import Axes3D
from admittance_controller import AdmittanceController
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline
import rospy
from std_msgs.msg import Float64MultiArray
import queue
import concurrent.futures
from spatialmath import SE3
import logging
from logging.handlers import RotatingFileHandler


# 创建一个logger
logger = logging.getLogger('calibration_forceSensor')

# 清除上一次的logger
if logger.hasHandlers():
    logger.handlers.clear()

def logger_init():
    
    logger.setLevel(logging.INFO) # Log等级总开关
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


# 定义报文头和尾
PACK_BEGIN = "<PACK_BEGIN"
PACK_END = "PACK_END>"
force_data = np.zeros(6) # fx fy fz Mx My Mz 原始力传感器数据
calib_force_data = np.zeros(6) # 标定后的力数据
force_zero_point = np.zeros(6) # 力传感器数据清零
# 线程池
executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

def process_data(packet):
    """
    解析和处理接收到的单个数据包
    """
    global force_data
    try:
        # 从数据包中解析 JSON 数据
        packet_content = packet[len(PACK_BEGIN): -len(PACK_END)]
        length_str = packet_content[:8].strip()
        length = int(length_str)
        json_data = packet_content[8:8 + length]
        json_obj = json.loads(json_data)
        
        # 更新全局力传感器数据
        force_data = json_obj["force_data"]
        # logger.info(f"解析成功: {force_data[1]}")
    
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"数据解析失败: {e}")

def read_force_data(sock):
    """
    从 socket 读取数据
    """
    buffer = b""
    global read_force_data_running

    while read_force_data_running:
        try:
            data = sock.recv(4096)  # 接收数据
            if not data:
                break
            
            buffer += data
            # 检查数据包是否完整
            begin_pos = buffer.find(PACK_BEGIN.encode())
            end_pos = buffer.find(PACK_END.encode())
            
            while begin_pos >= 0 and end_pos >= 0 and end_pos > begin_pos + len(PACK_BEGIN):
                # 提取完整数据包
                packet = buffer[begin_pos:end_pos + len(PACK_END)]
                # 提交数据包给线程池进行解析和处理
                executor.submit(process_data, packet.decode())
                
                # 移动缓冲区指针
                buffer = buffer[end_pos + len(PACK_END):]
                begin_pos = buffer.find(PACK_BEGIN.encode())
                end_pos = buffer.find(PACK_END.encode())
        
        except Exception as e:
            logger.error(f"读取数据时发生错误: {e}")
            break
    

def calib_force_data_func(robot):
    '''
    校准力传感器数据, 输出负载辨识和标定后的纯外力数据
    '''

    global calib_force_data,read_force_data_running, force_data, force_zero_point

    # 读取 calibration_result.JSON 文件得到标定结果
    file_name = './calibrate_forceSensor/calibration_result.json'
    with open(file_name, 'r') as json_file:
        json_data = json.load(json_file)
        gravity_bias = np.array(json_data['gravity_bias'])
        mass = np.array(json_data['mass'][0])
        force_zero = np.array(json_data['force_zero'])

    # 初始化一个队列用于存储前5次的力传感器数据
    length_force_data_queue = 20
    force_data_queue = deque(maxlen=length_force_data_queue)

    # 定义滤波器参数
    cutoff = 3  # 截止频率 (Hz)
    order = 2    # 滤波器阶数
    fs = 200  # 根据实际情况设置采样频率 (Hz)
    b, a = butter(order, cutoff / (0.5 * fs), btype='low', analog=False) # 设计低通Butterworth滤波器

    alpha = 0.1 # 滑动窗口加权滤波
    smoothed_force_data = np.zeros(6) # 平滑后的力传感器数据
    time_count = 0

    start_time = time.time()
    
    while read_force_data_running:
        
        if force_data is not None:
            
            raw_force_data = np.array(force_data).copy()

            # 将当前的力传感器数据添加到队列中
            force_data_queue.append(raw_force_data)
            # 如果队列中的数据少于5次，则继续等待
            if len(force_data_queue) < length_force_data_queue:
                continue
            # 将 force_data_queue 转换为 NumPy 数组
            force_data_array = np.array(force_data_queue)
            # 应用滤波器
            filtered_signal = filtfilt(b, a, force_data_array, axis=0)
            new_force_data = filtered_signal[-1]

            smoothed_force_data = alpha * new_force_data  + (1-alpha) * smoothed_force_data

            '''
            获取机械臂当前的末端姿态, 需要根据不同的机械臂型号和通信协议获取
            '''
            current_waypoint = robot.get_current_waypoint()
            # 当前末端位置 in m
            ee_pos = [np.round(i,6) for i in current_waypoint['pos']] 
            # 当前末端姿态
            ee_ori_rpy_rad = np.round(np.array(robot.quaternion_to_rpy(current_waypoint['ori'])),6) # in rad
            ee_ori_rpy_deg = np.round(ee_ori_rpy_rad/np.pi*180,2) # in degree
            pose_data = np.concatenate((ee_pos, ee_ori_rpy_deg), axis=0) # 6*1

            '''
            计算外部力数据 = 原始力数据 - 力传感器零点 - 负载重力分量
            '''
            rotation_matrix = np.array(transform_utils.get_matrix_from_quaternion(current_waypoint['ori']))
            inv_rotation_matrix = np.linalg.inv(rotation_matrix) # 3*3
            temp_gravity_vector = np.array([0, 0, -1]).reshape(3, 1) # 3*1
            gravity_vector = np.dot(inv_rotation_matrix, temp_gravity_vector) # 3*1
            G_force[:3] = np.transpose(mass * gravity_vector) # 3*1 Gx Gy Gz
            G_force[3] = G_force[2] * gravity_bias[1] - G_force[1] * gravity_bias[2] # Mgx = Gz × y − Gy × z 
            G_force[4] = G_force[0] * gravity_bias[2] - G_force[2] * gravity_bias[0] # Mgy = Gx × z − Gz × x
            G_force[5] = G_force[1] * gravity_bias[0] - G_force[0] * gravity_bias[1] # Mgz = Gy × x − Gx × y

            # 标定后的力数据 = 原始力传感器数据 - 力传感器零点 - 负载重力分量 - 手动清零分量
            calib_force_data = smoothed_force_data - force_zero - G_force - force_zero_point # 6*1
            
            # 保存数据
            force_data_inSensor_list.append(calib_force_data) # 末端受到的外部力
            time_list.append(time.time() - start_time) # 时间

            time.sleep(0.005) # 5ms/次
        
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

def robot_inverse_kinematic(target_pos,target_ori_rpy_rad):
    '''
    逆运动学求解关节角驱动机械臂运动
    :param target_pos: 末端期望位置 / m
    :param target_ori_rpy_rad 末端期望角度rqy / degree
    :return: 逆运动学求解的关节角度 / rad
    '''
    global robot
    current_waypoint = robot.get_current_waypoint()
    current_joint_rad = np.array(current_waypoint['joint']) # in rad
    target_rot = transform_utils.get_matrix_from_rpy(target_ori_rpy_rad)
    target_pose = np.eye(4);target_pose[:3, :3] = target_rot; target_pose[:3, 3] = target_pos
    r_qd = pykin.inverse_kin(current_joint_rad, target_pose, method="LM", max_iter=20)

    # 轴动到初始位置
    desired_joint = tuple(r_qd)
    return desired_joint


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

        # 判断是否满足最大速度和最大加速度的约束 或者超过了一定时间限制
        if (max_v <= max_velocity and max_a <= max_acceleration) or T >= 1.0:
            # print(f"planning time: {T}")
            break  # 如果满足约束，停止调整
        
        T +=0.02  # 增大轨迹时间，降低速度和加速度
    
    return t_vals, positions, velocities, accelerations, T  # 返回时间戳



def trapezoidal_velocity_corrected(theta_start, theta_end, acc_time, dcc_time, control_period, joint_vel_limit,prev_vel, prev_acc, max_delta_vel):
    """
    梯形速度规划，带有最大角速度和最大角加速度约束，支持多个关节
    """
    num_joints = len(theta_start)
    theta_total = np.array(theta_end) - np.array(theta_start)
    direction_vel = np.sign(theta_total)
    
    lookahead_time = acc_time + dcc_time + np.max(np.abs(theta_total) / joint_vel_limit)  # 确保lookahead_time为标量
    t2 = lookahead_time - (acc_time + dcc_time)
    
    # 计算最大速度（受限于最大角速度）
    vel_max = np.abs(theta_total) / ((acc_time / 2) + max(t2, 0) + (dcc_time / 2))
    vel_max = np.minimum(vel_max, joint_vel_limit) * direction_vel
    
    # 计算最大加速度基于前后速度，并考虑方向，取绝对值后比较
    vel_total = (np.array(vel_max) - np.array(prev_vel)) / acc_time
    direction_acc = np.sign(vel_total)
    acc_max = np.minimum(np.abs(vel_total), max_delta_vel) * direction_acc
    
    # 平滑加速度
    alpha = 0.1  # 指数平滑因子（调整以减少振荡）
    if prev_acc is not None:
        acc_max = alpha * acc_max + (1 - alpha) * prev_acc
    
    # 计算平滑后的速度
    vel_max = prev_vel + acc_max * acc_time
    
    # 生成轨迹时间序列
    t_vals = np.arange(0, lookahead_time + control_period, control_period)
    vel_vals = np.zeros((num_joints, len(t_vals)))
    theta_vals = np.zeros((num_joints, len(t_vals)))
    
    for j in range(num_joints):
        theta = theta_start[j]
        for i, t in enumerate(t_vals):
            if t < acc_time:
                vel = prev_vel[j] + acc_max[j] * t
            elif t < acc_time + t2:
                vel = vel_max[j]
            else:
                vel = vel_max[j] - acc_max[j] * (t - acc_time - t2)
            
            if i > 0:
                theta += (vel + vel_vals[j, i-1]) / 2 * control_period
            theta_vals[j, i] = theta
            vel_vals[j, i] = vel
    
    return t_vals, theta_vals, vel_vals

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

def admittance_controller():
    '''
    阻抗控制器, 用于机械臂外环控制, 控制周期25hz
    '''

    global flag_enter_tco2canbus, calib_force_data

    # 设置admittance control阻抗控制器参数
    mass = [10.0, 10.0, 10.0, 2.0, 2.0, 2.0]  # 定义惯性参数
    damping = [30.0, 30.0, 30.0, 10.0, 10.0, 10.0]  # 定义阻尼参数
    stiffness = [10.0, 10.0, 10.0, 5.0, 5.0, 5.0]  # 定义刚度参数
    control_period = 0.050 # 阻抗控制周期（单位：秒）要跟状态反馈时间保持一致，用于外环阻抗控制循环

    # 初始化导纳参数
    controller = AdmittanceController(mass, stiffness, damping, control_period)

    # 期望末端位置和速度
    des_eef_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    des_eef_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # 预设初始关节角度
    desired_joint = [1.3183549782708925, 0.2844580396074309, -1.9350581848325896, 0.9220872927851061, 1.318360580565033, 4.956739951040262] # in rad
    robot.move_joint(tuple(desired_joint))
    time.sleep(2)

    # 获取当前末端位姿
    current_waypoint = robot.get_current_waypoint()
    ee_pos = np.array(current_waypoint['pos']) # 当前末端位置 in m
    ee_ori_rpy_rad = np.array(robot.quaternion_to_rpy(current_waypoint['ori'])) # # 当前末端姿态 in rad
    flange_pose = SE3.Trans(ee_pos[0], ee_pos[1], ee_pos[2]) * SE3.RPY(ee_ori_rpy_rad[0], ee_ori_rpy_rad[1], ee_ori_rpy_rad[2])
    trans_inTcpFrame = SE3.Trans(0.0,0.0,0.0) * SE3.RPY(0.0/180*np.pi,0.0/180*np.pi,0.0/180*np.pi) # 末端坐标系到力传感器坐标系的转换矩阵
    tcp_pose = flange_pose * trans_flange2tcp * trans_inTcpFrame
    tcp_pos = np.array(tcp_pose.t); tcp_ori = np.array(tcp_pose.rpy())

    # 起点位置和姿态设置
    p_start_pos = tcp_pos.copy(); p_start_ori = tcp_ori.copy()
    p_start_pos[0] -= 0.004 # 小小修正一下
    p_start_pos[1] -= 0.002 # 小小修正一下
    p_start_pos[2] -= 0.06 # 小小修正一下
    desired_joint = robot_inverse_kinematic(p_start_pos,p_start_ori) # 末端期望位置 / m， 角度rqy / rad
    logger.info(f"起点关节角度：{desired_joint}")
    robot.move_joint(desired_joint)
    time.sleep(1)

    # 路径规划
    num_points = 1000  # 轨迹上的点数

    # trans_length = 0.02  # 平移距离
    # p_start = tcp_pos.copy() # 起点位置
    # delta_steps = np.linspace(0, trans_length, num_points) 

    # # 规划每一个目标点的位姿
    # for delta_step in delta_steps:

    #     x = p_start[0] 
    #     y = p_start[1]
    #     z = p_start[2] # + delta_step
    #     rx = tcp_ori[0]
    #     ry = tcp_ori[1]
    #     rz = tcp_ori[2]
    #     p_end = np.array([x, y, z, rx, ry, rz])
    #     p_end_list.append(p_end)

    # p_end_array = np.array(p_end_list)
    # num_points = p_end_array.shape[0] # 轨迹上点的数量

    # 梯形速度轨迹规划参数
    # acc_time = 1 * control_period  # 加速时间 (s)
    # dcc_time = 1 * control_period  # 减速时间 (s)
    # joint_vel_limit = np.ones(6) * 0.03  # 每个关节的最大角速度 (rad/s)
    # joint_acc_limit = np.ones(6) * 0.04  # 计算最大角加速度

    # 5次多项式轨迹规划
    joint_vel_limit = 0.02  # 每个关节的最大角速度 (rad/s) 0.04
    joint_acc_limit = 0.03  # 计算最大角加速度 0.08

    prev_joint_vel = np.zeros(6) # 保存上一周期周期的关节速度
    prev_joint_acc = np.zeros(6) # 保存上一周期周期的关节加速度

    # 进入 TCP 转 CAN 透传模式
    result = robot.enter_tcp2canbus_mode()
    if result != RobotErrorType.RobotError_SUCC:
        logger.info("TCP 转 CAN 透传模式失败, 错误码：{}".format(result))
    else:
        logger.info("TCP 转 CAN 透传模式成功")
        flag_enter_tco2canbus = True

    # 力传感器数据清零
    force_data_zero()

    logger.info("进入外环导纳控制周期，控制周期{}ms".format(control_period*1000))

    user_force_inTcp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # 用户主动施加力，末端坐标系下
    user_force_inBase = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # 用户主动施加力，基坐标系下
    
    # 整个控制周期的不同阶段标志位
    first_phase = True
    second_phase = False
    third_phase = False
    completed_flag = False # 完成标志位
    adjust_flag = False # 调整位置标志位

    max_fxy_inTcp = 9 # 最大碰撞力
    max_ty_inTcp = 0.5 # 最大碰撞扭矩
    threshold_fxy_inTcp = 9 # 碰撞力阈值
    threshold_ty_inTcp = 0.5 # 碰撞力阈值

    for i in range(num_points):

        # if i <= 5:
        #     force_data_zero()
        #     continue # 跳过前5个点

        st_time = time.time()
        logger.info("第{}个".format(i))

        p_end_pos = p_start_pos.copy(); p_end_ori = p_start_ori.copy()
        
        
        # 获取当前关节状态和末端状态
        current_waypoint = robot.get_current_waypoint()
        cur_joint = np.array(current_waypoint['joint']) # in rad 
        ee_pos = np.array(current_waypoint['pos']) # 当前末端位置 in m
        ee_ori_rpy_rad = np.array(robot.quaternion_to_rpy(current_waypoint['ori'])) # # 当前末端姿态 in rad
        flange_pose = SE3.Trans(ee_pos[0], ee_pos[1], ee_pos[2]) * SE3.RPY(ee_ori_rpy_rad[0], ee_ori_rpy_rad[1], ee_ori_rpy_rad[2])
        tcp_pose = flange_pose * trans_flange2tcp
        tcp_pos = np.array(tcp_pose.t); tcp_ori = np.array(tcp_pose.rpy())
        controller.eef_pose = np.concatenate((tcp_pos, tcp_ori), axis=0) # 6*1

        # 存储当前关节角
        all_positions_real.append(cur_joint) # in rad

        # 存储末端当前位姿
        pose_data_list.append(controller.eef_pose) # 6*1

        '''
        插接动作规划，分为了三个阶段：
        第一阶段: 向z+(Base)移动, 施加主动力z+(Base), 期望位置沿z+(Base), 期望姿态保持不变，直到调整结束 && 碰到孔，转到第二阶段
        第二阶段: 施加主动力x+(Tcp), 期望位置x+ y+ z不变(Tcp), 期望姿态保持不变, 直到三点稳定接触[Fy(Tcp)>3], 转到第三阶段
        第三阶段: 主动偏转, 施加主动力z+(Base), 期望位置保持不变, 期望姿态主动偏转rz+(Tcp), 直到贴住孔壁[Fy(Tcp)>3], 转到第三阶段
        第四阶段: 施加主动插接力x+(Tcp), 期望位置x+(Tcp), 期望姿态保持不变, 检测接触力, 直到接触力[Fx(Tcp)>5], 结束插接
        '''
        # 300*0.04=12s
        if first_phase: # 第一阶段

            controller.M = np.diag([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])  # 惯性矩阵
            controller.B = np.diag([30.0, 30.0, 30.0, 20.0, 20.0, 20.0])  # 阻尼矩阵
            controller.K = np.diag([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])  # 刚度矩阵
            user_force_inBase = np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0])
            p_end_pos[2] += 0.00005

        elif second_phase: # 第二阶段

            if adjust_flag: # 二阶段开头需要调整

                controller.M = np.diag([30.0, 30.0, 30.0, 10.0, 10.0, 10.0])  # 惯性矩阵
                controller.B = np.diag([50.0, 50.0, 50.0, 20.0, 20.0, 20.0])  # 阻尼矩阵
                controller.K = np.diag([5.0, 5.0, 5.0, 2.0, 2.0, 2.0])  # 刚度矩阵
      
            else:  

                user_force_inTcp = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # 用户主动施加力, 末端坐标系下

                trans_inTcpFrame = SE3.Trans(0.001,0.0005,0.0) * SE3.RPY(0.0,0.0,0.0)
                target_pose = tcp_pose * trans_inTcpFrame
                p_end_pos = target_pose.t
                p_end_ori = target_pose.rpy()
            
        elif third_phase: # 第三阶段

            controller.M = np.diag([40.0, 40.0, 40.0, 10.0, 10.0, 10.0])  # 惯性矩阵
            controller.B = np.diag([60.0, 60.0, 60.0, 20.0, 20.0, 20.0])  # 阻尼矩阵
            controller.K = np.diag([30.0, 30.0, 30.0, 10.0, 10.0, 10.0])  # 刚度矩阵

            user_force_inTcp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # 用户主动施加力, 末端坐标系下

            if adjust_flag:
                
                trans_inTcpFrame = SE3.Trans(-0.002,0.0,0.0) * SE3.RPY(0.0,0.0,0.0)
                target_pose = tcp_pose * trans_inTcpFrame
                p_end_pos = target_pose.t
                p_end_ori = target_pose.rpy()

                if i - second_phase_end > 200: # 如果调整时间结束，开始第三阶段主动偏转

                    adjust_flag = False
                    third_phase_start = i # 记录下第三阶段开始时的时间
            
            else:
                
                '''
                这里需要做一个绕指尖的旋转
                '''
                trans_inTipFrame = SE3.Trans(0.0,0.0,0.0) * SE3.RPY(0.0,0.0,3/180*np.pi)
                trans_Tip2Tcp = SE3.Trans(0.05,0.005,0.0) * SE3.RPY(0.0,0.0,0.0)
                trans_inTcpFrame = SE3.Trans(-0.001,-0.002,0.0) * SE3.RPY(0.0,0.0,0.0)
                target_pose = tcp_pose * trans_inTcpFrame * trans_Tip2Tcp * trans_inTipFrame
                p_end_pos = target_pose.t
                p_end_ori = target_pose.rpy()

        # 根据规划的路径获取旧的末端期望位姿
        des_eef_pose = np.concatenate((p_end_pos, p_end_ori), axis=0) # 6*1

        # 存储末端期望位姿
        p_end_list.append(des_eef_pose) 

        # 将传感器坐标系下的力转换到末端坐标系下，注意此时受力坐标系为末端坐标系
        force2tcp = SE3.Trans(0.0, 0.0, -0.196) * SE3.RPY(0, 0, 0) # 力传感器坐标系到末端坐标系的转换矩阵
        contact_force_data_inTcp = trans_force_data(force2tcp.A, calib_force_data) # 6*1
        contact_force_data_inTcp = contact_force_data_inTcp + user_force_inTcp # 加上用户期望受力，末端坐标系下
        all_contact_force_inTcp.append(contact_force_data_inTcp) # 保存末端坐标系下的受力

        if third_phase: # 第三阶段
            contact_force_data_inTcp[0] = contact_force_data_inTcp[0] / 10 # 降低x轴受力
            contact_force_data_inTcp[1] = contact_force_data_inTcp[1] / 10 # 降低y轴受力
            contact_force_data_inTcp[2] = contact_force_data_inTcp[2] / 10 # 降低z轴受力
            contact_force_data_inTcp[3] = contact_force_data_inTcp[3] / 10 # 降低Tx轴受力
            contact_force_data_inTcp[4] = contact_force_data_inTcp[4] / 10 # 降低Ty轴受力
            contact_force_data_inTcp[5] = contact_force_data_inTcp[5] / 10 # 降低Tz轴受力

        # 将末端坐标系下的力转换到基坐标系下
        contact_force_data_inBase = trans_force_data(tcp_pose.A, contact_force_data_inTcp) # 6*1
        contact_force_data = contact_force_data_inBase + user_force_inBase # 加上用户期望受力，基坐标系下
        all_contact_force_inBase.append(contact_force_data) # 保存基坐标系下的受力

        # 通过导纳控制器计算新的末端期望位姿
        updated_eef_pose = controller.update(des_eef_pose, des_eef_vel, contact_force_data)
        # updated_eef_pose = des_eef_pose.copy()

        if first_phase: # 第一阶段

            p_slover_pos = updated_eef_pose[:3] # 位置放松
            p_slover_ori = p_start_ori.copy() # 姿态锁住不变=初始姿态
            sensor_force = np.sqrt(np.square(contact_force_data_inTcp[0]) + np.square(contact_force_data_inTcp[1]))  # 检测碰撞力
            sensor_torque = np.abs(contact_force_data_inTcp[4]) # 检测碰撞扭矩

            if i > 5 and not adjust_flag and sensor_force > 0.5 and sensor_torque > 0.02: # 如果碰撞力Fx + Fy > 0.5 && Ty > 0.02，说明接触到了孔，在调整

                adjust_flag = True
                max_fxy_inTcp = 0 # 重置最大碰撞力
                max_ty_inTcp = 0 # 重置最大碰撞扭矩

            elif adjust_flag:
                if sensor_force > max_fxy_inTcp:
                    max_fxy_inTcp = sensor_force # 更新得到最大碰撞力
                if sensor_torque > max_ty_inTcp:
                    max_ty_inTcp = sensor_torque # 更新得到最大碰撞扭矩
                if sensor_force <= 1 and sensor_torque <= 0.05: # 如果碰撞力Fx + Fy <= 0.5 && Ty <= 0.02，说明调整完成
                    adjust_flag = False
                    # 如果调整过程中最大碰撞力和扭矩都小于阈值，说明调整完成，第一阶段结束
                    if max_fxy_inTcp < threshold_fxy_inTcp and max_ty_inTcp < threshold_ty_inTcp: # 
                        first_phase = False
                        second_phase = True
                        first_phase_end = i # 记录下第一阶段结束时的时间
                        adjust_flag = True # 二阶段开头需要调整
                        user_force_inBase = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # 用户主动施加力, 基坐标系下 清零

            logger.info("第一阶段") # 打印阶段信息
            logger.info("{}".format(adjust_flag))

        elif second_phase: # 第二阶段

            if adjust_flag: # 二阶段开头需要调整

                p_slover_pos = updated_eef_pose[:3] # x y z位置放松
                p_slover_ori = p_start_ori.copy() # 姿态锁住不变=初始姿态
                if np.linalg.norm(contact_force_data_inTcp[:3]) < 2 and np.linalg.norm(contact_force_data_inTcp[3:]) < 0.1:
                    temp_pos = p_slover_pos.copy() # 记录下稳定的位置
                    second_phase_start = i # 记录下第二阶段开始时的时间
                    adjust_flag = False # 二阶段开头调整结束

                logger.info("一二阶段过渡, 等待二阶段开始") # 打印阶段信息

            else: # 二阶段开始

                p_slover_pos = updated_eef_pose[:3] # x z位置放松
                p_slover_pos[1] = temp_pos[1] # y轴位置锁住
                p_slover_ori = p_end_ori.copy() # 姿态锁住偏转rz+(Tcp)
                sensor_force = contact_force_data_inTcp[0]  # 检测 inTcp x轴负方向接触力


                if sensor_force < -4: # 如果接触力大于阈值，切换到第三阶段

                    second_phase = False # 二阶段结束
                    second_phase_end = i # 记录下第二阶段结束时的时间

                    third_phase = True # 三阶段开始
                    adjust_flag = True # 三阶段开头需要调整

                logger.info("第二阶段") # 打印阶段信息
                logger.info("{}".format(adjust_flag))

        elif third_phase: # 第三阶段检测插接力

            p_slover_pos = updated_eef_pose[:3] # x z 位置放松
            p_slover_pos[1] = temp_pos[1] # y轴位置锁住 
            p_slover_ori = p_end_ori.copy() # 姿态锁住偏转rz+(Tcp)
            sensor_force = np.abs(contact_force_data_inTcp[0]) # 检测插接力

            if not adjust_flag and i - third_phase_start > 300: # 如果主动偏转时间结束，开始插接

                third_phase = False
                completed_flag = True
                third_phase_end = i # 记录下第三阶段结束时的时间

            logger.info("第三阶段") # 打印阶段信息
            logger.info("{}".format(adjust_flag))

        logger.info("sensor_force: {}".format(sensor_force))  
        logger.info("sensor_torque: {}".format(sensor_torque))

        # 设置期望位置、速度和加速度
        q_target = robot_inverse_kinematic(p_slover_pos,p_slover_ori)
        v_target = np.zeros(6)
        a_target = np.zeros(6)

        # # 梯形速度规划生成平滑轨迹
        # _, positions, velocities = trapezoidal_velocity_corrected(cur_joint, q_target, acc_time, dcc_time, control_period, joint_vel_limit, prev_joint_vel, prev_joint_acc, joint_acc_limit)

        # 5次多项式轨迹规划生成平滑轨迹
        _, positions, velocities, _, _ = generate_trajectory_with_constraints(
            cur_joint, prev_joint_vel, prev_joint_acc, q_target, v_target, a_target, control_period, lookahead_time, joint_vel_limit, joint_acc_limit)

        # 将规划好的轨迹放进共享队列
        trajectory_queue.put(positions)
        
        # # 梯形速度规划
        # prev_joint_acc = (velocities[:,1] - prev_joint_vel) / control_period # 保存当前周期的关节加速度
        # prev_joint_vel = velocities[:,1]

        # 五次多项式轨迹规划
        prev_joint_acc = (velocities[1,:] - prev_joint_vel) / control_period # 保存当前周期的关节加速度
        prev_joint_vel = velocities[1,:]

        all_positions_velocity.append(prev_joint_vel) # 保存期望关节速度

        if completed_flag:
            logger.info("插接完成, 退出导纳控制周期")
            break

        # 时间补偿, 保证控制周期为control_period
        elapsed_time = time.time()-st_time # 计算消耗时间=轨迹规划时间+逆运动学求解时间
        if elapsed_time > control_period:
            pass
        else:
            time.sleep(control_period - elapsed_time)

    # 退出 TCP 转 CAN 透传模式
    flag_enter_tco2canbus = False
    robot.leave_tcp2canbus_mode()
    logger.info("结束外环控制周期，退出 TCP 转 CAN 透传模式")


def servo_j():
    '''
    机械臂内环位置控制, 控制周期40ms
    '''

    global flag_enter_tco2canbus

    traj_data = []
    traj_index = 1
    traj_len = 0

    logger.info("进入内环位置控制周期，控制周期{}ms".format(move_period*1000))

    while servo_j_running:

        st_time = time.time()

        # 获取轨迹的逻辑：当前轨迹为空 or 当前轨迹已经执行完毕 or 轨迹队列中有新的轨迹
        if len(traj_data) == 0 or traj_index >= traj_len or not trajectory_queue.empty():

            # 从共享队列中获取轨迹数据
            while not trajectory_queue.empty():
                traj_data = trajectory_queue.get()
                positions = np.array(traj_data)
                traj_index = 1
                traj_len = positions.shape[0]
        
        # 如果没有轨迹，就再等一下
        if len(traj_data) == 0:
            time.sleep(move_period)
            continue

        # 直接can协议透传目标位置
        if flag_enter_tco2canbus and traj_index < traj_len:
            
            # # 梯形速度轨迹
            # robot.set_waypoint_to_canbus(tuple(positions[:,traj_index])) # 透传目标位置给机械臂
            # all_positions_desired.append(positions[:,traj_index]) # 保存期望关节角度

            # 5次多项式轨迹
            robot.set_waypoint_to_canbus(tuple(positions[traj_index,:])) # 透传目标位置给机械臂
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
    
if __name__ == '__main__':
 
    logger_init()


    # ==================socket连接力传感器====================
    server_address = ('192.168.26.103', 8896)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(server_address)
    logger.info("连接力传感器成功")
    read_force_data_running = True # 读取数据线程运行标志
    # ======================================================
    

    # =====================机器人连接=========================
    Auboi5Robot.initialize() 
    robot = Auboi5Robot()
    handle = robot.create_context() # 创建上下文
    result = robot.connect('192.168.26.103', 8899)
    robot.set_collision_class(6) # 设置碰撞等级
    robot.init_profile() # 初始化全局配置文件 自动清理掉之前设置的用户坐标系，速度，加速度等属性
    robot.set_joint_maxacc((0.8, 0.8, 0.8, 0.8, 0.8, 0.8)) # 设置关节最大加速度 rad/s^2
    robot.set_joint_maxvelc((0.8, 0.8, 0.8, 0.8, 0.8, 0.8)) # 设置关节最大速度 rad/s
    line_maxacc = 0.6 # 设置末端运动最大线加速度 m/s^2
    robot.set_end_max_line_acc(line_maxacc)
    line_maxvelc = 0.6 # 设置末端运动最大线速度 m/s
    robot.set_end_max_line_velc(line_maxvelc)
    trans_flange2tcp = SE3.Trans(0, 0, 0.211) # flange to tcp
    # ======================================================


    # =========================pykin=========================
    pykin = SingleArm(f_name="./description/aubo_i10.urdf")
    pykin.setup_link_name(base_name="base_link", eef_name="Tool_Link")
    # ======================================================

    
    # =====================保存读取数据初始化===================
    start_time = time.time() # 记录程序开始时间
    force_data_inSensor_list = [] # 记录每次读取的力数据
    joint_data_list = [] # 记录每次读取的关节角度
    pose_data_list = [] # 记录末端实际位姿
    p_end_list = [] # 记录末端期望位姿
    time_list = [] # 记录每次读取的时间
    read_force_data_running = True # 读取数据线程运行标志
    G_force = np.zeros(6) # Gx Gy Gz Mgx Mgy Mgz 负载重力在力传感器坐标系下的分量
    first_phase_end = 0 # 第一阶段结束时间
    second_phase_start = 0 # 第二阶段开始时间
    second_phase_end = 0 # 第二阶段结束时间
    third_phase_start = 0 # 第三阶段开始时间
    third_phase_end = 0 # 第三阶段结束时间
    
    # ======================================================


    # ========================读取外部力数据====================
    # 单开一个线程读取力传感器数据
    read_force_data_thread = threading.Thread(target=read_force_data, args=(sock,))
    read_force_data_thread.start()
    time.sleep(1) # 等待线程启动
    logger.info("读取原始外部力数据线程启动")

    calib_force_data_thread = threading.Thread(target=calib_force_data_func,args=(robot,))
    calib_force_data_thread.start()
    time.sleep(0.5) # 等待线程启动
    logger.info("读取校准外部力数据线程启动")
    # ======================================================


    # ===================设置 servo_j 内环位置控制参数===============
    trajectory_queue = queue.Queue() # 初始化轨迹队列
    move_period = 0.040 # 内环控制周期 
    lookahead_time = 0.2  # Lookahead time (s) 前瞻时间，用0.2s规划轨迹
    num_joints = 6  # Number of joints
    cur_joint = np.zeros(num_joints) # 当前关节角 [0, 0, 0, 0, 0, 0]
    cur_joint_vel = np.zeros(num_joints) # 当前关节角速度 [0, 0, 0, 0, 0, 0]
    cur_joint_acc = np.zeros(num_joints) # 当前关节角加速度 [0, 0, 0, 0, 0, 0]
    servo_j_running = True # servo_j 线程运行标志
    flag_enter_tco2canbus = False # 是否进入 TCP 转 CAN 透传模式
    # 记录整段轨迹的期望关节角度和实际关节角度    
    all_positions_desired = []
    all_positions_real = []
    all_positions_velocity = []
    # 记录整段轨迹的接触力信息在不同参考坐标系下
    all_contact_force_inTcp = [] # 在末端坐标系下
    all_contact_force_inBase = [] # 在基座标系下
    # ======================================================


    # ===========================夹具控制=========================
    # robot.set_gripper_init() # 初始化夹具
    # time.sleep(1)
    # robot.set_gripper_open() # 夹具打开
    # input("请放入插头，并按回车键继续")
    # time.sleep(1) 
    # robot.set_gripper_close() # 夹具闭合
    # time.sleep(5)
    # ==========================================================


    try:

        # ==========================主程序=========================
        servo_j_thread = threading.Thread(target=servo_j) # 内环位置控制线程
        servo_j_thread.start()
        admittance_controller() # 导纳控制器
        # ========================================================
        
    except KeyboardInterrupt: # CTRL+C退出程序

        logger.info("程序被中断")

    except Exception as e: # 其他异常

        logger.info("程序异常退出: ", e)

    finally:
        
        # ==========================清理工作=======================
        # 结束内环servoj线程
        servo_j_running = False
        servo_j_thread.join()

        # 结束读取力传感数据和机器人状态的线程
        read_force_data_running = False
        read_force_data_thread.join()
        calib_force_data_thread.join()

        logger.info("程序结束, 关闭socket连接和机器人连接")

        # 断开机器人服务器链接
        robot.disconnect()
        Auboi5Robot.uninitialize() # 释放库资源

        # 断开力传感器连接
        sock.close()

        logger.info("程序结束, 绘制接触力/力矩和末端位置/姿态")
        # ========================================================


        # ==========================数据保存=======================
        # 保存传感器坐标系下力数据
        force_data_inSensor_list_converted = [data.tolist() for data in force_data_inSensor_list]
        with open('force_data_inSensor.json', 'w') as f:
            json.dump(force_data_inSensor_list_converted, f)

        # 保存末端坐标系下力数据
        all_contact_force_inTcp_converted = [data.tolist() for data in all_contact_force_inTcp]
        with open('force_data_inTcp.json', 'w') as f:
            json.dump(all_contact_force_inTcp_converted, f)

        # 保存基坐标系下力数据
        all_contact_force_inBase_converted = [data.tolist() for data in all_contact_force_inBase]
        with open('force_data_inBase.json', 'w') as f:
            json.dump(all_contact_force_inBase_converted, f)

        # 保存期望关节角度
        all_positions_converted_desired = [pos.tolist() for pos in all_positions_desired]
        with open('trajectory_data.json', 'w') as f:
            json.dump(all_positions_converted_desired, f)
        
        # 保存实际关节角度
        all_positions_real_converted = [pos.tolist() for pos in all_positions_real]
        with open('trajectory_data_real.json', 'w') as f:
            json.dump(all_positions_real_converted, f)
        
        # 保存实际关节速度
        all_positions_velocity_converted = [pos.tolist() for pos in all_positions_velocity]
        with open('trajectory_velocity_real.json', 'w') as f:
            json.dump(all_positions_velocity_converted, f)
        # =========================================================


        # ==========================绘制传感器坐标系下的接触力数据（带时间戳）=======================
        # max_time = np.round(max(time_list)+1).astype(int)
        # fig, (ax1, ax2) = plt.subplots(2, 1)
        # lines1 = [ax1.plot([], [], label=label, color=color)[0] for label, color in zip(['fx', 'fy', 'fz'], ['r', 'g', 'b'])]
        # lines2 = [ax2.plot([], [], label=label, color=color)[0] for label, color in zip(['mx', 'my', 'mz'], ['r', 'g', 'b'])]

        # # 显示数据
        # ax1.set_ylabel('F/N'); ax1.legend()
        # ax2.set_xlabel('Time/s'); ax2.set_ylabel('M/Nm'); ax2.legend()

        # if len(time_list) == len(force_data_inSensor_list) and force_data_inSensor_list:
        #     data = np.array(force_data_inSensor_list)  # 转换为numpy数组
        #     times = np.array(time_list) 
        #     if len(times) == data.shape[0]:
        #         for i, line in enumerate(lines1):
        #             line.set_xdata(times)
        #             line.set_ydata(data[:, i])
        #         for i, line in enumerate(lines2):
        #             line.set_xdata(times)
        #             line.set_ydata(data[:, i + 3])
        #         ax1.set_xlim(0, max_time)
        #         ax1.set_ylim(np.min(data[:, :3]) - 1, np.max(data[:, :3]) + 1)
        #         ax2.set_xlim(0, max_time)
        #         ax2.set_ylim(np.min(data[:, 3:]) - 1, np.max(data[:, 3:]) + 1)

        # plt.tight_layout()
        # plt.savefig('force_data.png')
        # ========================================================================================
                

        # ====================================画末端位置和姿态图=====================================
        # 创建一个窗口，包含6个子图
        fig2, axes = plt.subplots(6, 1, figsize=(12, 18))
        pose_labels = ['x/m', 'y/m', 'z/m', 'rx/rad', 'ry/rad', 'rz/rad']
        colors = ['r', 'g', 'b', 'r', 'g', 'b']

        # 获取样本数量
        pose_data_array = np.array(pose_data_list)
        n_samples = pose_data_array.shape[0]

        # 绘制末端位姿变化曲线
        for i in range(6):
            axes[i].plot(np.arange(n_samples), pose_data_array[:, i], label=pose_labels[i], color=colors[i])
            axes[i].set_ylabel(pose_labels[i])
            axes[i].legend()
            axes[i].grid(True)
            # 添加垂直于x轴的直线
            axes[i].axvline(x=first_phase_end, color='r', linestyle='--')
            axes[i].axvline(x=second_phase_start, color='g', linestyle='--')
            axes[i].axvline(x=second_phase_end, color='g', linestyle='--')
            axes[i].axvline(x=third_phase_start, color='b', linestyle='--')
            axes[i].axvline(x=third_phase_end, color='b', linestyle='--')

        axes[5].set_xlabel('Sample')
        plt.tight_layout()
        plt.savefig('pose_data.png')
        # ========================================================================================


        # ===========================画末端期望位置和实际位置轨迹图（3D空间）===========================
        # 创建一个三维图形
        fig3 = plt.figure()
        ax = fig3.add_subplot(111, projection='3d')
        p_end_array = np.array(p_end_list)

        # 绘制末端实际位置的三维轨迹（蓝色）
        ax.plot(pose_data_array[:, 0], pose_data_array[:, 1], pose_data_array[:, 2], color='b', label='End Effector Position')
        # 绘制末端期望位置的三维轨迹（红色）
        ax.plot(p_end_array[:, 0], p_end_array[:, 1], p_end_array[:, 2], color='r', label='Desired End Effector Position')
        
        # 设置三维图形的标签
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title('End Effector Position Trajectory')
        ax.legend()
        plt.savefig('end_effector_position_trajectory.png')
        # ========================================================================================


        # ==============================实际关节速度曲线==============================
        # trajectory_velocity_real = np.array(all_positions_velocity)
        # n_samples = trajectory_velocity_real.shape[0]

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
        # plt.savefig('trajectory_velocity_real.png')
        # ==========================================================================


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
        # plt.savefig('trajectory_acc_real.png')
        # ============================================================================


        # ==============================实际关节变化曲线==============================
        # trajectory_array_real = np.array(all_positions_real)
        # # 获取样本数量
        # n_samples = trajectory_array_real.shape[0]

        # # 创建一个窗口，包含6个子图
        # fig, axes = plt.subplots(6, 1, figsize=(12, 18))
        # joint_labels = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6']

        # # 绘制每个关节的变化曲线
        # for i in range(6):

        #     axes[i].plot(np.arange(n_samples), trajectory_array_real[:, i], label=f'Real {joint_labels[i]}', color='red')
        #     axes[i].set_ylabel('Angle (rad)')
        #     axes[i].legend()
        #     axes[i].grid(True)

        # axes[5].set_xlabel('Sample')
        # plt.tight_layout()
        # plt.savefig('trajectory_array_real.png')
        # =========================================================================


        # ==============================期望关节变化曲线==============================
        # trajectory_array = np.array(all_positions_desired)
        # # 创建一个窗口，包含6个子图
        # fig, axes = plt.subplots(6, 1, figsize=(12, 18))
        # joint_labels = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6']

        # # 获取样本数量
        # n_samples = trajectory_array.shape[0]

        # # 绘制每个关节的变化曲线
        # for i in range(6):

        #     axes[i].plot(np.arange(n_samples), trajectory_array[:, i], label=f'Desired {joint_labels[i]}' ,color='blue')
        #     axes[i].set_ylabel('Angle (rad)')
        #     axes[i].legend()
        #     axes[i].grid(True)

        # axes[5].set_xlabel('Sample')
        # plt.tight_layout()
        # plt.savefig('trajectory_array_desired.png')
        # ==========================================================================


        # ==============================末端坐标系下接触力曲线==========================
        force_data_inTcp = np.array(all_contact_force_inTcp)
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
            # 添加垂直于x轴的直线
            axes[i].axvline(x=first_phase_end, color='r', linestyle='--')
            axes[i].axvline(x=second_phase_start, color='g', linestyle='--')
            axes[i].axvline(x=second_phase_end, color='g', linestyle='--')
            axes[i].axvline(x=third_phase_start, color='b', linestyle='--')
            axes[i].axvline(x=third_phase_end, color='b', linestyle='--')

        axes[5].set_xlabel('Sample')
        plt.tight_layout()
        plt.savefig('force_data_inTcp.png')
        # ==========================================================================


        # ==============================基座坐标系下接触力曲线==========================
        # force_data_inBase = np.array(all_contact_force_inBase)
        # # 创建一个窗口，包含6个子图
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
        #     # 添加垂直于x轴的直线
        #     axes[i].axvline(x=first_phase_end, color='k', linestyle='--')
        #     axes[i].axvline(x=second_phase_start, color='k', linestyle='--')
        #     axes[i].axvline(x=second_phase_end, color='k', linestyle='--')

        # axes[5].set_xlabel('Sample')
        # plt.tight_layout()
        # plt.savefig('force_data_inBase.png')
        # ==========================================================================

        plt.show()
