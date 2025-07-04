import os
import sys
import ikpy.inverse_kinematics
import numpy as np
import matplotlib.pyplot as plt
import time
import socket
import threading
import json
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
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
import pyRobotiqGripper
import serial
import minimalmodbus as mm


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

def gripper_init():
    #Constants
    BAUDRATE=115200
    BYTESIZE=8
    PARITY="N"
    STOPBITS=1
    TIMEOUT=0.2
    AUTO_DETECTION="auto"
    SLAVEADDRESS = 9
    ports=serial.tools.list_ports.comports()
    for port in ports:
        try:
            if port.serial_number == "DAA5HFG8":
                left_gripper_port = port.device
            elif port.serial_number == "DAA5H8OB":
                right_gripper_port = port.device
            # Try opening the port
            ser = serial.Serial(port.device,BAUDRATE,BYTESIZE,PARITY,STOPBITS,TIMEOUT)
            

            device=mm.Instrument(ser,SLAVEADDRESS,mm.MODE_RTU,close_port_after_each_call=False,debug=False)

            #Try to write the position 100
            device.write_registers(1000,[0,100,0])

            #Try to read the position request eco
            registers=device.read_registers(2000,3,4)
            posRequestEchoReg3=registers[1] & 0b0000000011111111

            #Check if position request eco reflect the requested position
            if posRequestEchoReg3 != 100:
                raise Exception("Not a gripper")

            del device

            ser.close()  # Close the port
        except:
            pass  # Skip if port cannot be opened

    return left_gripper_port, right_gripper_port

# 全局变量储存插口pose
headcam_all_hole = PoseArray()
leftcam_usb_plug = PoseArray() # 左手相机识别USB插头, 给出抓取点
rightcam_usb_pose = PoseArray() # 右手相机识别USB插头位姿，给出偏转角
recent_peg_positions = []
recent_hole_positions = []
peg_avg_position = np.zeros((3,1))
hole_avg_position = np.zeros((3,1))

def leftcam_usb_plug_callback(msg):

    """
    Func: 左手相机识别USB线缆, 给出抓取点, 做抓取
    Params: 
        位置信息成功对出现，对数不固定
        leftcam_usb_plug.poses[0] 是usb1头部pose
        leftcam_usb_plug.poses[1] 是usb1尾部pose
        leftcam_usb_plug.poses[2] 是usb2头部pose
        leftcam_usb_plug.poses[3] 是usb2尾部pose
        ...以此类推
    """ 

    global leftcam_usb_plug
    leftcam_usb_plug = msg
    # logger.info(f"leftcam_usb_plug: {leftcam_usb_plug.poses[0]}")


def rightcam_usb_pose_callback(msg):

    """
    Func: 右手相机识别USB插头, 计算偏转角, 区分正反面
    Params: 
        位置信息成功对出现，对数不固定
        rightcam_usb_pose.poses[0] 是usb插头的两个角点之一
        rightcam_usb_pose.poses[1] 是usb插头的两个角点之一
        rightcam_usb_pose.poses[2] 是usb插头的蓝色上的点
    """ 
    
    global rightcam_usb_pose
    rightcam_usb_pose = msg

def headcam_all_hole_callback(msg):

    """
    Func: 头部相机识别所有的插孔
    Params: 
        headcam_all_hole.poses[0]-[3] 是HDMI面板四角pose
        headcam_all_hole.poses[4] 是HDMI插口pose
        headcam_all_hole.poses[5] 是网线插口pose
        headcam_all_hole.poses[6] 是usb插口pose
    """   
    global headcam_all_hole
    headcam_all_hole = msg

def rightcam_usb_hole_callback(msg):
    
    """
    Func: 头部相机识别所有的插孔
    Params: 
        msg.poses[0].position 是插口的(x, y, z)
        msg.poses[1].position 是插头的(x, y, z)
    """ 

    global recent_peg_positions, recent_hole_positions, peg_avg_position, hole_avg_position, flag_vision_searchHole

    if len(msg.poses) == 2 and flag_vision_searchHole: # 必须要保证同时给出两个红框 and 视觉搜孔标志为1

        if (msg.poses[0].position.x == 0 and msg.poses[0].position.y == 0 and msg.poses[0].position.z == 0) \
        or (msg.poses[1].position.x == 0 and msg.poses[1].position.y == 0 and msg.poses[1].position.z == 0):
            
            pass
        
        else:

            peg_pos = msg.poses[1].position # 获取插头的(x, y, z)
            peg_position = np.array([peg_pos.x, peg_pos.y, peg_pos.z])
            hole_pos = msg.poses[0].position # 获取插口的(x, y, z)
            hole_position = np.array([hole_pos.x, hole_pos.y, hole_pos.z])

            # 保存到列表
            recent_peg_positions.append(peg_position)
            recent_hole_positions.append(hole_position)

            if len(recent_peg_positions) >= 25 and len(recent_hole_positions) >= 25:

                peg_avg_position = np.mean(recent_peg_positions, axis=0) # 计算平均值
                hole_avg_position = np.mean(recent_hole_positions, axis=0) # 计算平均值
                flag_vision_searchHole = False

    else:
        pass


# 全局变量存储udp上报数据
leftArm_latest_data = {
    'joint_position': [0.0] * 7,  # 7个关节角度
    'pose': [0.0] * 6,            # 前3个是position(x,y,z)，后3个是euler(rx,ry,rz)
    'zero_force': [0.0] * 6       # 传感器坐标系下系统收到的外力
}
rightArm_latest_data = {
    'joint_position': [0.0] * 7,  # 7个关节角度
    'pose': [0.0] * 6,            # 前3个是position(x,y,z)，后3个是euler(rx,ry,rz)
    'zero_force': [0.0] * 6       # 传感器坐标系下系统收到的外力
}

def arm_udp_status(data):

    '''
    Func: 双臂UDP上报的回调函数, 将数据保存到全局变量
    Params: 
        leftArm_latest_data 左臂数据，全局变量
        rightArm_latest_data 右臂数据，全局变量
    '''
    global leftArm_force_zero 

    if data.arm_ip == b'192.168.99.17': # 左臂

        leftArm_latest_data['joint_position'] = np.array(data.joint_status.joint_position) # 角度制
        cur_ee_pos = np.array([data.waypoint.position.x, data.waypoint.position.y, data.waypoint.position.z])
        cur_ee_ori = np.array([data.waypoint.euler.rx, data.waypoint.euler.ry, data.waypoint.euler.rz])
        leftArm_latest_data['pose'] = np.concatenate((cur_ee_pos, cur_ee_ori))
        leftArm_latest_data['zero_force'] = np.array(data.force_sensor.zero_force) - leftArm_force_zero

    elif data.arm_ip == b'192.168.99.18': # 右臂
        
        rightArm_latest_data['joint_position'] = np.array(data.joint_status.joint_position) # 角度制
        cur_ee_pos = np.array([data.waypoint.position.x, data.waypoint.position.y, data.waypoint.position.z])
        cur_ee_ori = np.array([data.waypoint.euler.rx, data.waypoint.euler.ry, data.waypoint.euler.rz])
        rightArm_latest_data['pose'] = np.concatenate((cur_ee_pos, cur_ee_ori))
        rightArm_latest_data['zero_force'] = np.array(data.force_sensor.zero_force)
    
    else: 
        pass

plug_in_state = np.zeros(6) # 插接各个阶段的状态标志

def pose_force_publisher():

    '''
    Func: 发布插接时的末端位姿和末端接触力数据, 给软件组界面用
    '''

    global raw_force_data, rightArm_latest_data, plug_in_state

    pub1 = rospy.Publisher('/right_arm_pose', Float64MultiArray, queue_size=10) # 发布插接各个阶段的右臂末端位姿
    pub2 = rospy.Publisher('/right_arm_force', Float64MultiArray, queue_size=10) # 发布插接各个阶段的右臂接触力
    pub3 = rospy.Publisher('/plug_in_state_pub', Float64MultiArray, queue_size=10) # 发布插接各个阶段的状态标志
    rate = rospy.Rate(10)  # 10Hz发布频率
    
    while pub_pose_force_running:

        pose_msg = Float64MultiArray()
        pose_msg.data = rightArm_latest_data['pose']
        
        force_msg = Float64MultiArray()
        force_msg.data = raw_force_data

        plug_in_state_msg = Float64MultiArray()
        plug_in_state_msg.data = plug_in_state
        
        # 发布消息
        pub1.publish(pose_msg)
        pub2.publish(force_msg)
        pub3.publish(plug_in_state_msg)
        
        rate.sleep()

def update_arm_left_state():

    """
    Func: 获取当前左臂信息, 从保存左臂状态的全局变量里面读取
    """
    return leftArm_latest_data['joint_position'].copy(), leftArm_latest_data['pose'].copy(), leftArm_latest_data['zero_force'].copy()

def update_arm_ri_state():

    """
    Func: 获取当前右臂信息, 从保存右臂状态的全局变量里面读取
    """
    return rightArm_latest_data['joint_position'].copy(), rightArm_latest_data['pose'].copy(), rightArm_latest_data['zero_force'].copy()

leftArm_force_zero = np.zeros(6)

def arm_le_clear_force():

    global leftArm_force_zero
    _, _, get_leftArm_force = update_arm_left_state()
    leftArm_force_zero += get_leftArm_force

def read_forcedata(sensor):

    """
    Func: 读取力传感器数据的线程函数
    """
    sensor.save_forcedata() # 启动数据流并测量频率

raw_force_data = np.zeros(6) # fx fy fz Mx My Mz 原始力传感器数据
calib_force_data = np.zeros(6) # 标定后的力数据
force_zero_point = np.zeros(6) # 力传感器数据清零
read_force_data_running = True # 读取数据线程运行标志
G_force = np.zeros(6) # Gx Gy Gz Mgx Mgy Mgz 负载重力在力传感器坐标系下的分量

def calib_force_data_func(robot):

    '''
    Func: 校准力传感器数据, 输出负载辨识和标定后的纯外力数据
    '''

    global calib_force_data,read_force_data_running, raw_force_data, force_zero_point

    # 读取 calibration_result.JSON 文件得到标定结果
    file_name = './calibrate_forceSensor/calibration_result.json'
    with open(file_name, 'r') as json_file:
        json_data = json.load(json_file)
        gravity_bias = np.array(json_data['gravity_bias'])
        mass = np.array(json_data['mass'][0])
        force_zero = np.array(json_data['force_zero'])

    alpha = 0.9 # 滑动窗口加权滤波
    smoothed_force_data = np.zeros(6) # 平滑后的力传感器数据
    time_count = 0

    start_time = time.time()
    
    while read_force_data_running:
        
        if raw_force_data is not None:
            
            raw_force_data = np.array(sensor.forcedata).copy()
            smoothed_force_data = alpha * raw_force_data  + (1-alpha) * smoothed_force_data

            '''
            获取机械臂当前的力传感器坐标系的末端姿态, 需要根据不同的机械臂型号和通信协议获取
            '''
            joint_r, _, _ = update_arm_ri_state()
            joints_r_rad = [j/180*np.pi for j in joint_r]
            r_ee = pykin_fsensor.forward_kin(joints_r_rad)
            r_ee_rot = r_ee[pykin_fsensor.eef_name].rot

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
            # logger.info(f"calib_force_data: {calib_force_data}") # 打印标定后的力数据
            
            # 保存数据
            force_data_inSensor_list.append(calib_force_data) # 末端受到的外部力
            time_list.append(time.time() - start_time) # 时间

            time.sleep(0.001) 
        
        else :

            logger.info("force_data is None, 没有收到原始力传感器数据")
            time.sleep(0.005)

def force_data_zero():

    '''
    Func: 力传感器数据清零
    '''
    global force_zero_point, calib_force_data
    force_zero_point += calib_force_data.copy()

def pose_add(pose1, pose2):

    '''
    Func: 位姿相加函数，计算两个位姿的和，位置直接相加，姿态用四元数相加: q1 + q2 = q2 * q1
    Params:
        pose1: 位姿1 [x1,y1,z1,rx1,ry1,rz1]
        pose2: 位姿2 [x2,y2,z2,rx2,ry2,rz2]
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
    Func: 位姿相减函数，计算两个位姿的差，位置直接相减，姿态用四元数相减: q1 - q2 = q2' * q1
    Params:
        pos1: 位姿1 [x1,y1,z1,rx1,ry1,rz1]
        pos2: 位姿2 [x2,y2,z2,rx2,ry2,rz2]
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

def robot_inverse_kinematic(refer_joints,target_pos,target_ori_rpy_rad):
    
    '''
    Func: 逆运动学求解关节角, 通过在服务器上运行pykin求解, 求解结果通过ros.service返回
    '''
    target_pose = [0.0] * 6
    target_pose[:3] = target_pos
    target_pose[3:] = target_ori_rpy_rad
    rsp = kin_inverse(refer_joints, target_pose)
    
    return rsp.joint_angles

def generate_trajectory_with_constraints(q_start, v_start, a_start, q_end, v_end, a_end, move_period, lookahead_time, max_velocity, max_acceleration):
    
    """
    Func: 生成五次多项式轨迹，确保速度和加速度不超限，并返回时间戳
    Params:
        q_start: 初始位置 (数组)
        v_start: 初始速度 (数组)
        a_start: 初始加速度 (数组)
        q_end: 目标位置 (数组)
        v_end: 目标速度 (数组)
        a_end: 目标加速度 (数组)
        move_period: 轨迹采样时间间隔
        lookahead_time: 轨迹总时长
        max_velocity: 允许的最大速度
        max_acceleration: 允许的最大加速度
    Return: 时间戳、轨迹的位置信息、速度信息、加速度信息、最终轨迹时长
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
            break  # 如果满足约束，停止调整
        
        T +=0.03  # 增大轨迹时间，降低速度和加速度
        
    return t_vals, positions, velocities, accelerations, T  # 返回时间戳


def trans_force_data(trans_matrix, force_data):
    '''
    Func: 将力传感器数据从A坐标系转换到B坐标系
    Params:
        trans_matrix: 坐标系转换矩阵: 从A坐标系到B坐标系, B坐标系下A坐标系的位姿
        force_data: A坐标系下的力传感器数据 [Fx, Fy, Fz, Tx, Ty, Tz]
    Return: B坐标系下的力传感器数据 [Fx, Fy, Fz, Tx, Ty, Tz]
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
    Func: 阻抗控制器, 用于机械臂外环控制, 控制周期25hz
    '''
    global num_joints, flag_enter_tcp2canbus, calib_force_data, p_end_list, peg_avg_position, hole_avg_position, flag_vision_searchHole

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


    '''
    前进阶段
    '''
    # input("enter to continue 前进")

    approaching_numPoints = 300 # 前进阶段的点数
    trans_length = 0.15  # 前进距离
    delta_steps = trans_length / approaching_numPoints # 前进的每一步的步长
    judge_force_threshold = -3 # -N
    ee_rotateMatrix = SO3.RPY(p_start_ori[0], p_start_ori[1], p_start_ori[2])  # 生成3D旋转对象(0.0, 0.0, 0.0)
    trans_vector_inTcp = np.array([delta_steps, 0.0, 0.0]).reshape(3,1) # Tcp坐标系下的前进位移向量
    trans_vector_inBase = (ee_rotateMatrix * trans_vector_inTcp ).flatten() # 转换到Base坐标系下的前进位移向量
    force2tcp = SE3.Trans(0.0, 0.0, -0.199) * SE3.RPY(1.57, 2.356, 0.0, order='xyz') # 从末端工具坐标系转换到力传感器坐标系
    approaching_control_period = 0.045 # 透传周期45ms
    plug_in_state[1] = 1 # 前进阶段标志位为1

    for i_0 in range(approaching_numPoints):

        logger.info("第{}个".format(i_0)+": 前进阶段")

        st_time = time.perf_counter()
        p_end_pos += trans_vector_inBase # 前半段用规划的轨迹，x向前
        p_end_list.append(np.concatenate((p_end_pos, p_end_ori), axis=0)) # 存储末端期望位姿

        q_target = np.array(robot_inverse_kinematic(joint_r, p_end_pos, p_end_ori))
        robot.Movej_CANFD(q_target,False,0) # 透传目标位置给机械臂
        all_positions_desired.append(q_target) # 保存期望关节角度

        joint_r, pose_r, _ = update_arm_ri_state()
        all_positions_real.append(joint_r) # 存储当前关节角
        pose_data_list.append(pose_r) # # 存储末端当前位姿

        contact_force_data_inTcp = trans_force_data(force2tcp.A, calib_force_data) # 力传感器坐标系下力转换到末端坐标系
        all_contact_force_inTcp.append(contact_force_data_inTcp) # 保存末端坐标系下的受力
        judge_force = contact_force_data_inTcp[0].copy() # 判断力传感器受力是否超过阈值
        logger.info(f'judge_force: {judge_force}')
        
        if judge_force <= judge_force_threshold:
            logger.info("前进阶段结束，进入搜孔阶段")
            plug_in_state[1] = 0 # 前进阶段结束
            plug_in_state[2] = 1 # 搜孔阶段开始
            break

        elapsed_time = time.perf_counter()-st_time
        logger.info(f"time cost: {elapsed_time}")
        if elapsed_time > approaching_control_period: # 固定透传周期
            pass
        else:
            time.sleep(approaching_control_period - elapsed_time)

    time.sleep(1) # 等待稳定
    _, pose_r, _ = update_arm_ri_state()
    p_end_pos = np.array((pose_r[:3]), dtype=float) # 更新当前末端位置 in m
    p_end_ori = np.array((pose_r[3:]), dtype=float) # 更新当前末端姿态 in rad


    '''
    视觉引导搜孔阶段
    '''
    # input("enter to continue 搜孔")

    flag_vision_searchHole = True # 视觉搜孔标志置为1
    searching_numPoints = 4000 # 搜孔阶段的点数

    while not rospy.is_shutdown():

        if flag_vision_searchHole: # 等待视觉回调函数计算平均值
            continue

        else:

            peg_avg_position = np.array(peg_avg_position).flatten()
            hole_avg_position = np.array(hole_avg_position).flatten()
            peg_pos_inTcp = arm_cam2robotiq_right @ np.array([peg_avg_position[0], peg_avg_position[1], peg_avg_position[2], 1.0]).reshape(4, 1)
            hole_pos_inTcp = arm_cam2robotiq_right @ np.array([hole_avg_position[0], hole_avg_position[1], hole_avg_position[2], 1.0]).reshape(4, 1)
            vector_peg2hole_inTcp = np.abs(np.array(hole_pos_inTcp - peg_pos_inTcp)[:3].reshape(3, 1)) # (3,1)
            logger.info(f"vector_peg2hole_inTcp:{vector_peg2hole_inTcp}")

            keyboard = input("请确认识别结果是否正确, 正确直接enter, 错误则e+enter")

            if keyboard == 'e':
                flag_vision_searchHole = True  # 重新识别20次，计算平均值
            else:
                break # 跳出循环，继续后续流程

        time.sleep(0.05)

    vector_peg2hole_inTcp[0] *= 0.40 # x往里走的距离减小，减小的话往里的力也会小
    vector_peg2hole_inTcp[1] *= 0.50 # y 越大越往左
    vector_peg2hole_inTcp[2] *= 1.00 # z往上走的距离减小
    norm = np.linalg.norm(vector_peg2hole_inTcp) # 计算模长
    if norm == 0:
        unit_vector_inTcp = np.zeros_like(vector_peg2hole_inTcp)
    else:
        unit_vector_inTcp = vector_peg2hole_inTcp / norm  # 单位向量  unit: m
    trans_vector_inTcp = unit_vector_inTcp / searching_numPoints  # 固定长度、方向与原向量平行 unit: mm
    ee_rotateMatrix = SO3.RPY(pose_r[3], pose_r[4], pose_r[5])  # 生成3D旋转对象(0.0, 0.0, 0.0)
    trans_vector_inBase = (ee_rotateMatrix * trans_vector_inTcp ).flatten() # 转换到Base坐标系下的前进位移向量
    trans_vector_inBase = np.array(trans_vector_inBase, dtype=float)
    searching_control_period = 0.045 # 透传周期45ms 
    judge_force_threshold = 7 # 搜孔阶段受力判断阈值, Tcp坐标系下的受力

    for i_1 in range(searching_numPoints):

        logger.info("第{}个".format(i_0+i_1+1)+": 搜孔阶段")

        st_time = time.perf_counter()
        p_end_pos += trans_vector_inBase # x往前走，yz做螺旋运动
        p_end_list.append(np.concatenate((p_end_pos, p_end_ori), axis=0)) # 存储末端期望位姿

        q_target = np.array(robot_inverse_kinematic(joint_r, p_end_pos, p_end_ori))
        robot.Movej_CANFD(q_target,False,0) # 透传目标位置给机械臂
        all_positions_desired.append(q_target) # 保存期望关节角度

        joint_r, pose_r, _ = update_arm_ri_state()
        all_positions_real.append(joint_r) # 存储当前关节角
        pose_data_list.append(pose_r) # # 存储末端当前位姿

        contact_force_data_inTcp = trans_force_data(force2tcp.A, calib_force_data) # 6*1
        all_contact_force_inTcp.append(contact_force_data_inTcp) # 保存末端坐标系下的受力
        judge_force = np.sqrt(np.sum(contact_force_data_inTcp[1]**2 + contact_force_data_inTcp[2]**2)) # 判断力传感器受力是否超过阈值
        logger.info(f'judge_force: {judge_force}')

        if judge_force >= judge_force_threshold:
            logger.info("搜孔阶段结束，进入插孔第一阶段")
            plug_in_state[2] = 0 # 搜孔阶段结束
            plug_in_state[3] = 1 # 第一阶段开始
            break

        elapsed_time = time.perf_counter()-st_time
        logger.info(f"time cost: {elapsed_time}")
        if elapsed_time > searching_control_period: # 固定透传周期
            pass
        else:
            time.sleep(searching_control_period - elapsed_time)

    
    _, pose_r, _ = update_arm_ri_state()
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
        joint_r, pose_r, _ = update_arm_ri_state()
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

            judge_force_threshold = -2 # 插孔的第一阶段受力判断阈值, Tcp坐标系下的受力

            trans_vector_inTcp = np.array([0.05/1000, 0.0, 0.0]).reshape(3,1) # Tcp坐标系下的前进位移向量
            ee_rotateMatrix = SO3.RPY(ee_ori_rpy_rad[0], ee_ori_rpy_rad[1], ee_ori_rpy_rad[2])  # 生成3D旋转对象(0.0, 0.0, 0.0)
            trans_vector_inBase = (ee_rotateMatrix * trans_vector_inTcp).flatten() # 转换到Base坐标系下的前进位移向量
            p_end_pos = p_end_pos + trans_vector_inBase # x往前走

            # user_force_inTcp = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # 用户主动施加力，Tcp坐标系下
            logger.info("插孔的第一阶段: 先往里插")

        elif second_phase: #  插孔第二阶段，主动偏转

            fingertip_in_tcp = SE3.Trans(0.018, 0.0, 0.0) * SE3.RPY(0.0, 0.0, 0.0) # 指尖在末端下的变换
            tcp_in_fingertip = fingertip_in_tcp.inv() # 末端在指尖下的变换（逆变换）

            if second_phase_startforce > 0: # 如果是绕着Tcp_y轴的正向扭矩
                delta_theta = 2/180*np.pi  # 每周期旋转的弧度
                judge_moment_threshold = 0.5 # Tcp坐标系下的绕y轴的力矩My的阈值
            else:
                delta_theta =  -2/180*np.pi  # 每周期旋转的弧度
                judge_moment_threshold = -0.5 # Tcp坐标系下的绕y轴的力矩My的阈值

            rotate_y_in_fingertip = SE3.RPY(0.0, delta_theta, 0.0)
            new_tcp_pose = tcp_pose * fingertip_in_tcp * rotate_y_in_fingertip * tcp_in_fingertip # 先变换到指尖坐标系，再绕y轴旋转，再变回末端坐标系
            new_pose = new_tcp_pose.A  # 如果需要numpy数组

            p_end_pos = new_pose[:3,3] 
            p_end_ori = transform_utils.get_rpy_from_matrix(new_pose[:3,:3])


        elif third_phase: # 插孔的第三阶段，直插到底

            judge_force_threshold = -20 # 插孔的第三阶段受力判断阈值, Tcp坐标系下的受力

            trans_vector_inTcp = np.array([0.15/1000, 0.0, 0.05/1000]).reshape(3,1) # Tcp坐标系下的前进位移向量
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
                logger.info(f"second_phase_startforce: {second_phase_startforce}")
                plug_in_state[3] = 0 # 第一阶段结束
                plug_in_state[4] = 1 # 第二阶段开始
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
            if i_2 - second_phase_startPoint >= 10 and  judge_momentY <= judge_moment_threshold: 
                second_phase = False
                third_phase = True
                controller.pose_error = np.zeros(6) # 重置导纳控制器的位姿误差
                controller.velocity_error = np.zeros(6) # 重置导纳控制器的速度误差
                _, pose_r, _ = update_arm_ri_state()
                p_end_pos = np.array((pose_r[:3])) # 更新当前末端位置 in m
                p_end_ori = np.array((pose_r[3:])) # 更新当前末端姿态 in rad
                plug_in_state[4] = 0 # 第二阶段结束
                plug_in_state[5] = 1 # 第三阶段开始
                logger.info("插孔的第二阶段结束，进入第三阶段直插到底")
                
        elif third_phase: # 插孔第三阶段，直插到底

            contact_force_data_inTcp[1] *= 0.01 # 减小末端y方向的受力
            contact_force_data_inTcp[2] *= 0.01 # 减小末端z方向的受力
            contact_force_data_inTcp[3] *= 0.01 # 减小末端绕x轴的力矩
            contact_force_data_inTcp[4] *= 0.01 # 减小末端绕y轴的力矩
            contact_force_data_inTcp[5] *= 0.01 # 减小末端绕z轴的力矩
            user_force_inTcp[0] = -contact_force_data_inTcp[0] + 15 # 用户主动施加力，Tcp坐标系下
                
            judge_force = contact_force_data_inTcp[0] # Tcp坐标系下受力，为-x，朝后
            logger.info(f'judge_force: {judge_force}')
            logger.info("插孔的第三阶段: 直插到底")

            if judge_force <= judge_force_threshold: # 如果受力超过阈值，说明插孔结束
                third_phase = False
                plug_in_state[5] = 0 # 第三阶段结束
                logger.info("插孔结束")
                break
        else:
            pass
        
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
    Func: 机械臂内环位置控制, 控制周期40ms
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

def dual_arm_move(joints_left, joints_right, speed=15):

    thread_le = threading.Thread(target=arm_le.Movej_Cmd, args=(joints_left, speed))
    thread_ri = threading.Thread(target=arm_ri.Movej_Cmd, args=(joints_right, speed))
    thread_le.start()
    thread_ri.start()
    thread_le.join()
    thread_ri.join()

def dual_gripper_move(left_gripper,right_gripper):

    thread_le = threading.Thread(target=gripper_le.goTo, args=(left_gripper))
    thread_ri = threading.Thread(target=gripper_ri.goTo, args=(right_gripper))
    thread_le.start()
    thread_ri.start()
    thread_le.join()
    thread_ri.join()

def arm_inverse_kine(pykin_arm,refer_joints,target_pose):

    '''
    Func: 逆运动学求解关节角
    Params: 
        refer_joints 迭代法求解的参考关节角
        target_pose 末端的目标位姿矩阵
    Return: arm_joint 求解的关节角 in degree
    '''

    r_qd = pykin_arm.inverse_kin([j/180*np.pi for j in refer_joints], target_pose, method="LM", max_iter=100)
    arm_joint = [j/np.pi*180 for j in r_qd]
    logger.info(f'arm_joint: {arm_joint}')

    return arm_joint

# def get_target_xyz_head():

#     '''
#     Func: 头部相机坐标下目标位姿转换到机械臂base坐标系下
#     Params:
#         headcam_all_hole.poses 目标在头部相机坐标系下的位置
#         head_cam2base_left 手眼标定矩阵, 从头部相机到左臂的base坐标系
#         head_cam2base_right 手眼标定矩阵, 从头部相机到右臂的base坐标系
#     Return: pose_out 机械臂base坐标系下目标的位姿矩阵 4*4
#     '''
#     global headcam_all_hole
#     while not rospy.is_shutdown():
#         if (headcam_all_hole.poses[5].position.x == 0 and headcam_all_hole.poses[5].position.y == 0 and headcam_all_hole.poses[5].position.z == 0):
#             logger.info("识别失败")
#         else:
#             break
#         time.sleep(0.01)
#     pose_out = np.eye(4)
#     pose_out[0,3] = headcam_all_hole.poses[5].position.x # 用网线孔的位置来示教定位USB孔的位置
#     pose_out[1,3] = headcam_all_hole.poses[5].position.y
#     pose_out[2,3] = headcam_all_hole.poses[5].position.z
#     # pose_put = head_cam2base_left @ pose_put
#     pose_out = head_cam2base_right @ pose_out
#     # logger.info(f"pose_out: {pose_out}")
#     # logger.info(f'base下目标位置: {pose_out[:3,3]}')
#     # logger.info(f'当前base下目标的旋转矩阵 : {pose_out}')

#     return pose_out

def get_target_xyz_head_aruco(trans,markid,markerwidth,count=1):

    dict_gen = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
    cameraMatrix = np.array([[389.541259765625, 0.0, 320.45037841796875],
                            [0.0, 388.84527587890625, 239.62796020507812],
                            [0.0, 0.0, 1.0]],np.float32)
    distCoeffs = np.array([-0.054430413991212845, 0.06396076083183289, -0.00016922301438171417, 0.000722995144315064, -0.021027561277151108])


    # Initialize the detector parameters using default values
    parameters =  cv2.aruco.DetectorParameters()
    time_count = 0
    time_now = time.time()
    # =========================实时视频流检测=============================
    
    bridge = CvBridge()
    pose_all = np.empty((3,0))
    while True:
        color_image = rospy.wait_for_message("/head_camera/color/image_raw", Image, timeout=None)
        color_img = bridge.imgmsg_to_cv2(color_image, 'bgr8')
        print('相机图像获取成功')
        # 显示帧
        # cv2.namedWindow('Video',cv2.WINDOW_NORMAL)  
        # cv2.imshow('Video', color_img)

        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)  

        # Detect the markers in the image
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(gray, dict_gen, parameters=parameters)
        # frame_markers = cv2.aruco.drawDetectedMarkers(color_img, markerCorners, markerIds)
        if markerIds is not None:
            if markid in markerIds:
                for i, id_array in enumerate(markerIds):
                    if id_array[0] == markid:
                        index = i
                        break
            else:
                continue
            markerCorner = markerCorners[index]
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorner, markerwidth, cameraMatrix, distCoeffs)
            print('tvecs',tvecs)
            for rvec_, tvec_ in zip(rvecs, tvecs):
                frame_axes = cv2.drawFrameAxes(color_img, cameraMatrix, distCoeffs, rvec_, tvec_, 1)
            
            #转换为位姿矩阵
            rvec = np.array(rvecs)
            tvec = np.array(tvecs)
            R,_ = cv2.Rodrigues(rvec)
            pose = np.zeros((4,4))
            pose[:3,:3] = R
            pose[:3,3] = tvec 
            pose[3,3] = 1

            #---------------------
            pose_put = pose * trans

            R = pose_put[:3,:3]
            # rvec = rot2euler(R) / 180 * np.pi
            rvec_put = rvec.reshape(1,1,3) #弧度
            tvec_put = pose_put[:3,3].reshape(1,1,3)#单位/cm

            for rvec_, tvec_ in zip(rvec_put, tvec_put):
                frame_axes = cv2.drawFrameAxes(color_img, cameraMatrix, distCoeffs, rvec_, tvec_, 1)
            cv2.imwrite("./images/frame_axes.jpg", frame_axes)
                # frame_axes = cv2.aruco.drawAxis(color_img, cameraMatrix, distCoeffs, rvec, tvec, 1)
                
            # if markerIds is not None:
            #     cv2.namedWindow('frame_makers',cv2.WINDOW_NORMAL)
            #     cv2.imshow('frame_markers',frame_axes)
            
            pose_put[:3,3]=pose_put[:3,3]/100
            # print('相机下目标位置：', pose_put[:3,3])
            pose_all = np.hstack((pose_all,pose_put[:3,3].reshape(3,1)))

            time_count += 1
            print('time_count:',time_count)
            if time_count >= count:
                # pose_filter = filter_pose(pose_all)
                # print('pose_filter:',np.mean(pose_filter,axis=1))
                
                pose_mean = np.mean(pose_all,axis=1)
                pose_put[:3,3] = pose_mean
                print('pose_mean:',pose_mean)
                
                pose_put = head_cam2base_right @ pose_put
                print('base下目标位置:',pose_put[:3,3])
                print('当前base下目标的旋转矩阵',pose_put)
                break
            # 按 'Esc' 退出循环/home/
            if cv2.waitKey(1) & 0xFF == 27:  # 27是Esc键的ASCII值
                print(f'pose_put[:3,3]/100: {pose_put[:3,3]/100}')
                break

    return pose_put

def cable_plugging_pre(robot):
    
    '''
    Func: 机械臂移动到插孔附近
    '''
    # 双臂移动到待插接位置
    pre_plugging_le = [84.5, 30.12, 43.56, 45.46, -30.93, 22.95, 27.0]
    pre_plugging_ri = [55.72, -83.69, -64.77, -31.44, -30.58, -82.46, 131.75]
    dual_arm_move(pre_plugging_le,pre_plugging_ri,30)

    input("enter to continue")
    
    refer_joints = [52.34, -79.31, -49.43, -26.93, -44.75, -93.21, 128.53]
    put_Tx = np.array( [[ 0.03095872, -0.99401906, -0.10472664,  0.14661507],
                        [-0.24970449, -0.10914618,  0.96215112, -0.02892161],
                        [-0.96782706, -0.00363625, -0.25159005,  0.01354862+0.01],
                        [ 0.,          0.,          0.,          1.        ]])

    trans = np.array([0, 0, 0])
    put_pose = get_target_xyz_head_aruco(SE3.Trans(trans),11,9.5)
    plug_in_state[0] = 1 # 开始阶段标志位为1，开始生成吸引域
    time.sleep(0.03)
    put_target = put_pose @ put_Tx
    arm_joints = arm_inverse_kine(pykin_ri,refer_joints,put_target)
    robot.Movej_Cmd(arm_joints,30)


def get_target_xyz_le():

    '''
    Func: 左臂相机坐标下目标插头位姿转换到左臂base坐标系下
    Params:
        leftcam_usb_plug.poses USB插头在左臂相机坐标系下的位置 poses[0]: 上面的点 poses[1]: 下面的点
        arm_cam2robotiq_left 手眼标定矩阵, 从左臂相机到左臂的base坐标系
    Return: 
        pose_out 左臂base坐标系下目标的位姿矩阵 4*4
        theta_deg USB插头与相机坐标系x轴(水平方向)的偏转角 in degree
    '''
    
    global leftcam_usb_plug
    
    while not rospy.is_shutdown():

        if (leftcam_usb_plug.poses[0].position.x == 0 and leftcam_usb_plug.poses[0].position.y == 0 and leftcam_usb_plug.poses[0].position.z == 0) \
        or (leftcam_usb_plug.poses[1].position.x == 0 and leftcam_usb_plug.poses[1].position.y == 0 and leftcam_usb_plug.poses[1].position.z == 0):
            logger.info("识别失败")
        else:
            point_a = np.array([leftcam_usb_plug.poses[0].position.x,leftcam_usb_plug.poses[0].position.y,leftcam_usb_plug.poses[0].position.z])
            point_b = np.array([leftcam_usb_plug.poses[1].position.x,leftcam_usb_plug.poses[1].position.y,leftcam_usb_plug.poses[1].position.z])
            logger.info(f"抓取插头的point_a: {point_a}")
            logger.info(f"抓取插头的point_b: {point_b}")
            keyboard = input("请确认识别结果是否正确, 正确直接enter, 错误则e+enter")
            if keyboard == 'e':
                continue  # 重新识别
            else:
                break # 跳出循环，继续后续流程
        time.sleep(0.01)
    
    proportion_k = -0.5 # 计算抓取点c，C- B = K* (B- A)
    point_c = point_b + proportion_k* (point_b- point_a)
    cam_graspPoint = np.ones(4); cam_graspPoint[:3] = point_c
    vector_pointBA = point_a - point_b
    x_axis = np.array([1, 0, 0]) # 水平方向的方向向量
    cos_theta = np.dot(vector_pointBA, x_axis) / (np.linalg.norm(vector_pointBA) * np.linalg.norm(x_axis))
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    theta_deg = np.degrees(theta_rad)
    pose_out = np.eye(4)
    pose_out[0,3] = point_c[0]; pose_out[1,3] = point_c[1]; pose_out[2,3] = point_c[2]
    _, _, pose_l, _, _ = arm_le.Get_Current_Arm_State()
    ee2base_transform_matrix = np.eye(4)
    ee2base_transform_matrix[:3, 3] = [pose_l[0], pose_l[1], pose_l[2]]
    ee2base_transform_matrix[:3, :3] = transform_utils.get_matrix_from_rpy([pose_l[3],pose_l[4],pose_l[5]])
    pose_out = ee2base_transform_matrix @ arm_cam2robotiq_left @ pose_out

    return pose_out, theta_deg


def cable_pick(robot):

    '''
    Func: 左臂抓取线缆
    '''
    
    pick_pre_le = [-31.71, 84.94, 59.68, 33.91, 28.57, 88.42, -60.59]
    pick_pre_ri = [-0.64, 10.07, 7.63, -35.83, -4.98, -85.54, 26.24]
    dual_arm_move(pick_pre_le,pick_pre_ri, 30)

    refer_joints = [-40.84, 85.74, 78.23, 64.48, 17.09, 63.83, -123.28]
    gripper2base_T = ([[ 0.47398583,  0.87794486,  0.0674556 ],
                       [-0.08185094, -0.03234561,  0.99611956],
                       [ 0.87671994, -0.47766786,  0.05652925]]) # 固定的末端姿态

    put_target, theta_deg= get_target_xyz_le()
    time.sleep(0.03)
    put_target[:3,:3] = gripper2base_T
    put_target[1][3] += 0.005 # y+=垂直于桌面向下
    put_target[2][3] += 0.03 # z+=水平朝左
    arm_joints = arm_inverse_kine(pykin_le,refer_joints,put_target)
    arm_joints[6] += (90 - theta_deg)
    arm_le.Movej_Cmd(arm_joints,15) # 移动到抓取点
    gripper_le.goTo(255) # 夹爪闭合
    time.sleep(0.5)
    arm_le.Movej_Cmd(pick_pre_le,30) # 抓起来线缆

def get_target_xyz_ri():

    '''
    Func: 计算右臂相机坐标下目标插头偏转角, 用于区分正反面和做偏转
    Params:
        rightcam_usb_pose.poses USB插头在右臂相机坐标下的点的位置
    Return: 
        rotate_angle USB插头与右臂相机坐标x轴(水平方向)的偏转角 in degree
        deflect_angle 将USB插头转正需要的偏转角 in degree
    '''

    global rightcam_usb_pose

    while not rospy.is_shutdown():

        if (rightcam_usb_pose.poses[0].position.x == 0 and rightcam_usb_pose.poses[0].position.y == 0 and rightcam_usb_pose.poses[0].position.z == 0) \
        or (rightcam_usb_pose.poses[1].position.x == 0 and rightcam_usb_pose.poses[1].position.y == 0 and rightcam_usb_pose.poses[1].position.z == 0) \
        or (rightcam_usb_pose.poses[2].position.x == 0 and rightcam_usb_pose.poses[2].position.y == 0 and rightcam_usb_pose.poses[2].position.z == 0):
            
            logger.info("识别失败")
        else:

            point_a = np.array([rightcam_usb_pose.poses[0].position.x,rightcam_usb_pose.poses[0].position.y,rightcam_usb_pose.poses[0].position.z]) # 两个角点之一 a
            point_b = np.array([rightcam_usb_pose.poses[1].position.x,rightcam_usb_pose.poses[1].position.y,rightcam_usb_pose.poses[1].position.z]) # 两个角点之一 b 
            point_c = np.array([rightcam_usb_pose.poses[2].position.x,rightcam_usb_pose.poses[2].position.y,rightcam_usb_pose.poses[2].position.z]) # 蓝色的点 c
            point_mid = (point_a+ point_b)/ 2 # 两个角点的中点 d
            vector_cd = point_mid - point_c # cd
            # 用arctan2计算与x轴的夹角，范围[0, 360)
            theta_deg = np.degrees(np.arctan2(vector_cd[1], vector_cd[0]))
            rotate_angle = theta_deg - 360 if theta_deg > 180 else theta_deg
            deflect_angle = 90 - theta_deg # 将USB插头转正需要的偏转角

            logger.info(f"USB插头cd向量与x轴的偏转角: {rotate_angle}")
            logger.info(f"将USB插头转正需要的偏转角: {deflect_angle}")

            keyboard = input("请确认识别结果是否正确, 正确直接enter, 错误则e+enter")
            if keyboard == 'e':
                continue  # 重新识别
            else:
                break # 跳出循环，继续后续流程
        time.sleep(0.01)

    return rotate_angle,deflect_angle


def shifting_hands():

    '''
    Func: 倒手, 区分插头正反面
    '''
    
    step1_le = [-31.71, 67.21, 55.41, 33.97, -48.96, 92.49, 13.41]
    step1_ri = [3.06, -59.49, 6.16, -36.58, -6.35, -87.42, 32.14]
    dual_arm_move(step1_le,step1_ri,30)
    time.sleep(0.5)

    gripper_le.goTo(213) # 左爪松一点
    gripper_ri.goTo(255) # 右爪闭合
    
    '''
    右臂抓着线缆往下拽, 拽的过程中检测左臂的六维力数据
    '''
    joint_r, pose_r, _ = update_arm_ri_state() 
    p_end_pos = np.array((pose_r[:3])) # 更新当前右臂末端位置 in m
    p_end_ori = np.array((pose_r[3:])) # 更新当前右臂末端姿态 in rad
    arm_le_clear_force() # 左臂清空六维力数据
    drag_force_threshold  = 7.8 # 左臂拉拽力的阈值

    while not rospy.is_shutdown():

        p_end_pos[1] += 0.0005 # 每次下拽1mm
        q_target = np.array(robot_inverse_kinematic(joint_r, p_end_pos, p_end_ori))
        arm_ri.Movej_CANFD(q_target,False)

        _, _, drag_force_le = update_arm_left_state()
        logger.info(f"drag_force_le: {drag_force_le}")

        if drag_force_le[1] > drag_force_threshold: # 当拉拽力大于拉拽力阈值时，停止拉拽
            break
    
    gripper_le.goTo(255) # 左爪夹紧
    gripper_ri.goTo(0) # 右爪打开
    
    _, joint_le, _, _, _ = arm_le.Get_Current_Arm_State()
    _, joint_ri, _, _, _ = arm_ri.Get_Current_Arm_State()
    dual_arm_move(joint_le,joint_ri,30) #保证下一步左右臂同步运动
    
    front_or_back_le = [-37.5, 91.49, 58.99, 18.22, 23.48, 107.82, -0.9]
    front_or_back_ri = [-29.67, -78.19, 52.65, -38.93, -53.13, -77.53, 25.8]
    dual_arm_move(front_or_back_le,front_or_back_ri,30)

    rotate_angle,deflect_angle = get_target_xyz_ri() # USB插头cd向量与x轴的偏转角,将USB插头转正需要的偏转角
    
    # 需根据识别角度调整最后一个关节角
    step3_ri = [-25.23, -86.41, 52.63, -36.0, -58.63, -73.26, 29.93]
    step3_ri[6] += rotate_angle
    arm_ri.Movej_Cmd(step3_ri, 30)
    gripper_ri.goTo(255) # 右爪闭合
    gripper_le.goTo(0) # 左爪打开
    
    # 需根据识别角度调整最后一个关节角转到蓝色朝上
    step4_ri = step3_ri.copy()
    step4_ri[6] += deflect_angle
    arm_ri.Movej_Cmd(step4_ri, 30)
    gripper_le.goTo(255) # 左爪闭合
    gripper_ri.goTo(0) # 右爪打开
    
    # 右手以45度角抓住usb
    pick_pre_ri = [-0.7, -15.26, 7.62, -35.86, -3.11, -85.56, 26.24]
    arm_ri.Movej_Cmd(pick_pre_ri, 30) # 右臂撤走，防止打到左臂
    step5_le = [11.03, 61.97, 2.7, 18.14, -19.48, 103.21, 26.97]
    step5_ri = [-14.59, -77.91, 56.32, -32.24, -44.15, -84.9, 36.6]
    dual_arm_move(step5_le,step5_ri,30)
    gripper_ri.goTo(255)
    gripper_le.goTo(0)


def over_action_func(arm_ri):

    gripper_ri.goTo(0) # 右爪松开

    overjoint1_ri = [75.13, -84.71, -64.71, -41.35, -24.76, -103.14, 131.67]
    arm_ri.Movej_Cmd(overjoint1_ri,30)

    overjoint2_ri = [-0.64, 10.07, 7.63, -35.83, -4.98, -85.54, 26.24]
    arm_ri.Movej_Cmd(overjoint2_ri,30)


if __name__ == '__main__':
    
    logger_init()

    # =====================ROS=============================
    rospy.init_node('force_control_realman')
    rospy.Subscriber("/head_camera_predict_pointXYZ", PoseArray, headcam_all_hole_callback) # 移动，头部相机看所有的插孔
    flag_vision_searchHole = False # 视觉搜孔标志
    rospy.Subscriber("/right_camera_predict_pointXYZ", PoseArray, rightcam_usb_hole_callback) # 搜孔，右臂相机看USB插孔
    rospy.Subscriber("/left_camera_predict_usb_pointXYZ", PoseArray, leftcam_usb_plug_callback) # 抓取，左臂相机看USB插头
    rospy.Subscriber("/right_camera_predict_handusb_pointXYZ", PoseArray, rightcam_usb_pose_callback) # 倒手，右臂相机看USB正反
    kin_inverse = rospy.ServiceProxy('/kin_inverse', KinInverse)
    # rostopic pub 末端位姿和力传感器数据
    pub_pose_force_running = True
    publisher_thread = threading.Thread(target=pose_force_publisher)
    publisher_thread.start()
    # ======================================================

    # =====================机械臂连接========================
    byteIP_L = '192.168.99.17'
    byteIP_R = '192.168.99.18'
    arm_le = Arm(75,byteIP_L)
    arm_ri = Arm(75,byteIP_R)
    # 设置工具坐标系，同时需要更改pykin和udp上报force_coordinate设置
    # Arm_Tip 对应 Link7, Hand_Frame 对应 tool_hand, robotiq 对应 robotiq
    # arm_le.Change_Tool_Frame('Arm_Tip')
    # arm_le.Change_Tool_Frame('Hand_Frame')
    arm_le.Change_Tool_Frame('robotiq')
    # arm_ri.Change_Tool_Frame('Arm_Tip')
    # arm_ri.Change_Tool_Frame('Hand_Frame')
    arm_ri.Change_Tool_Frame('robotiq')
    arm_le.Change_Work_Frame('Base')
    arm_ri.Change_Work_Frame('Base')
    # 关闭左臂udp主动上报, ，force_coordinate=0 为传感器坐标系, 1 为当前工作坐标系, 2 为当前工具坐标系
    arm_le.Set_Realtime_Push(enable=True, cycle=1, force_coordinate=0)
    arm_ri.Set_Realtime_Push(enable=True, cycle=1, force_coordinate=0)
    arm_udp_status = RealtimePush_Callback(arm_udp_status)
    arm_le.Realtime_Arm_Joint_State(arm_udp_status)
    arm_ri.Realtime_Arm_Joint_State(arm_udp_status)
    # =======================================================

    # =====================初始化夹爪========================
    left_gripper_port, right_gripper_port = gripper_init()
    gripper_le = pyRobotiqGripper.RobotiqGripper(left_gripper_port)
    gripper_ri = pyRobotiqGripper.RobotiqGripper(right_gripper_port)
    gripper_le.goTo(0) #全开
    gripper_ri.goTo(0) #全开
    # ======================================================

    # =========================pykin========================
    pykin_le = SingleArm(f_name='./urdf/rm_75b_hand_gripper.urdf')
    pykin_le.setup_link_name('base_link', 'left_robotiq')
    # pykin_le.setup_link_name('base_link', 'Link7')
    # pykin_le.setup_link_name('base_link', 'tool_hand') 
    pykin_ri = SingleArm(f_name='./urdf/rm_75b_hand_gripper.urdf')
    pykin_ri.setup_link_name('base_link', 'right_robotiq')
    # pykin_ri.setup_link_name('base_link', 'Link7')
    # pykin_ri.setup_link_name('base_link', 'tool_hand') 
    pykin_fsensor = SingleArm(f_name='./urdf/rm_75b_hand_gripper.urdf')
    pykin_fsensor.setup_link_name('base_link', 'f_sensor') # 力传感器坐标系
    # ======================================================

    # =====================保存读取数据初始化=================
    start_time = time.time() # 记录程序开始时间
    # 记录整段轨迹的接触力信息在不同参考坐标系下
    force_data_inSensor_list = [] # 传感器坐标系下的接触力数据
    all_contact_force_inTcp = [] # 末端坐标系下的接触力数据
    all_contact_force_inBase = [] # 基座标系下的接触力数据
    joint_data_list = [] # 记录每次读取的关节角度
    pose_data_list = [] # 记录末端实际位姿
    p_end_list = [] # 记录末端期望位姿
    time_list = [] # 记录每次读取的时间
    # =======================================================

    # ===============设置 servo_j 内环位置控制参数=============
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

    # =====================读取外部力数据====================
    sensor = ForceSensorRS485(port='/dev/ttyUSB2') # 创建力传感器对象
    read_forcedata_thread = threading.Thread(target=read_forcedata, args=(sensor,))
    read_forcedata_thread.start()
    # ======================================================

    # ======================校准外部力数据====================
    calib_force_data_thread = threading.Thread(target=calib_force_data_func,args=(arm_ri,))
    calib_force_data_thread.start()
    time.sleep(0.5) # 等待线程启动
    logger.info("校准外部力数据线程启动")
    # ======================================================

    try:

        # ==========================主程序=========================
        servo_j_thread = threading.Thread(target=servo_j,args=(arm_ri,)) # 内环位置控制线程
        servo_j_thread.start()
        time.sleep(0.5)

        for i in range(20):

            input("enter to 抓取")
            cable_pick(arm_le) # 抓取线缆
            shifting_hands() # 换手

            input("enter to 机械臂移动到插接位置")
            cable_plugging_pre(arm_ri) # 移动到插孔位置
            time.sleep(1) 

            input("enter to 插接")
            force_data_zero() # 清零力传感器数据
            admittance_controller(arm_ri) # 导纳控制器

            input("enter to 结束动作")
            over_action_func(arm_ri) # 执行结束动作

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