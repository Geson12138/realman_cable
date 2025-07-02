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
import pyRobotiqGripper

# 创建一个logger
logger = logging.getLogger('test_force_udp')

def logger_init():
    
    logger.setLevel(logging.INFO) # Log等级总开关
    # 清除上一次的logger
    if logger.hasHandlers():
        logger.handlers.clear()

    if not os.path.exists('./logfiles'): # 创建log目录
        os.mkdir('./logfiles')

    logfile = './logfiles/test_force_udp.log' # 创建一个handler，用于写入日志文件
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

def arm_le_clear_force():

    global leftArm_force_zero
    _, _, leftArm_force_zero = update_arm_left_state()

if __name__ == '__main__':
    
    logger_init()

    # =====================机械臂连接========================
    byteIP_L ='192.168.99.17'
    byteIP_R = '192.168.99.18'
    arm_ri = Arm(75,byteIP_R)
    arm_le = Arm(75,byteIP_L)
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

    leftArm_force_zero = np.zeros(6)

    # 关闭左臂udp主动上报, ，force_coordinate=0 为传感器坐标系, 1 为当前工作坐标系, 2 为当前工具坐标系
    arm_le.Set_Realtime_Push(enable=True, cycle=1, force_coordinate=0)
    arm_ri.Set_Realtime_Push(enable=True, cycle=1, force_coordinate=0)

    arm_udp_status = RealtimePush_Callback(arm_udp_status)
    arm_le.Realtime_Arm_Joint_State(arm_udp_status)
    arm_ri.Realtime_Arm_Joint_State(arm_udp_status)

    time.sleep(2)

    arm_le_clear_force() # 左臂清空六维力数据
    logger.info(f"leftArm_force_zero: {leftArm_force_zero}")

    # arm_le.Realtime_Arm_Joint_State(arm_udp_status)
    # arm_ri.Realtime_Arm_Joint_State(arm_udp_status)
    # time.sleep(1)

    for i in range(50):

        _, _, drag_force_le = update_arm_left_state()
        logger.info(f"drag_force_le: {drag_force_le}")
        # _, _, drag_force_ri = update_arm_ri_state()
        # logger.info(f"drag_force_ri: {drag_force_ri}") 
        # joint_l, _, _ = update_arm_left_state()
        # logger.info(f"joint_l: {joint_l}")

        # time.sleep(0.05)

    # arm_le.Realtime_Arm_Joint_State(arm_le_status)
    # time.sleep(1)

    # for i in range(1000):

    #     _, _, drag_force_le = update_arm_left_state()
    #     logger.info(f"drag_force_le: {drag_force_le}")

    #     time.sleep(0.05)


