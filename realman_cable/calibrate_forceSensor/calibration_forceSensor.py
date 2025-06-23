'''
参考链接： https://blog.csdn.net/qq_34935373/article/details/106208345
参考文章: 基于六维力传感器的工业机器人末端负载受力感知研究_张立建
'''
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import matplotlib.pyplot as plt
import time
import socket
import threading
import json
import concurrent.futures
# from lib.robotcontrol import Auboi5Robot, RobotError, RobotErrorType, RobotEventType, RobotEvent, RobotMoveTrackType, RobotCoordType, RobotIOType, RobotUserIoName
from pykin.robots.single_arm import SingleArm
from pykin.utils import transform_utils as transform_utils
from datetime import datetime
from collections import deque
from scipy.signal import butter, filtfilt
from lib.robotic_arm import *
from typing import List,Tuple
from read_forcedata import ForceSensorRS485

# 创建一个logger
logger = logging.getLogger('calibration_forceSensor')

def logger_init():
    
    logger.setLevel(logging.INFO) # Log等级总开关
    if not os.path.exists('./logfiles'): # 创建log目录
        os.mkdir('./logfiles')
    logfile = './logfiles/robot-ctl-python.log' # 创建一个handler，用于写入日志文件
    fh = RotatingFileHandler(logfile, mode='a', maxBytes=1024*1024*50, backupCount=30) # 以append模式打开日志文件
    fh.setLevel(logging.INFO) # 输出到file的log等级的开关
    ch = logging.StreamHandler() # 再创建一个handler，用于输出到控制台
    ch.setLevel(logging.INFO) # 输出到console的log等级的开关
    formatter = logging.Formatter("%(asctime)s [%(thread)u] %(levelname)s: %(message)s") # 定义handler的输出格式
    fh.setFormatter(formatter) # 为文件输出设定格式
    ch.setFormatter(formatter) # 控制台输出设定格式
    logger.addHandler(fh) # 设置文件输出到logger
    logger.addHandler(ch) # 设置控制台输出到logger

def read_forcedata(sensor):
    """
    读取力传感器数据的线程函数
    """
    # 启动数据流并测量频率
    sensor.save_forcedata()


## -----------------------力传感器标定参数-----------------------
force_zero = np.zeros(6) # fx0 fy0 fz0 Mx0 My0 Mz0 力传感器零点
G_force = np.zeros(6) # Gx Gy Gz Mgx Mgy Mgz 负载重力在力传感器坐标系下的分量
gravity_bias = np.zeros(3) # x y z 负载重心在力传感器坐标系下的分量
k_1 = 0; k_2 = 0; k_3 = 0 # 中间计算系数
mass = 0 # 负载重力

## -----------------------------------------------------------
obs_M = np.zeros(3).reshape(3,1) # 观测力矩 3*1
obs_Ms = [] # 观测力矩矩阵 3N*1
obs_F = np.hstack((np.zeros((3,3)),np.eye(3))) # 观测力 3*6
obs_Fs = [] # 观测力矩阵 3N*6
## -----------------------------------------------------------
obs_F1 = np.zeros(3).reshape(3,1) # 观测力 3*1
obs_Fs1 = [] # 观测力矩阵 3N*1
obs_R = np.hstack((np.zeros((3,3)),np.eye(3))) # 观测力 3*6
obs_Rs = [] # 观测力矩阵 3N*6


def least_squares(A, b):

    """
    使用最小二乘法求解线性方程组 Ax = b
    :param A: 系数矩阵 m*n
    :param b: 结果向量 m*1
    :return: 求解得到的 x 向量 n*1
    """

    # 使用 numpy 的 lstsq 函数求解最小二乘问题
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return x

force_data = np.zeros(6) # fx fy fz Mx My Mz 原始力传感器数据
calib_force_data = np.zeros(6) # 标定后的力数据
force_zero_point = np.zeros(6) # 力传感器数据清零

def online_calibrate_force(sensor):
    '''
    FUNCTION: 在线力传感器标定函数，最小二乘法标定
    '''
    
    count_point = 0 # 标定点的个数
    json_filename = 'calibration_data.json'
    
    # 打开 JSON 文件用于写入，并清空之前的内容
    with open(json_filename, 'w') as json_file: # 写入模式
        json.dump([], json_file)  # 写入一个空列表，表示 JSON 数组的开始
    
    with open(json_filename, 'a') as json_file: # 追加模式

        while True:

            key_in = input("按enter开始标定,按e+enter退出标定")

            if key_in in ['e', 'E']:
                break
            
            '''
            获取力传感器数据，需要根据不同的传感器型号和通信协议获取
            '''
            logger.info('获取的力传感数值: {}'.format(sensor.forcedata))
            force_data = sensor.forcedata.copy() # 6*1 力传感器数据 [fx, fy, fz, Mx, My, Mz]

            obs_M[0] = force_data[3]; obs_M[1] = force_data[4]; obs_M[2] = force_data[5]
            obs_Ms.append(obs_M.copy())

            obs_F[0,1] = force_data[2]; obs_F[0,2] = -force_data[1]
            obs_F[1,0] = -force_data[2]; obs_F[1,2] = force_data[0]
            obs_F[2,0] = force_data[1]; obs_F[2,1] = -force_data[0]
            obs_Fs.append(obs_F.copy())

            obs_F1[0] = force_data[0]; obs_F1[1] = force_data[1]; obs_F1[2] = force_data[2]
            obs_Fs1.append(obs_F1.copy())

            '''
            获取机械臂当前的力传感器坐标系的姿态, 需要根据不同的机械臂型号和通信协议获取
            '''
            _, _, pose_r, _, _ = arm_ri.Get_Current_Arm_State()
            rotation_matrix = np.array(transform_utils.get_matrix_from_rpy(pose_r[3:]))
            inv_rotation_matrix = np.linalg.inv(rotation_matrix)
            obs_R[:3,:3] = inv_rotation_matrix
            obs_Rs.append(obs_R.copy())
            logger.info('获取的力传感器坐标系的姿态矩阵: {}'.format(inv_rotation_matrix))

            count_point += 1
            logger.info("标定点数：{}".format(count_point))

            json_data = {
                'count_point': count_point,
                'force_data': force_data.tolist(),  # 力传感器数据 [fx, fy, fz, Mx, My, Mz]
                'rotation_matrix': inv_rotation_matrix.tolist()
            }

            json_file.seek(json_file.tell() - 1, os.SEEK_SET)  # 移动到文件末尾的前一个字符位置
            json_file.write(',\n')  # 写入逗号和换行符
            json.dump(json_data, json_file)
            json_file.write(']')  # 写入 JSON 数组的结束符
    
    if count_point >3: # 标定点数要求大于3，至少4个点，且方向向量不共面

        # 将观测力矩和观测力矩阵转换为 numpy 数组
        obs_Ms_np = np.vstack(obs_Ms)
        obs_Fs_np = np.vstack(obs_Fs)

        obs_F1s_np = np.vstack(obs_Fs1)
        obs_Rs_np = np.vstack(obs_Rs)

        # 使用最小二乘法求解
        result = least_squares(obs_Fs_np, obs_Ms_np)
        # logger.info("求解得到的系数向量:", result)
        gravity_bias[0] = result[0]; gravity_bias[1] = result[1]; gravity_bias[2] = result[2]
        logger.info("负载重心在力传感器坐标系下的坐标:x: {}, y: {}, z: {}".format(gravity_bias[0], gravity_bias[1], gravity_bias[2]))
        k_1 = result[3]; k_2 = result[4]; k_3 = result[5]

        # 使用最小二乘法求解
        result1 = least_squares(obs_Rs_np, obs_F1s_np)
        mass = np.sqrt(result1[0]**2 + result1[1]**2 + result1[2]**2)
        logger.info("负载重力:{}".format(mass))
        force_zero[0] = result1[3]; force_zero[1] = result1[4]; force_zero[2] = result1[5] # 传感器零点 fx0 fy0 fz0
        logger.info("传感器零点:fx0: {}, fy0: {}, fz0: {}".format(force_zero[0], force_zero[1], force_zero[2]))
        force_zero[3] = k_1 - force_zero[1]*gravity_bias[2] + force_zero[2]*gravity_bias[1]
        force_zero[4] = k_2 - force_zero[2]*gravity_bias[0] + force_zero[0]*gravity_bias[2]
        force_zero[5] = k_3 - force_zero[0]*gravity_bias[1] + force_zero[1]*gravity_bias[0]
        logger.info("传感器零点:Mx0: {}, My0: {}, Mz0: {}".format(force_zero[3], force_zero[4], force_zero[5]))

        # 保存标定结果到 JSON 文件
        calibration_result = {
            'gravity_bias': gravity_bias.tolist(),
            'mass': mass.tolist(),
            'force_zero': force_zero.tolist()
        }
        with open('calibration_result.json', 'w') as result_file:
            json.dump(calibration_result, result_file, indent=4)
        logger.info('保存标定结果到calibration_result.json文件')

    else:
        logger.info("标定点数不足，无法标定")

'''
FUNCTION: 已经有了calibration_data.json文件，离线标定力传感器
'''
def offline_calibrate_force():

    obs_Ms = []
    obs_Fs = []
    obs_Fs1 = []
    obs_Rs = []

    # 读取 JSON 文件
    with open('calibration_data.json', 'r') as json_file:
        whole_data = json.load(json_file)
        for json_data in whole_data:
            force_data = json_data['force_data']
            inv_rotation_matrix = json_data['rotation_matrix']

            obs_M = np.array(force_data[3:]).reshape(3, 1)
            obs_Ms.append(obs_M)

            obs_F = np.hstack((np.zeros((3, 3)), np.eye(3)))
            obs_F[0, 1] = force_data[2]
            obs_F[0, 2] = -force_data[1]
            obs_F[1, 0] = -force_data[2]
            obs_F[1, 2] = force_data[0]
            obs_F[2, 0] = force_data[1]
            obs_F[2, 1] = -force_data[0]
            obs_Fs.append(obs_F)

            obs_F1 = np.array(force_data[:3]).reshape(3, 1)
            obs_Fs1.append(obs_F1)

            obs_R = np.hstack((np.zeros((3, 3)), np.eye(3)))
            obs_R[:3, :3] = np.array(inv_rotation_matrix)
            obs_Rs.append(obs_R)

    if len(obs_Ms) > 3:
        obs_Ms_np = np.vstack(obs_Ms)
        obs_Fs_np = np.vstack(obs_Fs)
        obs_F1s_np = np.vstack(obs_Fs1)
        obs_Rs_np = np.vstack(obs_Rs)

        result = least_squares(obs_Fs_np, obs_Ms_np)
        gravity_bias[0] = result[0]
        gravity_bias[1] = result[1]
        gravity_bias[2] = result[2]
        logger.info("负载重心在力传感器坐标系下的坐标:x: {}, y: {}, z: {}".format(gravity_bias[0], gravity_bias[1], gravity_bias[2]))
        k_1 = result[3]
        k_2 = result[4]
        k_3 = result[5]

        result1 = least_squares(obs_Rs_np, obs_F1s_np)
        mass = np.sqrt(result1[0] ** 2 + result1[1] ** 2 + result1[2] ** 2)
        logger.info("负载重力:{}".format(mass))
        force_zero[0] = result1[3]
        force_zero[1] = result1[4]
        force_zero[2] = result1[5]
        logger.info("传感器零点:fx0: {}, fy0: {}, fz0: {}".format(force_zero[0], force_zero[1], force_zero[2]))
        force_zero[3] = k_1 - force_zero[1] * gravity_bias[2] + force_zero[2] * gravity_bias[1]
        force_zero[4] = k_2 - force_zero[2] * gravity_bias[0] + force_zero[0] * gravity_bias[2]
        force_zero[5] = k_3 - force_zero[0] * gravity_bias[1] + force_zero[1] * gravity_bias[0]
        logger.info("传感器零点:Mx0: {}, My0: {}, Mz0: {}".format(force_zero[3], force_zero[4], force_zero[5]))

        # 保存标定结果到 JSON 文件
        calibration_result = {
            'gravity_bias': gravity_bias.tolist(),
            'mass': mass.tolist(),
            'force_zero': force_zero.tolist()
        }
        with open('calibration_result.json', 'w') as result_file:
            json.dump(calibration_result, result_file, indent=4)
        logger.info('保存标定结果到calibration_result.json文件')

    else:
        logger.info("标定点数不足，无法标定")

'''
验证力传感器标定结果
'''
def validate_calibrate_result(sensor):

    global force_data_list, time_data_list, running

    # 读取 calibration_result.JSON 文件
    with open('calibration_result.json', 'r') as json_file:
        json_data = json.load(json_file)
        gravity_bias = np.array(json_data['gravity_bias'])
        mass = np.array(json_data['mass'][0])
        force_zero = np.array(json_data['force_zero'])

    alpha = 0.9
    smoothed_force_data = np.array(sensor.forcedata).copy()
    time_count = 0

    while running:

        raw_force_data = np.array(sensor.forcedata).copy()

        # 移动均值滤波
        smoothed_force_data = alpha * raw_force_data  + (1-alpha) * smoothed_force_data

        '''
        获取机械臂当前的力传感器坐标系姿态, 需要根据不同的机械臂型号和通信协议获取
        '''
        _, _, pose_r, _, _ = arm_ri.Get_Current_Arm_State()
        rotation_matrix = np.array(transform_utils.get_matrix_from_rpy(pose_r[3:]))
        inv_rotation_matrix = np.linalg.inv(rotation_matrix) # 3*3
        temp_gravity_vector = np.array([0, 0, -1]).reshape(3, 1) # 3*1
        gravity_vector = np.dot(inv_rotation_matrix, temp_gravity_vector) # 3*1
        G_force[:3] = np.transpose(mass * gravity_vector) # 3*1 Gx Gy Gz
        G_force[3] = G_force[2] * gravity_bias[1] - G_force[1] * gravity_bias[2] # Mgx = Gz × y − Gy × z 
        G_force[4] = G_force[0] * gravity_bias[2] - G_force[2] * gravity_bias[0] # Mgy = Gx × z − Gz × x
        G_force[5] = G_force[1] * gravity_bias[0] - G_force[0] * gravity_bias[1] # Mgz = Gy × x − Gx × y

        # 标定后的力数据 = 原始力传感器数据 - 力传感器零点 - 负载重力分量
        calib_force_data = smoothed_force_data - force_zero - G_force # 6*1
        # calib_force_data = smoothed_force_data - G_force 
        print('原始数据: ',smoothed_force_data)
        # print('重力分量: ',G_force)
        # print('传感器零点：',force_zero)
        print('标定后的外力数据:', calib_force_data)
        
        current_time = time.time()
        force_data_list.append(calib_force_data)
        time_data_list.append(current_time)

        time_count += 1 
        # if time_count % 2 == 0:
        #     print('calib_force_data:', calib_force_data[1])

        time.sleep(0.001) # 5ms 一次

    sensor.stop_stream()
    sensor.ser.close()

# 全局变量用于存储力传感数据和时间戳，最大长度为4000
force_data_list = deque(maxlen=1000)
time_data_list = deque(maxlen=1000)
# 全局变量用于控制程序运行
running = True

def on_close(event):
    global running
    running = False
    plt.close(event.canvas.figure)

def plot_external_force():
    
    global running
    
    plt.ion()  # 开启交互模式
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.canvas.mpl_connect('close_event', on_close)
    
    lines1 = [ax1.plot([], [], label=label, color=color)[0] for label, color in zip(['fx', 'fy', 'fz'], ['r', 'g', 'b'])]
    lines2 = [ax2.plot([], [], label=label, color=color)[0] for label, color in zip(['mx', 'my', 'mz'], ['r', 'g', 'b'])]
    
    ax1.set_xlim(0, 20)  # 显示最近20秒的数据
    ax1.set_ylim(-20, 20)
    ax1.set_ylabel('F/N')
    ax1.legend()
    
    ax2.set_xlim(0, 20)  # 显示最近20秒的数据
    ax2.set_ylim(-10, 10)
    ax2.set_xlabel('Time/s')
    ax2.set_ylabel('M/Nm')
    ax2.legend()
    
    while running:

        if len(force_data_list) == len(time_data_list) and force_data_list:
            data = np.array(force_data_list)  # 转换为numpy数组
            times = np.array(time_data_list) - time_data_list[0]  # 转换为相对时间
            
            if len(times) == data.shape[0]:
                for i, line in enumerate(lines1):
                    line.set_xdata(times)
                    line.set_ydata(data[:, i])
                
                for i, line in enumerate(lines2):
                    line.set_xdata(times)
                    line.set_ydata(data[:, i + 3])
                
                ax1.set_xlim(0, times[-1])
                ax1.set_ylim(np.min(data[:, :3]) - 1, np.max(data[:, :3]) + 1)
                
                ax2.set_xlim(0, times[-1])
                ax2.set_ylim(np.min(data[:, 3:]) - 1, np.max(data[:, 3:]) + 1)
                
                plt.draw()
                plt.pause(0.01)
    
if __name__ == "__main__":

    logger_init()
    # =====================机器人连接=========================
    byteIP_L ='192.168.99.17'
    byteIP_R = '192.168.99.18'
    arm_ri = Arm(75,byteIP_R)
    arm_le = Arm(75,byteIP_L)
    # 设置工具坐标系，同时需要更改pykin和udp上报force_coordinate设置
    # Arm_Tip 对应 Link7, Hand_Frame 对应 tool_hand, robotiq 对应 robotiq
    # 在标定中需要获取传感器坐标系的姿态
    arm_le.Change_Tool_Frame('Arm_Tip')
    # arm_le.Change_Tool_Frame('Hand_Frame')
    # arm_ri.Change_Tool_Frame('Arm_Tip')
    # arm_ri.Change_Tool_Frame('Hand_Frame')
    arm_ri.Change_Tool_Frame('f_sensor')
    arm_le.Change_Work_Frame('Base')
    arm_ri.Change_Work_Frame('Base')
    
    # 单开一个线程读取原始力传感器数据
    sensor = ForceSensorRS485(port='/dev/ttyUSB1')

    read_forcedata_thread = threading.Thread(target=read_forcedata, args=(sensor,))
    read_forcedata_thread.start()
    logger.info("连接力传感器成功，读取原始力传感器数据线程启动")

    '''
    没有calibration_force.json文件, 执行在线力传感器标定
    '''
    # logger.info("连接机器人和力传感器，执行在线力传感器标定")
    # online_calibrate_force(sensor) # 在线执行力传感器标定

    '''
    读取保存的calibration_data.json文件, 离线力传感器标定
    '''
    # logger.info("不连接机器人和力传感器，执行离线力传感器标定")
    # offline_calibrate_force() # 离线执行力传感器标定

    '''
    标定完成，验证标定结果
    '''
    # 单开一个线程获取外部力传感器数据，验证标定结果
    validate_thread = threading.Thread(target=validate_calibrate_result, args=(sensor,))
    validate_thread.start()
    logger.info("连接力传感器成功，获取外部力传感器数据线程启动")
    plot_external_force() # 绘制外部力
