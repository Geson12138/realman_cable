import socket
import json
import struct
import time
import threading
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from typing import List,Tuple
import logging
from logging.handlers import RotatingFileHandler
from lib.robotic_arm import *
from spatialmath import SE3
from read_forcedata import ForceSensorRS485


# 全局变量用于存储力传感数据和时间戳，最大长度为4000
force_data_list = deque(maxlen=1000)
time_data_list = deque(maxlen=1000)

# 全局变量用于控制程序运行

def read_forcedata(sensor):
    """
    读取力传感器数据的线程函数
    """
    # 启动数据流并测量频率
    sensor.save_forcedata()

save_running = True

def save_forcedata(sensor):

    global force_data_list, time_data_list, save_running
    st_time = time.time()
    print("开始保存力传感器数据...")
    while save_running:

        # 将传感器坐标系下的力转换到末端坐标系下，注意此时受力坐标系为末端坐标系
        force2tcp = SE3.Trans(0.0, 0.0, -0.199) * SE3.RPY(1.57, 2.356, 0.0, order='xyz') # 如何从末端工具坐标系转换到力传感器坐标系，注意不是另一种
        force_data_inTcp = trans_force_data(force2tcp.A, sensor.forcedata) # 6*1
        # print("Force Data in TCP:", force_data_inTcp)
        force_data_list.append(force_data_inTcp)
        time_data_list.append(time.time()-st_time)  # 记录相对时间
        time.sleep(0.01)  # 每10毫秒读取一次数据

plot_running = True

def on_close(event):

    global plot_running,save_running
    plot_running = False
    save_running = False
    plt.close(event.canvas.figure)

def plot():
    
    plt.ion()  # 开启交互模式
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.canvas.mpl_connect('close_event', on_close)
    
    lines1 = [ax1.plot([], [], label=label, color=color)[0] for label, color in zip(['fx', 'fy', 'fz'], ['r', 'g', 'b'])]
    lines2 = [ax2.plot([], [], label=label, color=color)[0] for label, color in zip(['mx', 'my', 'mz'], ['r', 'g', 'b'])]
    
    ax1.set_xlim(0, 20)  # 显示最近20秒的数据
    ax1.set_ylim(-10, 10)
    ax1.legend()
    
    ax2.set_xlim(0, 20)  # 显示最近20秒的数据
    ax2.set_ylim(-10, 10)
    ax2.legend()
    
    while plot_running:

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
    

time_stamp = time.time()

# 注册机械臂状态主动上报回调函数
def robotstatus (data):

    st_time = time.time()
    print(f"时间6: {time.time()-st_time}")
    print("RobotStatus RobotStatus RobotStatus")
    print("当前角度:", data.joint_status.joint_position[0], data.joint_status.joint_position[1],
    data.joint_status.joint_position[2])
    print("当前力:", data.force_sensor.force[0], data.force_sensor.force[1],
    data.force_sensor.force[2])
    # print("err:", data.errCode)
    global time_stamp
    # print("time_eplapse:", time.time() - time_stamp) # 打印上报周期
    time_stamp = time.time()
    global force_data_list, time_data_list
    force_data = np.array(data.force_sensor.zero_force)
    cur_joints = np.array(data.joint_status.joint_position)
    cur_ee_pos = np.array([data.waypoint.position.x, data.waypoint.position.y, data.waypoint.position.z])
    cur_ee_ori = np.array([data.waypoint.euler.rx, data.waypoint.euler.ry, data.waypoint.euler.rz])
    # print(type(cur_ee_pos), type(cur_joints), type(force_data), type(cur_ee_ori))
    cur_ee_pose = np.concatenate((cur_ee_pos, cur_ee_ori))
    # print("Force Sensor Data:", force_data)
    # print("Current Joints:", cur_joints)
    # print("Current End Pose:",cur_ee_pose)
    # 将传感器坐标系下的力转换到末端坐标系下，注意此时受力坐标系为末端坐标系
    force2tcp = SE3.Trans(0.0, 0.0, -0.199) * SE3.RPY(0, 2.356, 1.57) # 如何从末端工具坐标系转换到力传感器坐标系，注意不是另一种
    force_data_inTcp = trans_force_data(force2tcp.A, force_data) # 6*1
    print("Force Data in TCP:", force_data_inTcp)
    force_data_list.append(force_data_inTcp)
    time_data_list.append(time_stamp)


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


if __name__ == "__main__":

    # byteIP_L ='192.168.99.17'
    # byteIP_R = '192.168.99.18'
    # arm_ri = Arm(75,byteIP_R)
    # arm_le.Change_Tool_Frame('Arm_Tip')
    # arm_le = Arm(75,byteIP_L)
    # 设置工具坐标系，同时需要更改pykin和udp上报force_coordinate设置
    # Arm_Tip 对应 Link7, Hand_Frame 对应 tool_hand, robotiq 对应 robotiq
    # arm_le.Change_Work_Frame('Base')
    # arm_le.Change_Tool_Frame('Hand_Frame')
    # arm_ri.Change_Tool_Frame('Arm_Tip')
    # arm_ri.Change_Tool_Frame('Hand_Frame')
    # arm_ri.Change_Tool_Frame('robotiq')
    # arm_ri.Change_Work_Frame('Base')
    #关闭左臂udp主动上报, force_coordinate=2代表上报工具坐标系受力数据
    # arm_le.Set_Realtime_Push(enable=False)
    # arm_ri.Set_Realtime_Push(enable=True, cycle=1, force_coordinate=0)
    # robotstatus = RealtimePush_Callback(robotstatus)
    # arm_ri.Realtime_Arm_Joint_State(robotstatus)

    # try:

    #     while True:

    #         time.sleep(1)

    # except KeyboardInterrupt:

    #     arm_ri.Arm_Socket_Close()
    #     print("程序已终止")

    try:
        sensor = ForceSensorRS485(port='/dev/ttyUSB1')

        read_forcedata_thread = threading.Thread(target=read_forcedata, args=(sensor,))
        read_forcedata_thread.start()

        save_forcedata_thread = threading.Thread(target=save_forcedata, args=(sensor,))
        save_forcedata_thread.start()
        
    
        plot() # 画图
        sensor.stop_stream()
        sensor.ser.close()

    except KeyboardInterrupt:
        
        print("程序已终止")
        save_running = False
        plot_running = False
        read_forcedata_thread.join()
        save_forcedata_thread.join()

        sensor.stop_stream()
        sensor.ser.close()
        