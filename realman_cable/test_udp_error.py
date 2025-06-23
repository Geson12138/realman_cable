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

# 全局变量用于存储力传感数据和时间戳，最大长度为4000
force_data_list = deque(maxlen=4000)
time_data_list = deque(maxlen=4000)

monitor_running = True
    
def monitor_loop():

    # # 机械臂 ip
    # ip = '192.168.99.18'
    # port_no = 8080
    # # Create a socket and connect to the server
    # client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # client.connect((ip, port_no))
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # # sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 新增此行
    # sock.bind(("192.168.99.100", 8099))
    
    while monitor_running:
        
        st_time = time.time()

        try:
            data, addr = sock.recvfrom(1024)
            print(f"Time taken to receive data: {time.time() - st_time:.4f} seconds")
            message = json.loads(data)
            print(f"Time taken to parse JSON: {time.time() - st_time:.4f} seconds")
            # logger.info(f"Received message from {addr}: {data}")
            
            #处理六维力传感器数据
            force_sensor = message.get("six_force_sensor", {})
            raw_zero_force = force_sensor.get("zero_force", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
            processed_zero_force = [0.0] * 6
            for i in range(6):
                try:
                    processed_zero_force[i] = float(raw_zero_force[i]/1000) if i < len(raw_zero_force) else 0.0
                except (ValueError, TypeError):
                    processed_zero_force[i] = 0.0
            
            force_data_list.append(processed_zero_force)
            time_data_list.append(time.time())

            print(f"Received data: {processed_zero_force}, Time: {time.time() - st_time:.4f} seconds")      
                    
        except (json.JSONDecodeError, socket.error):
            print("数据解析错误")
            continue
            
    sock.close()


# 全局变量用于控制程序运行
running = True

def on_close(event):
    global running, monitor_running
    running = False
    monitor_running = False
    plt.close(event.canvas.figure)

def main():
    
    read_thread = threading.Thread(target=monitor_loop, args=())
    read_thread.start()
    
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
    
    read_thread.join()
    arm_ri.Arm_Socket_Close()

if __name__ == "__main__":

    byteIP_L ='192.168.99.17'
    byteIP_R = '192.168.99.18'
    arm_ri = Arm(75,byteIP_R)
    # arm_le.Change_Tool_Frame('Arm_Tip')
    # arm_le = Arm(75,byteIP_L)
    # 设置工具坐标系，同时需要更改pykin和udp上报force_coordinate设置
    # Arm_Tip 对应 Link7, Hand_Frame 对应 tool_hand, robotiq 对应 robotiq
    # arm_le.Change_Work_Frame('Base')
    # arm_le.Change_Tool_Frame('Hand_Frame')
    # arm_ri.Change_Tool_Frame('Arm_Tip')
    # arm_ri.Change_Tool_Frame('Hand_Frame')
    arm_ri.Change_Tool_Frame('robotiq')
    arm_ri.Change_Work_Frame('Base')
    #关闭左臂udp主动上报, force_coordinate=2代表上报工具坐标系受力数据
    # arm_le.Set_Realtime_Push(enable=False)
    # arm_ri.Set_Realtime_Push(enable=True, cycle=1, force_coordinate=0)

    main()