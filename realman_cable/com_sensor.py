from numpy import *
import numpy as np
import math
import threading
from lib.robotic_arm import *
from read_forcedata import ForceSensorRS485 as SensorData
from scipy.spatial.transform import Rotation as R
import rospy
from matplotlib import pyplot as plt
import time
import copy
import queue
'''
Made by 水木皆Ming
重力补偿计算
'''
class GravityCompensation:

    pri_data = np.array([0,0,0,0,0,0])
    
    M = np.empty((0, 0))
    F = np.empty((0, 0))
    f = np.empty((0, 0))
    R = np.empty((0, 0))

    x=  0.008725100139216051
    y=  -4.5511809495531186e-05
    z=  0.0758334179376014
    k1=  0.7793016394885708
    k2=  -0.8607796482220615
    k3=  0.11344380200307311
    g=  0.3151078097925119
    U=  -3.2169786261607856
    V=  72.7832850054421
    F_x0=  6.417249003945723
    F_y0=  1.0358374617831787
    F_z0=  1.305842451929013

    M_x0 = 0
    M_y0 = 0
    M_z0 = 0

    F_ex = 0
    F_ey = 0
    F_ez = 0

    M_ex = 0
    M_ey = 0
    M_ez = 0

    com_force = np.zeros(6, dtype=float)
    def Update_M(self, torque_data):
        M_x = torque_data[0]
        M_y = torque_data[1]
        M_z = torque_data[2]

        if (any(self.M)):
            M_1 = matrix([M_x, M_y, M_z]).transpose()
            self.M = vstack((self.M, M_1))
        else:
            self.M = matrix([M_x, M_y, M_z]).transpose()

    def Update_F(self, force_data):
        F_x = force_data[0]
        F_y = force_data[1]
        F_z = force_data[2]

        if (any(self.F)):
            F_1 = matrix([[0, F_z, -F_y, 1, 0, 0],
                          [-F_z, 0, F_x, 0, 1, 0],
                          [F_y, -F_x, 0, 0, 0, 1]])
            self.F = vstack((self.F, F_1))
        else:
            self.F = matrix([[0, F_z, -F_y, 1, 0, 0],
                             [-F_z, 0, F_x, 0, 1, 0],
                             [F_y, -F_x, 0, 0, 0, 1]])

    def Solve_A(self):
        A = dot(dot(linalg.inv(dot(self.F.transpose(), self.F)), self.F.transpose()), self.M)

        self.x = A[0, 0]
        self.y = A[1, 0]
        self.z = A[2, 0]
        self.k1 = A[3, 0]
        self.k2 = A[4, 0]
        self.k3 = A[5, 0]
        # print("A= \n" , A)
        print("x= ", self.x)
        print("y= ", self.y)
        print("z= ", self.z)
        print("k1= ", self.k1)
        print("k2= ", self.k2)
        print("k3= ", self.k3)

    def Update_f(self, force_data):
        F_x = force_data[0]
        F_y = force_data[1]
        F_z = force_data[2]

        if (any(self.f)):
            f_1 = matrix([F_x, F_y, F_z]).transpose()
            self.f = vstack((self.f, f_1))
        else:
            self.f = matrix([F_x, F_y, F_z]).transpose()

    def Update_R(self, euler_data):
        # 机械臂末端到基坐标的旋转矩阵
        R_array = self.eulerAngles2rotationMat(euler_data)

        alpha = 2.617

        # 力传感器到末端的旋转矩阵
        R_alpha = np.array([[math.cos(alpha), -math.sin(alpha), 0],
                            [math.sin(alpha), math.cos(alpha), 0],
                            [0, 0, 1]
                            ])

        R_array = np.dot(R_alpha, R_array.transpose())

        if (any(self.R)):
            R_1 = hstack((R_array, np.eye(3)))
            self.R = vstack((self.R, R_1))
        else:
            self.R = hstack((R_array, np.eye(3)))

    def Solve_B(self):
        B = dot(dot(linalg.inv(dot(self.R.transpose(), self.R)), self.R.transpose()), self.f)

        self.g = math.sqrt(B[0] * B[0] + B[1] * B[1] + B[2] * B[2])
        self.U = math.asin(-B[1] / self.g)
        self.V = math.atan(-B[0] / B[2])

        self.F_x0 = B[3, 0]
        self.F_y0 = B[4, 0]
        self.F_z0 = B[5, 0]

        # print("B= \n" , B)
        print("g= ", self.g / 9.81)
        print("U= ", self.U * 180 / math.pi)
        print("V= ", self.V * 180 / math.pi)
        print("F_x0= ", self.F_x0)
        print("F_y0= ", self.F_y0)
        print("F_z0= ", self.F_z0)

    def Solve_Force(self, force_data, euler_data):
        Force_input = matrix([force_data[0], force_data[1], force_data[2]]).transpose()

        my_f = matrix([cos(self.U)*sin(self.V)*self.g, -sin(self.U)*self.g, -cos(self.U)*cos(self.V)*self.g, self.F_x0, self.F_y0, self.F_z0]).transpose()

        R_array = self.eulerAngles2rotationMat(euler_data)
        R_array = R_array.transpose()
        R_1 = hstack((R_array, np.eye(3)))

        Force_ex = Force_input - dot(R_1, my_f)

        self.com_force[0] = Force_ex[0]
        self.com_force[1] = Force_ex[1]
        self.com_force[2] = Force_ex[2]

        # print('接触力：', Force_ex)

    def Solve_Torque(self, torque_data, euler_data):
        Torque_input = matrix([torque_data[0], torque_data[1], torque_data[2]]).transpose()
        M_x0 = self.k1 - self.F_y0 * self.z + self.F_z0 * self.y
        M_y0 = self.k2 - self.F_z0 * self.x + self.F_x0 * self.z
        M_z0 = self.k3 - self.F_x0 * self.y + self.F_y0 * self.x

        Torque_zero = matrix([M_x0, M_y0, M_z0]).transpose()

        Gravity_param = matrix([[0, -self.z, self.y],
                                [self.z, 0, -self.x],
                                [-self.y, self.x, 0]])

        Gravity_input = matrix([cos(self.U)*sin(self.V)*self.g, -sin(self.U)*self.g, -cos(self.U)*cos(self.V)*self.g]).transpose()

        R_array = self.eulerAngles2rotationMat(euler_data)
        R_array = R_array.transpose()

        Torque_ex = Torque_input - Torque_zero - dot(dot(Gravity_param, R_array), Gravity_input)

        self.com_force[3] = Torque_ex[0]
        self.com_force[4] = Torque_ex[1]
        self.com_force[5] = Torque_ex[2]
        # print('接触力矩：', Torque_ex.T)

    def eulerAngles2rotationMat(self, theta):
        theta = [i * math.pi / 180.0 for i in theta]  # 角度转弧度

        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]), math.cos(theta[0])]
                        ])

        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0, 1, 0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])

        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])

        # 第一个角为绕X轴旋转，第二个角为绕Y轴旋转，第三个角为绕Z轴旋转
        R = np.dot(R_x, np.dot(R_y, R_z))
        return R


    def get_com_data(self,sensor,robot_arm, set_zero=False):
        try:

            cur_data= sensor.forcedata
            force_data=cur_data[:3]
            torque_data=cur_data[3:]
            err_code, joint_r, robot_pose, _, _=robot_arm.Get_Current_Arm_State()
            euler_degree = robot_pose[3:6]  # 获取当前末端执行器的欧拉角
            self.Solve_Force(force_data,euler_degree)
            self.Solve_Torque(torque_data,euler_degree)

        except KeyboardInterrupt:
            rospy.signal_shutdown('KeyboardInterrupt')
        if set_zero:
            return self.com_force- self.pri_data
        else:
            return self.com_force

def plot_data(sensor_data, time_list):
    plt.clf()  # 清除当前图像
    force_data = np.array(sensor_data)
    time_data = np.array(time_list)
    # 确保time_data和force_data的长度一致
    min_length = min(len(time_data), len(force_data))
    time_data = time_data[:min_length]
    force_data = force_data[:min_length]
    
    plt.plot(time_data, force_data[:, 0], label='Fx')
    plt.plot(time_data, force_data[:, 1], label='Fy')
    plt.plot(time_data, force_data[:, 2], label='Fz')
    # plt.plot(time_data, force_data[:, 3], label='Mx')
    # plt.plot(time_data, force_data[:, 4], label='My')
    # plt.plot(time_data, force_data[:, 5], label='Mz')
    plt.legend()
    # plt.draw()  # 使用draw()而不是pause()来刷新图像
    plt.pause(0.01)  # 暂停以更新图像

    
def read_forcedata(sensor):
    """
    读取力传感器数据的线程函数
    """
    sensor.save_forcedata()

def robot_read(ft_sensor, data_queue, stop_event):
    robot_ip = '192.168.99.18'
    arm_ri = Arm(75, robot_ip)
    arm_ri.Change_Tool_Frame('robotiq')
    arm_ri.Change_Work_Frame('Base')
    gravity_comp = GravityCompensation()
    start_time = time.time()
    time_list = []
    force_data = []
    joint_init = [60.36, -88.26, -51.22, -40.0, 138.2, 77.82, -40.63]
    arm_ri.Movej_Cmd(joint_init, 10)
    time.sleep(1)

    while not rospy.is_shutdown() and not stop_event.is_set():
        try:
            sensor_data = gravity_comp.get_com_data(ft_sensor, arm_ri, set_zero=True)
            if np.linalg.norm(sensor_data) > 50:
                continue
            force_data.append(copy.deepcopy(sensor_data))
            time_list.append(time.time() - start_time)
            # 只把数据放到队列，不做绘图
            data_queue.put((copy.deepcopy(force_data), copy.deepcopy(time_list)))
            time.sleep(0.1)
        except KeyboardInterrupt:
            rospy.signal_shutdown('KeyboardInterrupt')
            stop_event.set()
            break

def plot_loop(data_queue, stop_event):
    plt.ion()
    force_data = []
    time_list = []
    while not stop_event.is_set():
        try:
            # 只取最新一组数据
            while not data_queue.empty():
                force_data, time_list = data_queue.get_nowait()
            if force_data and time_list:
                plot_data(force_data, time_list)
            plt.pause(0.01)
            #按esc键退出
            if plt.waitforbuttonpress(0.1):
                if plt.get_current_fig_manager().canvas.key == 'escape':
                    stop_event.set()
                    break
        except Exception as e:
            print("Plot error:", e)
            break
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    ft_sensor = SensorData()
    data_queue = queue.Queue()
    stop_event = threading.Event()
    read_forcedata_thread = threading.Thread(target=read_forcedata, args=(ft_sensor,))
    robot_read_thread = threading.Thread(target=robot_read, args=(ft_sensor, data_queue, stop_event))
    read_forcedata_thread.start()
    robot_read_thread.start()
    try:
        plot_loop(data_queue, stop_event)  # 主线程绘图
    except KeyboardInterrupt:
        stop_event.set()
    read_forcedata_thread.join()
    robot_read_thread.join()