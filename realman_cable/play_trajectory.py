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
    portName=None
    for port in ports:
        try:
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
            portName=port.device
            del device

            ser.close()  # Close the port
        except:
            pass  # Skip if port cannot be opened

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
    print(f"arm_joint: {arm_joint}")

    return arm_joint

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

def dual_arm_move(joints_left, joints_right, speed=15):

    thread_le = threading.Thread(target=arm_le.Movej_Cmd, args=(joints_left, speed))
    thread_ri = threading.Thread(target=arm_ri.Movej_Cmd, args=(joints_right, speed))
    thread_le.start()
    thread_ri.start()
    thread_le.join()
    thread_ri.join()

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
    time.sleep(0.03)
    put_target = put_pose @ put_Tx
    arm_joints = arm_inverse_kine(pykin_ri,refer_joints,put_target)
    robot.Movej_Cmd(arm_joints,30)

if __name__=='__main__':

    rospy.init_node('play_trajectory')

    #---------------------连接机器人----------------------
    byteIP_L ='192.168.99.17'
    byteIP_R = '192.168.99.18'
    arm_ri = Arm(75,byteIP_R)
    arm_le = Arm(75,byteIP_L)
    arm_le.Change_Tool_Frame('robotiq')
    arm_ri.Change_Tool_Frame('robotiq')
    arm_le.Change_Work_Frame('Base')
    arm_ri.Change_Work_Frame('Base')

    # =========================pykin========================
    pykin_le = SingleArm(f_name='./urdf/rm_75b_hand_gripper.urdf')
    pykin_le.setup_link_name('base_link', 'left_robotiq')
    pykin_ri = SingleArm(f_name='./urdf/rm_75b_hand_gripper.urdf')
    pykin_ri.setup_link_name('base_link', 'right_robotiq')
    pykin_fsensor = SingleArm(f_name='./urdf/rm_75b_hand_gripper.urdf')
    pykin_fsensor.setup_link_name('base_link', 'f_sensor') # 力传感器坐标系
    # ======================================================

    # =====================初始化夹爪========================
    gripper_init()
    left_gripper_port = "/dev/ttyUSB1"
    right_gripper_port = "/dev/ttyUSB0"
    gripper_le = pyRobotiqGripper.RobotiqGripper(left_gripper_port)
    gripper_ri = pyRobotiqGripper.RobotiqGripper(right_gripper_port)
    gripper_le.goTo(255) #全开
    gripper_ri.goTo(255) #全开
    # ======================================================


    input("enter to 机械臂移动到插接位置")
    cable_plugging_pre(arm_ri) # 移动到插孔位置
    time.sleep(1) 

    input("enter to play trajectory")

    # 读取 JSON 文件
    with open('./saved_data/trajectory_data_real.json', 'r') as f:
        joint_angle_list = json.load(f)

    # 依次播放每一组关节角
    for idx, joint_angles in enumerate(joint_angle_list):

        arm_ri.Movej_CANFD(joint_angles,False,0) # 透传目标位置给机械臂
        time.sleep(0.050)  # 每组间隔10ms