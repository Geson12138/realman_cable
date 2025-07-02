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

# 全局变量储存插口pose
armcam_ri_pose = PoseArray()
leftcam_usb_pose = PoseArray()
rightcam_usb_pose = PoseArray()

def leftcam_usb_pose_callback(msg):
    """
    位置信息成功对出现，对数不固定
    leftcam_usb_pose.poses[0] 是usb1头部pose
    leftcam_usb_pose.poses[1] 是usb1尾部pose
    leftcam_usb_pose.poses[2] 是usb2头部pose
    leftcam_usb_pose.poses[3] 是usb2尾部pose
    ...以此类推
    """ 

    global leftcam_usb_pose
    leftcam_usb_pose = msg

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

def cable_pick():

    global leftcam_usb_pose
    proportion_k = 5
    point_a = np.array([leftcam_usb_pose.poses[0].position.x,leftcam_usb_pose.poses[0].position.y,leftcam_usb_pose.poses[0].position.z])
    point_b = np.array([leftcam_usb_pose.poses[1].position.x,leftcam_usb_pose.poses[1].position.y,leftcam_usb_pose.poses[1].position.z])
    point_c = point_b + proportion_k* (point_b- point_a)
    cam_graspPoint = np.ones(4); cam_graspPoint[:3] = point_c
    _, joint_l, pose_l, _, _ = arm_le.Get_Current_Arm_State()
    ee2base_transform_matrix = np.eye(4)
    ee2base_transform_matrix[:3, :3] = transform_utils.get_matrix_from_rpy([pose_l[3],pose_l[4],pose_l[5]])
    ee2base_transform_matrix[:3, 3] = [pose_l[0], pose_l[1], pose_l[2]]
    pose_out = ee2base_transform_matrix @ arm_cam2robotiq_left @ cam_graspPoint.reshape(4,1)
    print(f"pose_out:{pose_out}")

    vector_pointBA = point_a - point_b
    x_axis = np.array([1, 0, 0])
    cos_theta = np.dot(vector_pointBA, x_axis) / (np.linalg.norm(vector_pointBA) * np.linalg.norm(x_axis))
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    theta_deg = np.degrees(theta_rad)
    print("BA与x轴夹角:", theta_deg)

    joint_l[6] += (90- theta_deg)
    arm_le.Movej_Cmd(joint_l,10)


def get_target_xyz_ri():

    '''
    Func: 计算右臂相机坐标下目标插头偏转角, 用于区分正反面和做偏转
    Params:
        rightcam_usb_pose.poses USB插头在右臂相机坐标下的点的位置
    Return: 
        theta_deg USB插头与右臂相机坐标x轴(水平方向)的偏转角 in degree
    '''

    global rightcam_usb_pose
    point_a = np.array([rightcam_usb_pose.poses[0].position.x,rightcam_usb_pose.poses[0].position.y,rightcam_usb_pose.poses[0].position.z]) # 两个角点之一 a
    point_b = np.array([rightcam_usb_pose.poses[1].position.x,rightcam_usb_pose.poses[1].position.y,rightcam_usb_pose.poses[1].position.z]) # 两个角点之一 b 
    point_c = np.array([rightcam_usb_pose.poses[2].position.x,rightcam_usb_pose.poses[2].position.y,rightcam_usb_pose.poses[2].position.z]) # 蓝色的点 c
    point_mid = (point_a + point_b)/ 2 # 两个角点的中点 d
    vector_cd = point_mid - point_c # cd
    x_axis = np.array([1, 0, 0]) # 水平方向的方向向量
    cos_theta = np.dot(vector_cd, x_axis) / (np.linalg.norm(vector_cd) * np.linalg.norm(x_axis))
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    theta_deg = np.degrees(theta_rad)
    print("BA与x轴夹角:", theta_deg)


if __name__ == '__main__':
    
    # =====================ROS=============================
    rospy.init_node('test_graspUSB')
    rospy.Subscriber("/left_camera_predict_usb_pointXYZ", PoseArray, leftcam_usb_pose_callback)
    rospy.Subscriber("/right_camera_predict_handusb_pointXYZ", PoseArray, rightcam_usb_pose_callback) # 倒手，右臂相机看USB正反
    # =====================机械臂连接========================
    byteIP_L ='192.168.99.17'
    byteIP_R = '192.168.99.18'
    arm_le = Arm(75,byteIP_L)
    arm_ri = Arm(75,byteIP_R)
    arm_le.Change_Tool_Frame('robotiq')
    arm_ri.Change_Tool_Frame('robotiq')
    arm_le.Change_Work_Frame('Base')
    arm_ri.Change_Work_Frame('Base')

    while not rospy.is_shutdown():
        get_target_xyz_ri()
        time.sleep(1)