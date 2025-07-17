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

headcam_all_hole = PoseArray()

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

    if (msg.poses[0].position.x == 0 and msg.poses[0].position.y == 0 and msg.poses[0].position.z == 0) \
    or (msg.poses[1].position.x == 0 and msg.poses[1].position.y == 0 and msg.poses[1].position.z == 0) \
    or (msg.poses[2].position.x == 0 and msg.poses[2].position.y == 0 and msg.poses[2].position.z == 0) \
    or (msg.poses[3].position.x == 0 and msg.poses[3].position.y == 0 and msg.poses[3].position.z == 0):
        
        print("识别失败")

    else: 

        headcam_all_hole = msg.poses[:3]
        print('-------------------------------------')
        print(f"headcam_all_hole[0].position.x: {headcam_all_hole[0].position.x}")
        print(f"headcam_all_hole[0].position.y: {headcam_all_hole[0].position.y}")
        print(f"headcam_all_hole[0].position.z: {headcam_all_hole[0].position.z}")
        print(f"headcam_all_hole[1].position.x: {headcam_all_hole[1].position.x}")
        print(f"headcam_all_hole[1].position.y: {headcam_all_hole[1].position.y}")
        print(f"headcam_all_hole[1].position.z: {headcam_all_hole[1].position.z}")

def compute_normal_vector():

    global headcam_all_hole

    # 定义3D世界点 (矩形平面) 长5cm 宽4cm
    width = 0.04; height=0.05
    world_points = np.float32([
        [0.0,    0.0,      0.0], 
        [0.0,    -height,  0.0], 
        [width,  -height,  0.0], 
        [width,  0.0,      0.0]
    ])

    # 像素坐标
    image_points = np.float32([
        [headcam_all_hole[0].position.x,headcam_all_hole[0].position.y],
        [headcam_all_hole[1].position.x,headcam_all_hole[1].position.y],
        [headcam_all_hole[2].position.x,headcam_all_hole[2].position.y],
        [headcam_all_hole[3].position.x,headcam_all_hole[3].position.y]
    ])  
    
    cameraMatrix = np.array([[389.541259765625, 0.0, 320.45037841796875],
                            [0.0, 388.84527587890625, 239.62796020507812],
                            [0.0, 0.0, 1.0]],np.float32)
    
    # 解PnP
    _, rvec, tvec = cv2.solvePnP(
        world_points, image_points, cameraMatrix, None
    )
    
    # 旋转向量转矩阵
    R, _ = cv2.Rodrigues(rvec)
    normal = R[:, 2]  # 相机坐标系下的法向量
    
    # 方向校正
    plane_center = np.array([width/2, height/2, 0])
    plane_center_camera = R.dot(plane_center) + tvec.flatten()
    view_dir = -plane_center_camera
    
    if np.dot(normal, view_dir) < 0:
        normal = -normal
        
    return normal / np.linalg.norm(normal)  # 单位化


        


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


if __name__ == '__main__':

    # =====================ROS=============================
    rospy.init_node('replace_aruco')
    rospy.Subscriber("/head_camera_predict_pointXYZ", PoseArray, headcam_all_hole_callback) # 移动，头部相机看所有的插孔

    trans = np.array([0, 0, 0])
    put_pose = get_target_xyz_head_aruco(SE3.Trans(trans),11,9.5)

    rospy.spin()