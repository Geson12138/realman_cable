import numpy as np
from lib.robotic_arm import *
from pykin.robots.single_arm import SingleArm
import pykin.utils.transform_utils  as transform_utils
import requests
import time
from cv_bridge import CvBridge
from std_msgs.msg import Float32
import rospy
from sensor_msgs.msg import Image
import cv2
import math
from pykin.utils.transform_utils import get_matrix_from_quaternion
from spatialmath import SE3
import serial
import serial.tools.list_ports
import time
import math
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Ro
from include.predefine_pose import *


#滤波处理
def filter_pose(pose):
    pose = np.apply_along_axis(savgol_filter,axis=1,arr=pose,window_length=5,polyorder=2)
    return pose

def get_target_xyz(trans,markid,markerwidth):
    rospy.init_node('get_wrench_images',anonymous=True)
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
            
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, markerwidth, cameraMatrix, distCoeffs)
            print(f'旋转：{rvecs}')
            print(f'平移：{tvecs}')
            # for rvec_, tvec_ in zip(rvecs, tvecs):
            #     frame_axes = cv2.drawFrameAxes(color_img, cameraMatrix, distCoeffs, rvec_, tvec_, 1)
            
            #转换为位姿矩阵
            rvec = rvecs.reshape(3)
            tvec = tvecs.reshape(3)
            # print(rvec)
            break
    return rvec, tvec


if __name__ == '__main__':
    rotate_list_x = []
    rotate_list_y = []
    rotate_list_z = []
    trans_list_x = []
    trans_list_y = []
    trans_list_z = []
    for i in range(100):
        tran1 = SE3.Trans(np.array([0, 0, 0]))
        rvec, tvec = get_target_xyz(tran1,11,9.5)
        rotate_list_x.append(rvec[0])
        rotate_list_y.append(rvec[1])
        rotate_list_z.append(rvec[2])
        trans_list_x.append(tvec[0])
        trans_list_y.append(tvec[1])
        trans_list_z.append(tvec[2])
        print(i)
    print(f'max_rot_x:{max(rotate_list_x)} max_rot_y:{max(rotate_list_y)} max_rot_z:{max(rotate_list_z)}')
    print(f'min_rot_x:{min(rotate_list_x)} min_rot_y:{min(rotate_list_y)} min_rot_z:{min(rotate_list_z)}')
    print(f'max_rot_x:{max(trans_list_x)} max_rot_y:{max(trans_list_y)} max_rot_z:{max(trans_list_z)}')
    print(f'min_rot_x:{min(trans_list_x)} min_rot_y:{min(trans_list_y)} min_rot_z:{min(trans_list_z)}')

    
        