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

# 手眼标定矩阵
# T_camera2leftbase = [[ 0.01869706, -0.40052846,  0.91609354,  0.13454828],
#                    [ 0.01227538,  0.91627661,  0.40035796, -0.24771904],
#                    [-0.99974984,  0.00375988,  0.0220483, -0.01920365],
#                    [ 0.,          0.,          0.,          1.        ]]


# T_camera2leftbase = [[-0.00775612, -0.32200265,  0.94670699,  0.15620742],
#                     [-0.00348822,  0.94673842,  0.32198477, -0.19900207],
#                     [-0.99996384, -0.00080497, -0.00846623, -0.01890053],
#                     [ 0.,          0.,          0.,          1.        ]]

# T_camera2rightbase = [[-0.01921201,  0.42771819, -0.90370794, -0.14084976],
#                    [ 0.03918901,  0.90350232,  0.42678775, -0.24625343],
#                    [ 0.99904711, -0.02721597, -0.03411995, -0.00085085],
#                    [ 0.,          0.,          0.,          1.        ]]
#

# T_camera2rightbase = [[-0.01736682,  0.36009764, -0.93275296, -0.1610298 ],
#                     [ 0.00929465,  0.93291149,  0.35998578, -0.20111395],
#                     [ 0.99980598, -0.0024178,  -0.01954868, -0.00523836],
#                     [ 0.,          0.,          0.,          1.        ]]


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
            if time_count >= 5:
                # pose_filter = filter_pose(pose_all)
                # print('pose_filter:',np.mean(pose_filter,axis=1))
                
                pose_mean = np.mean(pose_all,axis=1)
                pose_put[:3,3] = pose_mean
                print('pose_mean:',pose_mean)
                ################################################################
                pose_put = head_cam2base_right @ pose_put
                # pose_put = T_camera2rightbase @ pose_put
                print('base下目标位置:',pose_put[:3,3])
                print('当前base下目标的旋转矩阵',pose_put)
                break
            # 按 'Esc' 退出循环/home/
            if cv2.waitKey(1) & 0xFF == 27:  # 27是Esc键的ASCII值
                print(f'pose_put[:3,3]/100: {pose_put[:3,3]/100}')
                break

    return pose_put

#转为旋转矩阵
def euler2rot(euler):
    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]
    r = Ro.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    return r.as_matrix()

#获得位姿矩阵
def get_po_maxtr(t,euler):
    pose = np.zeros((4,4))
    maxtrix = euler2rot(euler)
    pose[:3,:3] = maxtrix
    pose[:3,3] = t
    pose[3,3] = 1
    return pose

if __name__ == '__main__':

    #---------------------连接机器人----------------------
    byteIP_L ='192.168.99.17'
    byteIP_R = '192.168.99.18'
    arm_ri = Arm(75,byteIP_R)
    arm_le = Arm(75,byteIP_L)

    arm_le.Change_Tool_Frame('robotiq')
    arm_ri.Change_Tool_Frame('robotiq')

    arm_le.Change_Work_Frame('Base')
    arm_ri.Change_Work_Frame('Base')

    # 回到home
    arm_le.Set_Hand_Speed(200)
    arm_ri.Set_Hand_Speed(200)
    arm_le.Set_Hand_Force(500)
    arm_ri.Set_Hand_Force(500)
    # arm_ri.Set_Hand_Angle([1000, 1000, 1000, 1000, 1000, 1000])
    #-----------------------------初始--------------------------

    # arm_le.Set_Hand_Angle([1000,1000,1000,1000,1000,0])
    #-------------------一次视觉识别
    time.sleep(3)
    tran1 = SE3.Trans(np.array([0, 0, 0]))

    input("开始示教计算")
    
    while True:
        pose = get_target_xyz(tran1,11,9.5)
        ##############################################################################
        err_code, joint_l, pose_l, arm_err_ptr, sys_err_ptr = arm_ri.Get_Current_Arm_State()
        print('pose_l:',pose_l)
        maxt = get_po_maxtr(pose_l[:3],pose_l[3:])
        print("maxt",maxt)
        
        result = np.linalg.inv(pose) @ maxt
        print("result",result)
        input("enter to continue")