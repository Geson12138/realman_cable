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
from srv._ChakouXYZ import *
from include.predefine_pose import *
from geometry_msgs.msg import PoseArray

point6_pose = PoseArray()

def point6_pose_callback(msg):
    global point6_pose
    point6_pose = msg

def get_target_xyz():
    global point6_pose
    pose_out = np.eye(4)
    rpy = np.array(point6_pose.poses.orientation.x,point6_pose.poses.orientation.y,point6_pose.poses.orientation.z)
    rotation_matrix = np.array(transform_utils.get_matrix_from_quaternion(rpy))
    pose_out[:3,:3] = rotation_matrix
    pose_out[0,3] = point6_pose.poses.position.x
    pose_out[1,3] = point6_pose.poses.position.y
    pose_out[2,3] = point6_pose.poses.position.z
    print(pose_out)
    ################################################################
    # pose_put = head_cam2base_left @ pose_put
    pose_out = head_cam2base_right @ pose_out
    print('base下目标位置:',pose_out[:3,3])
    print('当前base下目标的旋转矩阵',pose_out)

    return pose_out

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
    rospy.init_node('test_m0')
    #####################################################################
    rospy.Subscriber("/head_camera_predict_euler_t", PoseArray, point6_pose_callback)


    #---------------------连接机器人----------------------
    byteIP_L ='192.168.99.17'
    byteIP_R = '192.168.99.18'
    arm_ri = Arm(75,byteIP_R)
    arm_le = Arm(75,byteIP_L)

    arm_le.Change_Tool_Frame('robotiq')
    arm_ri.Change_Tool_Frame('robotiq')

    arm_le.Change_Work_Frame('Base')
    arm_ri.Change_Work_Frame('Base')

    tran1 = SE3.Trans(np.array([0, 0, 0]))

    input("开始示教计算")
    
    while True:
        pose = get_target_xyz()
        ##############################################################################
        err_code, joint_l, pose_l, arm_err_ptr, sys_err_ptr = arm_ri.Get_Current_Arm_State()
        print('joint:', [round(float(j), 2) for j in joint_l])
        print('pose_l:',pose_l)
        maxt = get_po_maxtr(pose_l[:3],pose_l[3:])
        print("maxt",maxt)
        
        result = np.linalg.inv(pose) @ maxt
        print("result",result)
        input("enter to continue")    


