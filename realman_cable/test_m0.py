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

class ServoUtils:
    def __init__(self) -> None:
        hot = '/dev/ttyUSB0'
        bps = 115200
        timex = 0.1
        self.ser = serial.Serial(hot, bps, timeout=timex)
        self.angle1 = 0
        self.angle2 = 0
        

    def read_angle(self):
        
        return self.angle1, self.angle2
        
        
    def write_angle(self, angle1, angle2):
        # angle = int(angle/0.18)
        self.angle1 = angle1
        self.angle2 = angle2
        angle1 = 1500+int(angle1/0.09)
        angle2 = 1500+int(angle2/0.09)
        # if id == 1:
        if angle1<1200:
            angle1 = 1200
        if angle1>2000:
            angle1 = 2000
        
        if angle2<500:
            angle2 = 500
        if angle2>2500:
            angle2 = 2500
            # hex_angle = hex(angle)
        # print('dasdda')
        command = '{#000P'+str(angle1)+'T0100!#015P'+str(angle2)+'T0100!}'
        command = command.encode('utf-8')
        self.ser.write(command)

    def move_angle(self, move_angle1, move_angle2):
        if move_angle1 == 0 and move_angle2==0:
            return
        try:
            current_angle_1, current_angle_2 = self.read_angle()
            target_angle_1, target_angle_2 = current_angle_1+move_angle1, current_angle_2+move_angle2
            print('target: ',target_angle_1, target_angle_2)
            self.write_angle(target_angle_1, target_angle_2)
        except Exception:
            print('servo error!')

# a=ServoUtils()

# try:
#     # print(a.read_angle(1))
#     # print(a.read_angle(2))
#     a.write_angle(0,-20)
#     # time.sleep(1)
#     # a.move_angle(30,30)
#     # a.write_angle(2,90)
#     print(a.read_angle())
#     # print(a.read_angle(2))
#     # print(s.read_angle(1))
# except Exception as e:
#     a.ser.close()
#     print(e)

# 手眼标定矩阵


# T_camera2leftbase = [[-0.00775612, -0.32200265,  0.94670699,  0.15620742],
#                     [-0.00348822,  0.94673842,  0.32198477, -0.19900207],
#                     [-0.99996384, -0.00080497, -0.00846623, -0.01890053],
#                     [ 0.,          0.,          0.,          1.        ]]


# T_camera2rightbase = [[-0.01736682,  0.36009764, -0.93275296, -0.1610298 ],
#                     [ 0.00929465,  0.93291149,  0.35998578, -0.20111395],
#                     [ 0.99980598, -0.0024178,  -0.01954868, -0.00523836],
#                     [ 0.,          0.,          0.,          1.        ]]


#滤波处理
def filter_pose(pose):
    pose = np.apply_along_axis(savgol_filter,axis=1,arr=pose,window_length=5,polyorder=2)
    return pose

point6_pose = PoseArray()

def point6_pose_callback(msg):
    global point6_pose
    point6_pose = msg

def get_target_xyz():
    global point6_pose
    pose_out = np.eye(4)
    # chakou_position_handle = rospy.ServiceProxy('chakou_predict', ChakouXYZ)
    # pose_out[0,3] = chakou_position_handle().point2[0]
    # pose_out[1,3] = chakou_position_handle().point2[1]
    # pose_out[2,3] = chakou_position_handle().point2[2]
    pose_out[0,3] = point6_pose.poses[5].position.x
    pose_out[1,3] = point6_pose.poses[5].position.y
    pose_out[2,3] = point6_pose.poses[5].position.z
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
    rospy.Subscriber("/predict_pointXYZ", PoseArray, point6_pose_callback)


    #---------------------连接机器人----------------------
    byteIP_L ='192.168.99.17'
    byteIP_R = '192.168.99.18'
    arm_ri = Arm(75,byteIP_R)
    arm_le = Arm(75,byteIP_L)

    arm_le.Change_Tool_Frame('Hand_Frame')
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


