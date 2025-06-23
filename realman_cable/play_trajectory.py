import json
import time
import numpy as np
from lib.robotic_arm import *
from pykin.robots.single_arm import SingleArm
import pykin.utils.transform_utils  as transform_utils
from include.predefine_pose import *

if __name__=='__main__':

    #---------------------连接机器人----------------------
    byteIP_L ='192.168.99.17'
    byteIP_R = '192.168.99.18'
    arm_ri = Arm(75,byteIP_R)
    arm_le = Arm(75,byteIP_L)

    # 读取 JSON 文件
    with open('./saved_data/trajectory_data_real.json', 'r') as f:
        joint_angle_list = json.load(f)

    # 依次播放每一组关节角
    for idx, joint_angles in enumerate(joint_angle_list):

        arm_ri.Movej_CANFD(joint_angles,False,0) # 透传目标位置给机械臂
        time.sleep(0.050)  # 每组间隔10ms