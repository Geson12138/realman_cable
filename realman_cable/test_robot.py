import numpy as np
from lib.robotic_arm import *
from pykin.robots.single_arm import SingleArm
import pykin.utils.transform_utils  as transform_utils
from include.predefine_pose import *
import time
import ikpy
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

if __name__=='__main__':

    #---------------------连接机器人----------------------
    byteIP_L = '192.168.99.17'
    byteIP_R = '192.168.99.18'

    arm_le = Arm(75,byteIP_L)
    arm_ri = Arm(75,byteIP_R)

    arm_le.Change_Tool_Frame('robotiq')
    arm_ri.Change_Tool_Frame('robotiq')

    arm_le.Change_Work_Frame('Base')
    arm_ri.Change_Work_Frame('Base')

    # input("enter to continue")

    pick_pre_le = [-45.15, 84.27, 64.16, 65.31, 21.57, 76.74, -94.29]
    arm_le.Movej_Cmd(pick_pre_le,10) 

    pick_pre_ri = [-0.64, 10.07, 7.63, -35.83, -4.98, -85.54, 26.24]
    arm_ri.Movej_Cmd(pick_pre_ri,10)

    step1_ri = [2.56, -59.42, 6.56, -36.24, -6.61, -87.46, 35.11]
    arm_ri.Movej_Cmd(step1_ri,10)

    input("enter to continue")

    step1_le = [-31.72, 67.27, 55.5, 33.97, -50.26, 92.33, 11.29]
    arm_le.Movej_Cmd(step1_le,10) 





    
