import numpy as np
from lib.robotic_arm import *
from pykin.robots.single_arm import SingleArm
import pykin.utils.transform_utils  as transform_utils
import requests
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

    step5_ri = [55.72, -83.69, -64.77, -31.44, -30.58, -82.46, 131.75]
    arm_ri.Movej_Cmd(step5_ri,15)
    
