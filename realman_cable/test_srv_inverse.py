import numpy as np
from lib.robotic_arm import *
from pykin.robots.single_arm import SingleArm
import pykin.utils.transform_utils  as transform_utils
import requests
from include.predefine_pose import *
import time
from srv._KinInverse import *
import rospy

if __name__=='__main__':

    kin_inverse = rospy.ServiceProxy('KinInverse', KinInverse)
    refer_joints = [48.643001556396484, -83.36499786376953, -58.14899826049805, -28.052000045776367, 135.4759979248047, 71.65499877929688, -40.444000244140625]
    target_pose = [-0.64808387, -0.00523004,  0.30859018,-1.59899998, -0.208 , -3.125]
    # st_time = time.time()
    # rsp = kin_inverse(refer_joints, target_pose)
    # print(f'time {time.time()-st_time}')
    # print(rsp.joint_angles)
    
    ave_st_time = time.time()
    for i in range (1000):
        target_pose[0] += 0.00005
        st_time = time.time()
        rsp = kin_inverse(refer_joints, target_pose)
        print(f'time {time.time()-st_time}')
        i += 1
            
    print(f'average time {(time.time()-ave_st_time)/1000}')
