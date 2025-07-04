import numpy as np
from lib.robotic_arm import *
from pykin.robots.single_arm import SingleArm
import pykin.utils.transform_utils  as transform_utils
from include.predefine_pose import *
import time
import argparse
from spatialmath import SE3
from spatialmath import SO3

def arm_inverse_kine(pykin_arm,refer_joints,target_pose):

    '''
    Func: 逆运动学求解关节角
    Params: 
        refer_joints 迭代法求解的参考关节角
        target_pose 末端的目标位姿矩阵
    Return: arm_joint 求解的关节角 in degree
    '''

    r_qd = pykin_arm.inverse_kin([j/180*np.pi for j in refer_joints], target_pose, method="LM", max_iter=100)
    arm_joint = [j/np.pi*180 for j in r_qd]
    print(f'arm_joint: {arm_joint}')

    return arm_joint

def parse_args():

    parser = argparse.ArgumentParser(description="输入在末端坐标系下的位置偏置")
    parser.add_argument('arm', choices=['l', 'r', 'L', 'R'], help='选择左臂(l)或右臂(r)')
    parser.add_argument('-x', default=0.0, type=float, help='x方向前进量')
    parser.add_argument('-y', default=0.0, type=float, help='y方向前进量')
    parser.add_argument('-z', default=0.0, type=float, help='z方向前进量')
    parser.add_argument('-rx', default=0.0, type=float, help='rx方向旋转量')
    parser.add_argument('-ry', default=0.0, type=float, help='ry方向旋转量')
    parser.add_argument('-rz', default=0.0, type=float, help='rz方向旋转量')

    return parser.parse_args()

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

    # =========================pykin========================
    pykin_le = SingleArm(f_name='./urdf/rm_75b_hand_gripper.urdf')
    pykin_le.setup_link_name('base_link', 'left_robotiq')

    pykin_ri = SingleArm(f_name='./urdf/rm_75b_hand_gripper.urdf')
    pykin_ri.setup_link_name('base_link', 'right_robotiq')

    pykin_fsensor = SingleArm(f_name='./urdf/rm_75b_hand_gripper.urdf')
    pykin_fsensor.setup_link_name('base_link', 'f_sensor') # 力传感器坐标系
    # ======================================================

    args = parse_args()

    if args.arm in ['l', 'L']:
        arm = arm_le
        pykin = pykin_le
    elif args.arm in ['r', 'R']:
        arm = arm_ri
        pykin = pykin_ri
    else:
        print("只能选择左臂l L或者右臂r R")

    bias_vector_inTcp = SE3.Trans(args.x,args.y,args.z)* SE3.RPY(args.rx/180*np.pi,args.ry/180*np.pi,args.rz/180*np.pi)
    
    _, cur_joint, ee_pose, _, _ = arm.Get_Current_Arm_State()
    print(f"cur_joint: {cur_joint}")
    print(f"ee_pose: {ee_pose}")
    
    cur_pose_matrix = SE3.Trans(ee_pose[0],ee_pose[1],ee_pose[2])* SE3.RPY(ee_pose[3],ee_pose[4],ee_pose[5])
    bias_vector_inBase = cur_pose_matrix * bias_vector_inTcp
    print(f"bias_vector_inBase: {bias_vector_inBase}")
          
    des_joint = arm_inverse_kine(pykin, cur_joint, bias_vector_inBase.A)

    arm.Movej_Cmd(des_joint,10)












    

