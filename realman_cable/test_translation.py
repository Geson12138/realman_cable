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
from spatialmath import SE3
from spatialmath import SO3

def robot_inverse_kinematic(refer_joints,target_pos,target_rot):
    '''
    逆运动学求解关节角驱动机械臂运动
    :param target_pos: 末端期望位置 / m
    :param target_ori_rpy_rad 末端期望角度rqy / degree
    :return: 逆运动学求解的关节角度 / rad
    '''
    
    target_pose = np.eye(4);target_pose[:3, :3] = target_rot; target_pose[:3, 3] = target_pos
    r_qd = pykin_ri.inverse_kin([j/180*np.pi for j in refer_joints], target_pose, method="LM", max_iter=20)
    arm_joint = [j/np.pi*180 for j in r_qd]

    return arm_joint


if __name__=='__main__':

    #---------------------连接机器人----------------------
    byteIP_L ='192.168.99.17'
    byteIP_R = '192.168.99.18'
    arm_ri = Arm(75,byteIP_R)
    arm_le = Arm(75,byteIP_L)

    # arm_le.Change_Tool_Frame('Arm_Tip')
    arm_le.Change_Tool_Frame('Hand_Frame')
    arm_ri.Change_Tool_Frame('robotiq')
    # arm_ri.Change_Tool_Frame('f_sensor')

    arm_le.Change_Work_Frame('Base')
    arm_ri.Change_Work_Frame('Base')

    # =========================pykin=========================
    pykin_ri = SingleArm(f_name='./urdf/rm_75b_hand_gripper.urdf')
    pykin_ri.setup_link_name('base_link', 'robotiq')

    move_period = 0.03 # 50ms

    for i in range(200):

        st_time = time.perf_counter()
        delta_theta =  -0.3/180*np.pi  # 每周期旋转的弧度
        rotate_y_in_fingertip = SE3.RPY(0.0, delta_theta, 0.0)

        _, joint_r, pose_r, _, _ = arm_ri.Get_Current_Arm_State()

        # 当前末端在base下的位姿
        second_phase_tcpPose = SE3.Trans(pose_r[0], pose_r[1], pose_r[2]) * SE3.RPY(pose_r[3], pose_r[4], pose_r[5])
        # 指尖在末端下的变换
        fingertip_in_tcp = SE3.Trans(0.018, 0.0, 0.0) * SE3.RPY(0.0, 0.0, 0.0)
        # 末端在指尖下的变换（逆变换）
        tcp_in_fingertip = fingertip_in_tcp.inv()

        # 先变换到指尖坐标系，再绕y轴旋转，再变回末端坐标系
        new_tcp_pose = second_phase_tcpPose * fingertip_in_tcp * rotate_y_in_fingertip * tcp_in_fingertip

        new_pose = new_tcp_pose.A  # 如果需要numpy数组

        p_end_pos = new_pose[:3,3] 
        p_end_rot = new_pose[:3, :3]

        arm_joint = robot_inverse_kinematic(joint_r,p_end_pos,p_end_rot)

        arm_ri.Movej_CANFD(arm_joint,False)

        elapsed_time = time.perf_counter()-st_time

        if elapsed_time > move_period:
            pass
        else:
            time.sleep(move_period - elapsed_time)
        time.sleep(0.02)


