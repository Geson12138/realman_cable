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
    
    
    # err_code, joint_l, pose_l, arm_err_ptr, sys_err_ptr = arm_le.Get_Current_Arm_State()
    _, joint_l, pose_l, _, _ = arm_le.Get_Current_Arm_State()

    
    # err_code, joint_r, pose_r, arm_err_ptr, sys_err_ptr = arm_ri.Get_Current_Arm_State()
    _, joint_r, pose_r, _, _ = arm_ri.Get_Current_Arm_State()
    # joint_r_round = [round(float(j), 1) for j in joint_r]
    # pose_r_round = [round(float(j), 1) for j in pose_r]
    print('left arm joint:', [round(float(j), 2) for j in joint_l])
    print('right arm joint:', [round(float(j), 2) for j in joint_r])

    print(f'l_ee_position: {[round(float(j),3) for j in pose_l[:3]]}\n l_ee_euler: {[round(float(j),3) for j in pose_l[3:]]}')
    print(f'r_ee_position: {[round(float(j),3) for j in pose_r[:3]]}\n r_ee_euler: {[round(float(j),3) for j in pose_r[3:]]}')

    # ------------------pykin-------------------------------
    print('\n------------------pykin-------------------\n')

    pykin_ri = SingleArm(f_name='./urdf/rm_75b_hand_gripper.urdf')
    pykin_ri.setup_link_name('base_link', 'Link7')
    pykin_le = SingleArm(f_name='./urdf/rm_75b_hand_gripper.urdf')
    pykin_le.setup_link_name('base_link', 'tool_hand') 
    # pykin_le.setup_link_name('base_link', 'Link7')
    pykin_fsensor = SingleArm(f_name='./urdf/rm_75b_hand_gripper.urdf')
    pykin_fsensor.setup_link_name('base_link', 'f_sensor') # 力传感器坐标系

    joints_l_rad = [j/180*np.pi for j in joint_l]
    joints_r_rad = [j/180*np.pi for j in joint_r]
    
    l_ee = pykin_le.forward_kin(joints_l_rad)
    r_ee = pykin_ri.forward_kin(joints_r_rad)

    l_ee_pos = l_ee[pykin_le.eef_name].pos
    l_ee_rot = l_ee[pykin_le.eef_name].rot
    l_ee_euler = transform_utils.get_rpy_from_quaternion(l_ee_rot)
    print("lee_pose:", [round(float(j),3) for j in l_ee_pos], "lee_ori:", [round(float(j),3) for j in l_ee_rot])
    print("l_ee_euler:", [round(float(j),3) for j in l_ee_euler])
    
    r_ee_pos = r_ee[pykin_ri.eef_name].pos
    r_ee_rot = r_ee[pykin_ri.eef_name].rot
    r_ee_euler = transform_utils.get_rpy_from_quaternion(r_ee_rot)
    print("ree_pose:", [round(float(j),3) for j in r_ee_pos], "ree_ori:", [round(float(j),3) for j in r_ee_rot])
    print("r_ee_euler:", [round(float(j),3) for j in r_ee_euler])
    
    def robot_ikpy(target_pos,target_ori_rpy_rad):
        chain = ikpy.chain.Chain.from_urdf_file("./urdf/rm_75b.urdf",["base_link"])
        r_qd = chain.inverse_kinematics(target_pos,target_ori_rpy_rad,'all')
        arm_joints = [j/np.pi*180 for j in r_qd]
        print(f'ikpy:{arm_joints}')
        
        return arm_joints
    st_time = time.time()
    robot_ikpy(r_ee_pos,r_ee_euler)
    print(f'time {time.time()-st_time}')
    # arm_ri.Movej_Cmd([31.31, -58.05, 12.57, -57.48, 78.69, 82.96, -109.26],10)
    # input('dong!')
    # time.sleep(10)
    
    # conduct_start_ri = [-71.97, 2.18, 73.84, -62.92, 14.11, -59.01, 242-231.17]
    # conduct_start_ri = [2.2444705179893605, 100.97570640718084, 37.51248020898643, 0, 3.8859273709570017, -18.182498403820812, 6.780325245196895]
    # conduct_start_ri = [0, 0, 0, 0, 0, 0, 245]
    # conduct_start_le = [0, 0, 0, 0, 0, 0, 0]
    # conduct_start_le = [71.97, -2.18, -73.84, 62.92, -14.11, 59.01, 231.17]
    # arm_le.Movej_Cmd(conduct_start_le,15)
    # arm_ri.Movej_Cmd(conduct_start_ri,15)
    # arm_le.Movej_Cmd(down_joint_l,15)
    # arm_ri.Movej_Cmd(down_joint_r,15)
    # input()
    # arm_le.Set_Modbus_Mode(1,115200,10,True)
    # arm_ri.Set_Modbus_Mode(1,115200,10,True)

    # arm_le.Set_InspireHand_ControlMode([0,0,0,0,0,0])
    # arm_ri.Set_InspireHand_ControlMode([0,0,0,0,0,0])
    
    # arm_le.Set_InspireHand_Angle([1000,1000,1000,1000,1000,0])
    # arm_ri.Set_InspireHand_Angle([850,1000,1000,1000,850,0])
    # arm_le.Set_Hand_Force(1000)
    # arm_le.Set_Hand_?Angle([0,0,0,0,0,0])
    # input()
    # arm_ri.Set_Hand_Angle([0,0,0,0,0,0])
    # input()
    # time.sleep(1)
    # arm_le.Set_Hand_Speed(1000)
    # arm_le.Set_Hand_Angle([1000,1000,1000,1000,1000,0])
    # arm_ri.Set_Hand_Angle([1000,1000,1000,1000,1000,1000])
    # time.sleep(1)
    # input()
    # arm_ri.Set_Hand_Angle([1000,1000,1000,500,500,0])
    # arm_le.Set_Hand_Angle([100,100,100,100,100,300])
    # arm_le.Set_Hand_Angle([750,750,750,700,850,0])
    # arm_ri.Set_Hand_Angle([750,750,750,700,850,0])
    # time.sleep(1)

    # arm_le.Set_Hand_Angle([1000,1000,1000,1000,1000,0])
    # arm_ri.Set_Hand_Angle([1000,1000,1000,1000,1000,0])

    # arm_le.Close_Modbus_Mode(1)
    # arm_ri.Close_Modbus_Mode(1)

    arm_le.Arm_Socket_Close()
    arm_ri.Arm_Socket_Close()

    # play_voice('接下来我要操作咖啡机给您制作咖啡')
