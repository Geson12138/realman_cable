
import numpy as np
from admittance_controller import AdmittanceController
from lib.robotic_arm import *
import pyRobotiqGripper
from include.predefine_pose import *
import pyRobotiqGripper
import serial
import minimalmodbus as mm
import threading


def gripper_init():
    #Constants
    BAUDRATE=115200
    BYTESIZE=8
    PARITY="N"
    STOPBITS=1
    TIMEOUT=0.2
    AUTO_DETECTION="auto"
    SLAVEADDRESS = 9
    ports=serial.tools.list_ports.comports()
    for port in ports:
        try:
            if port.serial_number == "DAA5HFG8":
                right_gripper_port = port.device
            elif port.serial_number == "DAA5H8OB":
                left_gripper_port = port.device
            elif port.serial_number == "B002XFTO":
                force_sensor_port = port.device
            # Try opening the port
            ser = serial.Serial(port.device,BAUDRATE,BYTESIZE,PARITY,STOPBITS,TIMEOUT)
            device=mm.Instrument(ser,SLAVEADDRESS,mm.MODE_RTU,close_port_after_each_call=False,debug=False)

            #Try to write the position 100
            device.write_registers(1000,[0,100,0])

            #Try to read the position request eco
            registers=device.read_registers(2000,3,4)
            posRequestEchoReg3=registers[1] & 0b0000000011111111

            #Check if position request eco reflect the requested position
            if posRequestEchoReg3 != 100:
                raise Exception("Not a gripper")

            del device

            ser.close()  # Close the port
        except:
            pass  # Skip if port cannot be opened

    return left_gripper_port, right_gripper_port, force_sensor_port


def dual_arm_move(joints_left, joints_right, speed=15):

    thread_le = threading.Thread(target=arm_le.Movej_Cmd, args=(joints_left, speed))
    thread_ri = threading.Thread(target=arm_ri.Movej_Cmd, args=(joints_right, speed))
    thread_le.start()
    thread_ri.start()
    thread_le.join()
    thread_ri.join()


if __name__ == '__main__':
    
    # =====================机械臂连接========================
    byteIP_L = '192.168.99.17'
    byteIP_R = '192.168.99.18'

    arm_le = Arm(75,byteIP_L)
    arm_ri = Arm(75,byteIP_R)

    arm_le.Change_Tool_Frame('robotiq')
    arm_ri.Change_Tool_Frame('robotiq')
    arm_le.Change_Work_Frame('Base')
    arm_ri.Change_Work_Frame('Base')
    # =======================================================

    # =====================初始化夹爪========================
    left_gripper_port, right_gripper_port, force_sensor_port = gripper_init()
    gripper_le = pyRobotiqGripper.RobotiqGripper(left_gripper_port)
    gripper_ri = pyRobotiqGripper.RobotiqGripper(right_gripper_port)
    gripper_le.goTo(0) #全开
    gripper_ri.goTo(0) #全开
    # ======================================================

    # =====================初始化手臂位置========================
    input("enter to 双手抬起")
    dual_arm_move(ready_joint_le,ready_joint_ri,15)
    # ======================================================
