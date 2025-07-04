import pyRobotiqGripper
import serial
import minimalmodbus as mm
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
    portName=None
    for port in ports:
        try:
            print(f"port.serial_number: {port.serial_number}")
            if port.serial_number == "DAA5HFG8":
                left_gripper_port = port.device
            elif port.serial_number == "DAA5H8OB":
                right_gripper_port = port.device
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
            portName=port.device
            del device

            ser.close()  # Close the port
        except:
            pass  # Skip if port cannot be opened
    return left_gripper_port, right_gripper_port

left_gripper_port, right_gripper_port = gripper_init()
gripper_le = pyRobotiqGripper.RobotiqGripper(left_gripper_port)
gripper_le.goTo(0) #全开
# gripper_le.goTo(255)
# gripper_le.goTo(213) # 左爪松一点

gripper_ri = pyRobotiqGripper.RobotiqGripper(right_gripper_port)
gripper_ri.goTo(0) #全开
# gripper_ri.goTo(255)


# gripper.activate() # 激活，第一次使用要运行，后面不需要
# gripper.open() # 全开，0
# gripper.close() # 全闭，255
# gripper.goTo(0) # 全开
# gripper.goTo(223)
# position_in_bit = gripper.getPosition()
# print(position_in_bit)
# gripper.goTomm(25)
# position_in_mm = gripper.getPositionmm()
# print(position_in_mm)
