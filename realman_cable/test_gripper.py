import pyRobotiqGripper
gripper = pyRobotiqGripper.RobotiqGripper()
# gripper.activate() # 激活，第一次使用要运行，后面不需要
# gripper.open() # 全开，255
# gripper.close() # 0
gripper.goTo(210)
gripper.goTo(255) # 位置
# position_in_bit = gripper.getPosition()
# print(position_in_bit)
# gripper.goTomm(25)
# position_in_mm = gripper.getPositionmm()
# print(position_in_mm)
