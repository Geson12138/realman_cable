import os
import sys
# 添加项目根目录到sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import os
import sys
import numpy as np
from lib.robotic_arm import *
from lib.log_setting import CommonLog

logger_ = logging.getLogger(__name__)
logger_ = CommonLog(logger_)

images_path = './collected_images.txt'  #存储采集到的标定板的图片的文件夹路径
poses_path = './collected_poses.txt' #存放采集到的机械臂末端位姿的文件路径
jpg_path = "./"

# 加载预定义的字典
dict_gen = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
###########################################################################################
cameraMatrix = np.array([[907.5158081054688, 0.0, 658.6470947265625],
                        [0.0, 906.7430419921875, 391.1750793457031],
                        [0.0, 0.0, 1.0]],np.float32)
distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# Initialize the detector parameters using default values
parameters =  cv2.aruco.DetectorParameters()

charucoboard = cv2.aruco.CharucoBoard(
    size=(4,4), 
    squareLength= 0.037, 
    markerLength= 0.027, 
    dictionary=dict_gen)


def run():


    rospy.init_node('collect_data', anonymous=True)

    count = 0
    bridge = CvBridge()
    cv2.namedWindow('frame_makers',cv2.WINDOW_NORMAL)

    # 清空上一次运行的数据
    with open(poses_path, 'w') as f:
        pass  # 打开文件后立即关闭，目的是清空文件内容

    with open(images_path, 'w') as f:
        pass  # 同上

    while True:

        print(count)
        ##############################################更改图像来源
        color_image = rospy.wait_for_message("/left_camera/color/image_raw", Image, timeout=None)
        color_img = bridge.imgmsg_to_cv2(color_image, "bgr8")
        cv2.imshow("Capture_Video", color_img)  # 窗口显示，显示名为 Capture_Test
        cv2.waitKey(1)
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)  

        # Detect the markers in the image
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(gray, dict_gen, parameters=parameters)
        
        # if at least one marker detected
        if markerIds is not None:

            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(markerCorners, markerIds, gray, charucoboard)

            if charuco_retval:
                retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, charucoboard, cameraMatrix, distCoeffs, None, None)   

                if retval:
                    frame_axes = cv2.drawFrameAxes(color_img, cameraMatrix, distCoeffs, rvec, tvec, 0.1)

                    # print(f'旋转向量：{rvec}')
                    # print(f'平移向量：{tvec}')
                    rvec = rvec.flatten()
                    tvec = tvec.flatten()

                    cv2.imshow('frame_markers',frame_axes)
                    # cv2.waitKey(1)

        key = cv2.waitKey(1)
        if key == 27 : # esc
            exit()

        elif key == 115: # s

            error_code, _, curr_pose, _, _ = arm.Get_Current_Arm_State()  # 获取当前机械臂状态

            logger_.info(f'获取状态：{"成功" if error_code == 0 else "失败"},{f"当前位姿为{curr_pose}" if error_code == 0 else None} ')
            if error_code == 0:
                with open(poses_path, 'a+') as f:
                    # 将列表中的元素用空格连接成一行
                    curr_pose = [str(i) for i in curr_pose]
                    new_line = f'{",".join(curr_pose)}\n'
                    # 将新行附加到文件的末尾
                    f.write(new_line)

            pose_ = np.concatenate((rvec, tvec))
            new_line = ','.join(map(str, pose_)) + '\n'
            with open(images_path, 'a+') as f:
                f.write(new_line)

            cv2.imwrite(jpg_path +'/'+ str(count) + '.jpg', color_img)  # 保存；

            count += 1

if __name__ == '__main__':
    ##############################################################################################
    arm = Arm(75, '192.168.99.17') # 左臂
    # arm = Arm(75, '192.168.99.18') # 右臂
    
    ##############################################################################################
    arm.Change_Work_Frame('Base')  # 切换到Base坐标系
    # arm.Change_Tool_Frame('Hand_Frame')  # 左臂末端坐标系
    arm.Change_Tool_Frame('robotiq')  # 右臂末端坐标系
    run()
