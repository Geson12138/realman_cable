import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from pykin.utils import transform_utils as transform_utils

charcuodiamond_pub = rospy.Publisher('/robot1_camera/color/charcuodiamond', Image, queue_size=10)

# 加载预定义的字典
dict_gen = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
# cameraMatrix = np.array([[617.306, 0.0, 424.5],
#                         [0.0, 617.306, 240.5],
#                         [ 0.0, 0.0, 1.0]])
# distCoeffs = np.array([-0.001, 0.0, 0.0, 0.0, 0.0])

# cameraMatrix = np.array([[389.05096435546875, 0.0, 321.7474365234375],
#                         [0.0, 388.4710998535156, 238.11932373046875],
#                         [0.0, 0.0, 1.0]],np.float32)
cameraMatrix = np.array([[910.7659301757812, 0.0, 646.5628051757812],
                        [0.0, 910.3656616210938, 356.5755310058594],
                        [0.0, 0.0, 1.0]],np.float32)
distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0],np.float32)

# Initialize the detector parameters using default values
parameters =  cv2.aruco.DetectorParameters()

# =========================实时视频流检测=============================
rospy.init_node('get_images', anonymous=True)
bridge = CvBridge()

rvec = None
tvec = None

cv2.namedWindow('Video',cv2.WINDOW_NORMAL)
cv2.namedWindow('frame_makers',cv2.WINDOW_NORMAL)
time_count = 0
while True:

        color_image = rospy.wait_for_message("/robot_camera/color/image_raw", Image, timeout=None)
        color_img = bridge.imgmsg_to_cv2(color_image, 'bgr8')

        # 显示帧
        cv2.imshow('Video', color_img)

        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)  

        # Detect the markers in the image
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(gray, dict_gen, parameters=parameters)
        
        # if at least one marker detected
        if markerIds is not None:

            diamondCorners, diamondIds = cv2.aruco.detectCharucoDiamond(gray, markerCorners, markerIds, 1.36, cameraMatrix, distCoeffs) # square length / marker length

            print(f'diamondIds: {diamondIds}\n')
            print(type(diamondIds))
            print(f'diamondIds: {diamondIds}')
            error_flag = False
            
            if diamondIds.ndim >1:
            
                for i in range(len(diamondIds)):
                    if diamondIds[i][0][0] == 0:
                        diamondCorner = diamondCorners[i]
                        error_flag = False
                        break
                    else:
                        error_flag = True
                if not error_flag:
                      
                    print(f'diamondCorner:{diamondCorner}')
                    diamondCorner = diamondCorner.reshape(1,-1,2)
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(diamondCorner, 0.053, cameraMatrix, distCoeffs) # square length

                    rvec = np.array(rvecs) 
                    tvec = np.array(tvecs)
                    Rot = cv2.Rodrigues(rvec)[0]
                    euler_angle_rad = transform_utils.get_rpy_from_matrix(Rot)
                    euler_angle = np.rad2deg(euler_angle_rad)
                    tvec = tvec.reshape(3,1)

                    transMatrix = np.column_stack((Rot,tvec))
                    transMatrix = np.row_stack((transMatrix,np.array([0,0,0,1])))
                    print(f'euler_angle: {euler_angle}\n')
                    print(f'tvec: {tvec}\n')
                    # print(transMatrix)

                    trans_cam2robot_Rot = transform_utils.get_matrix_from_rpy(np.array([-15.0, 0.0, 0.0])/180*np.pi)
                    trans_cam2robot_Pos = np.array([0.0, 0.0, 0.0])
                    trans_cam2robot = np.column_stack((trans_cam2robot_Rot,trans_cam2robot_Pos))
                    trans_cam2robot = np.row_stack((trans_cam2robot,np.array([0,0,0,1])))

                    trans_aruco2robot =  trans_cam2robot @ transMatrix
                    rPose_aruco2robot = np.concatenate((np.rad2deg(transform_utils.get_rpy_from_matrix(trans_aruco2robot[:3,:3])),trans_aruco2robot[:3,3].transpose()))
                    print(f'rPose_aruco2robot: {rPose_aruco2robot}')
                    
                    frame_axes = cv2.drawFrameAxes(color_img, cameraMatrix, distCoeffs, rvec, tvec, 0.1)   
                    cv2.aruco.drawDetectedMarkers(color_img, markerCorners,markerIds)
                    time_count += 1
                    
                    color_img = bridge.cv2_to_imgmsg(color_img, 'bgr8')
                    charcuodiamond_pub.publish(color_img)
                    cv2.imshow('frame_markers',frame_axes)
                
            else:
                continue

        else:
            continue
        # print(f'time_count: {time_count}\n')
        # if time_count >= 1:
        #     print(f'tvec: {tvec}')
        #     break

        # 按 'Esc' 退出循环
        if cv2.waitKey(1) & 0xFF == 27:  # 27是Esc键的ASCII值
            break

# 关闭所有OpenCV窗口
cv2.destroyAllWindows()