import cv2
import numpy as np
import matplotlib.pyplot as plt
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# 加载预定义的字典
dict_gen = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250)
cameraMatrix = np.array([[910.7659, 0.0, 646.563],
                        [0.0, 910.365, 356.575],
                        [ 0.0, 0.0, 1.0]])
distCoeffs = np.array([-0.001, 0.0, 0.0, 0.0, 0.0])

# Initialize the detector parameters using default values
parameters =  cv2.aruco.DetectorParameters()

# =========================实时视频流检测=============================
rospy.init_node('get_images', anonymous=True)
bridge = CvBridge()
while True:

    color_image = rospy.wait_for_message("/robot_camera/color/image_raw", Image, timeout=None)
    color_img = bridge.imgmsg_to_cv2(color_image, 'bgr8')

    # 显示帧
    cv2.namedWindow('Video',cv2.WINDOW_NORMAL)  
    cv2.imshow('Video', color_img)

    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)  

    # Detect the markers in the image
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(gray, dict_gen, parameters=parameters)
    print(markerIds)
    
    # frame_markers = cv2.aruco.drawDetectedMarkers(color_img, markerCorners, markerIds)

    if markerIds is not None:
        # rvecs and tvecs are the rotation and translation vectors respectively, for each of the markers in corners.
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 7.91, cameraMatrix, distCoeffs)
        print("tvecs",tvecs)

        for rvec, tvec in zip(rvecs, tvecs):
            frame_axes = cv2.drawFrameAxes(color_img, cameraMatrix, distCoeffs, rvec, tvec, 1)
            # frame_axes = cv2.aruco.drawAxis(color_img, cameraMatrix, distCoeffs, rvec, tvec, 1)

    if markerIds is not None:
        cv2.namedWindow('frame_makers',cv2.WINDOW_NORMAL)
        cv2.imshow('frame_markers',frame_axes)

    # 按 'Esc' 退出循环
    if cv2.waitKey(1) & 0xFF == 27:  # 27是Esc键的ASCII值
        break

# 关闭所有OpenCV窗口
cv2.destroyAllWindows()