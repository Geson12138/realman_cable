import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载预定义的字典
dict_gen = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)

# # save spawned ArUco Marker image to print
# for i in list(range(21,31)):

marker_id = 11 # 生成标记的id
marker_size = 300 # 生成标记的像素大小
marker_image = cv2.aruco.generateImageMarker(dict_gen, marker_id, marker_size,1)
cv2.imwrite(f'markerImage_{marker_size}_id{marker_id}.jpg',marker_image)

'''
# Initialize the detector parameters using default values
parameters =  cv2.aruco.DetectorParameters()

frame = cv2.imread('markerImage_100_id23.jpg')
gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(gray, dict_gen, parameters=parameters)
frame_markers = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

cv2.namedWindow('frame_makers',cv2.WINDOW_NORMAL)
while True:
    cv2.imshow('frame_markers',frame_markers)
    # 按 'Esc' 退出循环
    if cv2.waitKey(1) & 0xFF == 27:  # 27是Esc键的ASCII值
        break
cv2.destroyAllWindows()
'''

# =========================实时视频流检测=============================

'''

# 初始化摄像头
cap = cv2.VideoCapture(0)  # 0 通常是默认摄像头的标识

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    # 读取帧
    ret, frame = cap.read()

    # 如果正确读取帧，ret为True
    if not ret:
        print("无法接收帧，请退出")
        break

    # 显示帧
    # cv2.imshow('Video', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  

    # Detect the markers in the image
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(gray, dict_gen, parameters=parameters)
    
    frame_markers = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

    # if markerIds is not None:
    #     # rvecs and tvecs are the rotation and translation vectors respectively, for each of the markers in corners.
    #     rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 1, cameraMatrix, distCoeffs)

    # for rvec, tvec in zip(rvecs, tvecs):
    #     frame_axes = cv2.aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec, tvec, 1)

    cv2.namedWindow('frame_makers',cv2.WINDOW_NORMAL)
    cv2.imshow('frame_markers',frame_axes)
    # 按 'Esc' 退出循环
    if cv2.waitKey(1) & 0xFF == 27:  # 27是Esc键的ASCII值
        
        break

# 释放摄像头资源
cap.release()
# 关闭所有OpenCV窗口
cv2.destroyAllWindows()

'''
