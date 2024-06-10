#!/usr/bin/env python3
import rospy
import sys
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

# 画像の読み込み
image = cv2.imread('/home/go/slope_ws/src/ros_Golib/slope_edge_detection/date_set/slope.jpg')

# 画像の表示
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# しきい値処理による領域の検出
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold', threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 輪郭検出による領域の検出
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
cv2.imshow('Contours', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 機械学習を用いた領域の検出
model = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'model.caffemodel')
blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
model.setInput(blob)
detections = model.forward()
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
        start_x, start_y, end_x, end_y = box.astype('int')
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
