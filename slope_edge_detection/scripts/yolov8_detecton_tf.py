# import cv2
# import numpy as np
# import pyrealsense2 as rs
# import csv
# import rospy
# import sys

# from ultralytics import YOLO
# from datetime import datetime
# from sensor_msgs.msg import Image, CameraInfo
# from geometry_msgs.msg import TransformStamped
# from cv_bridge import CvBridge, CvBridgeError
# from message_filters import Subscriber, ApproximateTimeSynchronizer
# from tf import TransformBroadcaster
# from detector import Detector, parse_opt  # ROS 1用に適切なモジュールに変更してください
# from detection_tf import ObjectDetection

# class YOLOv8Detection(ObjectDetection):
#     ObjectDetection.