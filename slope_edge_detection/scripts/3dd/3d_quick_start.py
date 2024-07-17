#!/usr/bin/env python3
import time

import numpy as np
import ros_numpy
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from ultralytics import YOLO

rospy.init_node("3d_quick")
time.sleep(1)

segmentation_model = YOLO("pl2_best.pt")

classes_pub = rospy.Publisher("/ultralytics/detection/distance", String, queue_size=5)


def callback(data):
    """Callback function to process depth image and RGB image."""
    image = rospy.wait_for_message("/camera/color/image_raw", Image)
    image = ros_numpy.numpify(image)
    depth = ros_numpy.numpify(data)
    result = segmentation_model(image)

    for index, cls in enumerate(result[0].boxes.cls):
        class_index = int(cls.cpu().numpy())
        name = result[0].names[class_index]
        mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)
        obj = depth[mask == 1]
        obj = obj[~np.isnan(obj)]
        avg_distance = np.mean(obj) if len(obj) else np.inf

    classes_pub.publish(String(data=str(all_objects)))


rospy.Subscriber("/camera/depth/image_raw", Image, callback)

while True:
    rospy.spin()