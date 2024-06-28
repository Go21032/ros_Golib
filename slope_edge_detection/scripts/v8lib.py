# #参考サイト公式https://docs.ultralytics.com/ja/guides/ros-quickstart/?h=distance#using-yolo-with-depth-images
# import time
# import rospy
# import numpy as np
# import ros_numpy
# from std_msgs.msg import String
# from ultralytics import YOLO
# from sensor_msgs.msg import Image

# rospy.init_node("ultralytics")
# time.sleep(1)

# segmentation_model = YOLO("/home/go/slope_ws/src/ros_Golib/slope_edge_detection/scripts/best.pt")

# classes_pub = rospy.Publisher("ultralytics/detection/distance", String, queue_size=5)


# def callback(data):
#     """Callback function to process depth image and RGB image."""
#     image = rospy.wait_for_message("/camera/color/image_raw", Image)
#     image = ros_numpy.numpify(image)
#     depth = ros_numpy.numpify(data)
#     result = segmentation_model(image)

#     for index, cls in enumerate(result[0].boxes.cls):
#         class_index = int(cls.cpu().numpy())
#         name = result[0].names[class_index]
#         mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)
#         obj = depth[mask == 1]
#         obj = obj[~np.isnan(obj)]
#         avg_distance = np.mean(obj) if len(obj) else np.inf

#     classes_pub.publish(String(data=str(all_objects)))


# rospy.Subscriber("camera/aligned_depth_to_color/image_raw", Image, callback)

# while True:
#     rospy.spin()

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()

color_image = None
depth_image = None

def color_image_callback(data):
    global color_image
    color_image = bridge.imgmsg_to_cv2(data, "bgr8")
    display_images()

def depth_image_callback(data):
    global depth_image
    depth_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    display_images()

def display_images():
    if color_image is not None and depth_image is not None:
        depth_normalized = cv2.normalize(depth_image, None, 0, 1, cv2.NORM_MINMAX)
        depth_colormap = cv2.applyColorMap(np.uint8(depth_normalized*255), cv2.COLORMAP_JET)
        cv2.imshow("Color Image", color_image)
        cv2.imshow("Depth Image", depth_colormap)
        cv2.waitKey(1)

rospy.init_node('image_display_node')
rospy.Subscriber("/camera/color/image_raw", Image, color_image_callback)
rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, depth_image_callback)

while not rospy.is_shutdown():
    rospy.spin()

