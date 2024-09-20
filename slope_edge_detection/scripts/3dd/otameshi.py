#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import csv
import rospy
from datetime import datetime
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge, CvBridgeError
from message_filters import Subscriber, ApproximateTimeSynchronizer
from tf import TransformBroadcaster
from ultralytics import YOLO
from PIL import Image as PilImage

class SlopeDetection:

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        self.frame_id = 'slope'
        
        # Subscribers for synchronized image and camera info
        self.sub_info = Subscriber('camera/aligned_depth_to_color/camera_info', CameraInfo)
        self.sub_color = Subscriber('camera/color/image_raw', Image)
        self.sub_depth = Subscriber('camera/aligned_depth_to_color/image_raw', Image)
        
        # Synchronize the image and camera info topics
        self.ts = ApproximateTimeSynchronizer([self.sub_info, self.sub_color, self.sub_depth], 10, 0.1)
        self.ts.registerCallback(self.images_callback)
        
        # Transform broadcaster
        self.broadcaster = TransformBroadcaster()
        
        # Prepare CSV file
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f'slope_coordinates_{current_time}.csv'
        self.csv_file = open(file_name, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['X', 'Y', 'Z'])

    def images_callback(self, msg_info, msg_color, msg_depth):
        try:
            img_color = self.bridge.imgmsg_to_cv2(msg_color, 'bgr8')
            img_depth = self.bridge.imgmsg_to_cv2(msg_depth, 'passthrough').copy()
        except CvBridgeError as e:
            rospy.logwarn(str(e))
            return

        if img_color.shape[0:2] != img_depth.shape[0:2]:
            rospy.logwarn('カラーと深度の画像サイズが異なる')
            return

        # Use YOLO model to detect slope
        pil_img = PilImage.fromarray(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        results = self.model.predict(source=pil_img)
        masks = results[0].masks

        if masks:
            # Get the top 50 points with highest y-coordinates
            point = np.array(masks[0].xy)
            point = sorted(point, key=lambda x: x[1], reverse=True)[:50]
            
            # Calculate median point
            median_x = int(np.median([p[0] for p in point]))
            median_y = int(np.median([p[1] for p in point]))

            # Get depth at median point
            depth = img_depth[median_y, median_x]

            if depth != 0:
                z = depth * 1e-3
                fx = msg_info.K[0]
                fy = msg_info.K[4]
                cx = msg_info.K[2]
                cy = msg_info.K[5]

                x = z / fx * (median_x - cx)
                y = z / fy * (median_y - cy)

                rospy.loginfo(f'Slope 3D Coordinate: ({x:.3f}, {y:.3f}, {z:.3f})')

                # Write to CSV
                self.csv_writer.writerow([x, y, z])

                # Broadcast transform
                ts = TransformStamped()
                ts.header = msg_depth.header
                ts.child_frame_id = self.frame_id
                ts.transform.translation.x = x
                ts.transform.translation.y = y
                ts.transform.translation.z = z
                self.broadcaster.sendTransform((x, y, z), (0, 0, 0, 1), rospy.Time.now(), self.frame_id, msg_depth.header.frame_id)

    def __del__(self):
        self.csv_file.close()

if __name__ == '__main__':
    rospy.init_node('slope_detection')
    model_path = "/home/go/slope_ws/src/ros_Golib/slope_edge_detection/scripts/best.pt"
    node = SlopeDetection(model_path)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        if node.csv_file:
            node.csv_file.close()
        pass
    rospy.signal_shutdown('KeyboardInterrupt')
