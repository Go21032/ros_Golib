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
        
        self.sub_info = Subscriber('camera/aligned_depth_to_color/camera_info', CameraInfo)
        self.sub_color = Subscriber('camera/color/image_raw', Image)
        self.sub_depth = Subscriber('camera/aligned_depth_to_color/image_raw', Image)
        
        self.ts = ApproximateTimeSynchronizer([self.sub_info, self.sub_color, self.sub_depth], 20, 0.1)
        self.ts.registerCallback(self.images_callback)
        
        self.broadcaster = TransformBroadcaster()
        
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f'slope_3次元座標{current_time}.csv'
        self.csv_file = open(file_name, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['name', 'X', 'Y', 'Z', 'median_x', 'median_y'])

    def images_callback(self, msg_info, msg_color, msg_depth):
        try:
            img_color = self.bridge.imgmsg_to_cv2(msg_color, 'bgr8')
            img_depth = self.bridge.imgmsg_to_cv2(msg_depth, 'passthrough').copy()
        except CvBridgeError as e:
            rospy.logwarn(str(e))
            return
        except Exception as e:
            print(f"変換エラー: {e}")
            return
        
        if img_color.shape[0:2] != img_depth.shape[0:2]:
            rospy.logwarn('カラーと深度の画像サイズが異なる')
            return
        
        self.process_image(img_color, img_depth, msg_depth)

    def process_image(self, img_color, img_depth, msg_depth):
        pil_img = PilImage.fromarray(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        results = self.model.predict(source=pil_img)

        if not results or not results[0].masks:
            rospy.logwarn("予測結果が存在しません")
            return
        
        mask_confidences = results[0].boxes.conf
        print("信頼度:", mask_confidences)
        
        if results[0].boxes and results[0].boxes.conf[0] > 0.8:
            masks = results[0].masks
            x_numpy = masks[0].data.to('cpu').detach().numpy().copy()

            name = results[0].names
            point = masks[0].xy
            point = np.array(point)

            result = []
            for i in range(len(point)):
                for j in range(len(point[i])):
                    my_list = []
                    for k in range(len(point[i][j])):
                        my_list.append(point[i][j][k])
                    result.append(my_list)
            point = np.array(result)

            point = sorted(point, key=lambda x: x[1], reverse=True)
            top_points = point[:70]

            for i in range(len(top_points)):
                (u, v) = (int(top_points[i][0]), int(top_points[i][1]))
                cv2.circle(img_color, (u, v), 10, (0, 0, 255), -1)
            
            top_points_y_sorted = sorted(top_points, key=lambda p: p[1], reverse=True)[:70]

            median_x_value = np.median([p[0] for p in top_points_y_sorted])
            median_x_candidates = [p for p in top_points_y_sorted if abs(p[0] - median_x_value) < 70]

            (u1, v1) = (int(top_points_y_sorted[0][0]), int(top_points_y_sorted[0][1]))
            (u2, v2) = (int(top_points_y_sorted[1][0]), int(top_points_y_sorted[1][1]))

            median_x = int(np.median([p[0] for p in median_x_candidates]))
            median_y = int(np.median([p[1] for p in top_points_y_sorted]))

            cv2.circle(img_color, (median_x, median_y), 10, (255, 0, 0), -1)

            # Assume x, y, z are obtained from some calculation
            x, y, z = 0, 0, 0  # Replace with actual calculation
            self.csv_writer.writerow([self.frame_id, x, y, z, median_x, median_y])

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
    model_path = "/home/carsim/gakuhari_ws/src/ros_Golib/slope_edge_detection/scripts/best.pt"
    node = SlopeDetection(model_path)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        if node.csv_file:
            node.csv_file.close()
        pass
    rospy.signal_shutdown('KeyboardInterrupt')
