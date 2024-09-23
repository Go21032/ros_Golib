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
        
        # 画像とカメラ情報のトピックを同期させる
        self.ts = ApproximateTimeSynchronizer([self.sub_info, self.sub_color, self.sub_depth], 20, 0.1)
        self.ts.registerCallback(self.images_callback)
        
        # Transform broadcaster
        self.broadcaster = TransformBroadcaster()
        
        # CSV file
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f'slope_3次元座標{current_time}.csv'
        self.csv_file = open(file_name, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['name', 'X', 'Y', 'Z', 'distance'])

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

        # 予測結果がない場合の処理
        if not results or not results[0].masks:
            rospy.logwarn("予測結果が存在しません")
            return

        masks = results[0].masks
        x_numpy = masks[0].data.to('cpu').detach().numpy().copy()

        name = results[0].names
        point = masks[0].xy
        point = np.array(point)

        # ３次元を２次元に変換する
        result = []
        for i in range(len(point)):
            for j in range(len(point[i])):
                my_list = []
                for k in range(len(point[i][j])):
                    my_list.append(point[i][j][k])
                result.append(my_list)
        point = np.array(result)

        #y座標が高い順にソート
        point = sorted(point, key=lambda x: x[1], reverse=True)

        # 上位50個のy座標が高い座標を取得
        top_points = point[:50]

        # 座標の表示
        for i in range(len(top_points)):
            (u, v) = (int(top_points[i][0]), int(top_points[i][1]))
            # 画像内に指定したクラス(results[0]の境界線を赤点で描画
            cv2.circle(img_color, (u, v), 10, (0, 0, 255), -1)
        # 上位50個の座標の中から最もy座標が高い2つの点を選びu1u2などをそれに当てる
        (u1, v1) = (int(top_points[0][0]), int(top_points[0][1]))
        (u2, v2) = (int(top_points[1][0]), int(top_points[1][1]))

        # 上位50個の座標の中央値を算出
        median_x = int(np.median([p[0] for p in top_points]))
        median_y = int(np.median([p[1] for p in top_points]))
        cv2.circle(img_color, (median_x, median_y), 10, (255, 0, 0), -1)
        
        #映像出力rosbag playでやるときのみ外す
        # cv2.imshow('color', img_color)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     rospy.signal_shutdown('closed')
        #     return

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

            # 距離に変換
            dis_x = x ** 2
            dis_y = y ** 2
            dis_z = z ** 2
            dis = np.sqrt(dis_x + dis_y + dis_z)
            rospy.loginfo(f'{self.frame_id} ({dis:.3f})')
            # Write to CSV
            self.csv_writer.writerow([self.frame_id, x, y, z, f'{dis:.3f}'])

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
