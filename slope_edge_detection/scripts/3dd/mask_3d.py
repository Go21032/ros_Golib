#!/usr/bin/env python3
import numpy as np
import cv2
from ultralytics import YOLO
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from message_filters import Subscriber, ApproximateTimeSynchronizer

class SlopeDetectionNode:
    def __init__(self):
        rospy.init_node('mask_3d', anonymous=True)
        
        self.bridge = CvBridge()

        # モデルの初期化
        self.model = YOLO("/home/go/slope_ws/src/ros_Golib/slope_edge_detection/scripts/3dd/pl2_best.pt")

        # サブスクライバの初期化
        self.sub_info = Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo)
        self.sub_color = Subscriber('/camera/color/image_raw', Image)
        self.sub_depth = Subscriber('/camera/aligned_depth_to_color/image_raw', Image)

        # メッセージフィルタの設定
        self.ts = ApproximateTimeSynchronizer([self.sub_info, self.sub_color, self.sub_depth], 10, 0.1)
        self.ts.registerCallback(self.images_callback)

    def pixel_to_camera_coordinates(self, u, v, depth, fx, fy, cx, cy):
        z = depth * 1e-3  # 深度値をメートルに変換
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z])

    def images_callback(self, msg_info, msg_color, msg_depth):
        try:
            img_color = self.bridge.imgmsg_to_cv2(msg_color, 'bgr8')
            img_depth = self.bridge.imgmsg_to_cv2(msg_depth, 'passthrough')
            img_depth = img_depth.copy()
        except CvBridgeError as e:
            rospy.logwarn(str(e))
            return

        if img_color.shape[0:2] != img_depth.shape[0:2]:
            rospy.logwarn('カラーと深度の画像サイズが異なる')
            return

        fx = msg_info.K[0]
        fy = msg_info.K[4]
        cx = msg_info.K[2]
        cy = msg_info.K[5]

        results = self.model.predict(source=img_color, save=True)
        if not results or not results[0].masks:
            rospy.logwarn('予測結果が得られませんでした')
            return

        masks = results[0].masks
        points = masks[0].xy
        points = np.array(points)

        points_3d = []
        points_uv = []

        for i in range(len(points)):
            for j in range(len(points[i])):
                u, v = int(points[i][j][0]), int(points[i][j][1])
                if 0 <= u < img_depth.shape[1] and 0 <= v < img_depth.shape[0]:
                    depth = img_depth[v, u]
                    point_3d = self.pixel_to_camera_coordinates(u, v, depth, fx, fy, cx, cy)
                    points_3d.append(point_3d)
                    points_uv.append((u, v))
                else:
                    rospy.logwarn(f'ポイント ({u}, {v}) が画像の範囲外です')

        points_3d = np.array(points_3d)
        points_uv = np.array(points_uv)

        distances = np.linalg.norm(points_3d, axis=1)

        for idx, distance in enumerate(distances):
            print(f"Point {points_uv[idx]} -> Distance: {distance} mm")

        sorted_indices = np.argsort(distances)
        nearest_points_3d = points_3d[sorted_indices[:10]]
        nearest_points_uv = points_uv[sorted_indices[:10]]

        print("Nearest 3D Points: ", nearest_points_3d)
        print("Nearest UV Points: ", nearest_points_uv)

        img_bgr = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)

        for (u, v) in nearest_points_uv:
            cv2.circle(img_bgr, (u, v), 5, (0, 255, 0), -1)

        cv2.imwrite('Mask_3D.jpg', img_bgr)
        cv2.imshow('Nearest Points', img_bgr)
        cv2.waitKey(1)


if __name__ == '__main__':
    try:
        node = SlopeDetectionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
