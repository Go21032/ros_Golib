#!/usr/bin/env python3
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge, CvBridgeError
from message_filters import Subscriber, ApproximateTimeSynchronizer
from tf import TransformBroadcaster
import rospy

#これはボツコード
class maskDetectionv8:
    def __init__(self):
        self.model = YOLO("/home/carsim05/slope_ws/src/ros_Golib/slope_edge_detection/scripts/best.pt")

        self.bridge = CvBridge()
        
        self.sub_info = Subscriber('camera/aligned_depth_to_color/camera_info', CameraInfo)
        self.sub_color = Subscriber('camera/color/image_raw', Image)
        self.sub_depth = Subscriber('camera/aligned_depth_to_color/image_raw', Image)
        self.ts = ApproximateTimeSynchronizer([self.sub_info, self.sub_color, self.sub_depth], 10, 0.1)
        self.ts.registerCallback(self.images_callback)
        self.broadcaster = TransformBroadcaster(self)
        
        # Initialize camera parameters
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

    def images_callback(self, cmsg_info, msg_color, msg_depth):
        try:
            img_color = self.bridge.imgmsg_to_cv2(msg_color, 'bgr8')
            img_depth = self.bridge.imgmsg_to_cv2(msg_depth, 'passthrough')#passthroughは何も変換せず生のデータを
            
            # img_depth配列をコピーして書き込み可能にする
            img_depth = img_depth.copy()

        except CvBridgeError as e:
            rospy.logwarn(str(e))
            return

        if img_color.shape[0:2] != img_depth.shape[0:2]:
            rospy.logwarn('カラーと深度の画像サイズが異なる')
            return

        img_color, result = self.detector.detect(img_color)
        rospy.loginfo("Detection completed")

        cv2.imshow('color', img_color)
        target = None
        for r in result:
            if r.name == self.target_name:
                target = r
                break
            
        # Perform mask detection
        img = Image.fromarray(cv2.cvtColor(msg_color, cv2.COLOR_BGR2RGB))
        results = self.model.predict(source=img, save=True)

        masks = results[0].masks
        if masks is None:
            print("No masks detected")
            return

        x_numpy = masks[0].data.to('cpu').detach().numpy().copy()
        print(x_numpy.shape)

        name = results[0].names
        print(name)
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
        result = np.array(result)
        point = result

        # PIL Imageをnumpy配列に変換
        img = np.array(img)

        # OpenCVで使用するためにRGBからBGRに変換
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        for i in range(len(point)):
            (u, v) = (int(point[i][0]), int(point[i][1]))
            print((u, v))
            cv2.circle(img, (u, v), 10, (0, 0, 255), -1)
            
            # Get depth value from depth image
            depth = depth_cv_image[v, u]
            if np.isnan(depth) or depth <= 0:
                continue  # Skip invalid depth values

            # Convert (u, v, depth) to 3D coordinates
            x = (u - self.cx) * depth / self.fx
            y = (v - self.cy) * depth / self.fy
            z = depth

            print(f"3D coordinates: ({x}, {y}, {z})")

        # 画像を表示＆保存
        cv2.imwrite('seg_mask11.jpg', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    mask_detector = maskDetectionv8()
