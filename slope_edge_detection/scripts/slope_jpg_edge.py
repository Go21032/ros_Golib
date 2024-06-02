#!/usr/bin/env python3
import rospy
import sys
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import os  # ディレクトリ操作用に追加

class cvBridgeDemo:
    def __init__(self):
        self.node_name = "cv_bridge_demo"
        rospy.init_node(self.node_name)

        rospy.on_shutdown(self.cleanup)
        
        self.bridge = CvBridge()
        
        # 画像のパスを指定する部分を追加
        self.image_path = "/home/go/slope_ws/src/ros_Golib/slope_edge_detection/date_set/slope.jpg"  # ここに指定した画像ファイルを読み込む
        
        # 画像出力のためのPublisherはそのまま
        self.image_pub = rospy.Publisher("/output/image_raw", Image, queue_size=1)

        # ROSのSubscriberではなく、直接画像を読み込むように変更
        self.process_and_publish_images()

    def process_and_publish_images(self):
        try:
            input_image = cv2.imread(self.image_path)  # 画像を読み込む
            if input_image is not None:
                output_image = self.process_image(input_image)
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(output_image, "mono8"))
                cv2.imshow(self.node_name, output_image)
                cv2.waitKey(1)
            else:
                rospy.logwarn("Image read failed: {}".format(self.image_path))
        except CvBridgeError as e:
            print(e)
    
    def process_image(self, frame):
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(grey, (7, 7))
        edges = cv2.Canny(blur, 50.0, 80.0)
        return edges
        
    def cleanup(self):
        cv2.destroyAllWindows()
    
if __name__ == '__main__':
    cvBridgeDemo()
    rospy.spin()
