#!/usr/bin/env python3
import rospy
import sys
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class cvBridgeDemo:
    #selfとはコンストラクタ(クラスのオブジェクト生成するとき)のこと
    def __init__(self):
        self.node_name = "cv_bridge_demo"
        rospy.init_node(self.node_name)

        #「on_shutdown」はスクリプトが終了するときに呼ばれるコールバック
        rospy.on_shutdown(self.cleanup)
        
        #OpenCVとROSのやり取りに必要なやつ
        self.bridge = CvBridge()
        
        #画像入力
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=1)
        #画像出力
        self.image_pub = rospy.Publisher("/output/image_raw", Image, queue_size=1)

    def image_callback(self, ros_image):
        try:
            #ROSの形式からOpenCVの形式（実質はnumpy）に変換します。カラー画像なのでbgr8と指定します。
            input_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError as e:
            print(e)
        output_image = self.process_image(input_image)

        #処理したグレー画像をpubに出力
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(output_image, "mono8"))
        
        #変換前と後の画像を新しくウィンドウを作って表示します
        cv2.imshow(self.node_name, output_image)   
        cv2.waitKey(1)
                          
    def process_image(self, frame):
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #カラー画像をグレースケール画像に変換しています。
        blur = cv2.blur(grey, (7, 7)) #グレースケール画像に7*7サイズのブラー(ぼかし)処理を適用しています
        #ブラー処理された画像に対してCanny法によるエッジ検出を行っています。15.0と30.0は、エッジを検出する閾値
        edges = cv2.Canny(blur, 50.0, 80.0)
        return edges
        
    def cleanup(self):
        #スクリプトが終了されたときに呼ばれるもので、これで開いているウィンドウをすべて閉じる
        cv2.destroyAllWindows()
    
if __name__ == '__main__':
    cvBridgeDemo()
    rospy.spin()