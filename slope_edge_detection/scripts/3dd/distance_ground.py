import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

# コールバック関数
def depth_image_callback(msg):
    bridge = CvBridge()
    try:
        # ROS ImageメッセージをOpenCV画像に変換
        depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
        # 中央のピクセルの距離を取得
        height, width = depth_image.shape
        center_distance = depth_image[height // 2, width // 2] / 1000
        
        print(f"中央の距離: {center_distance:.2f} メートル")
        
    except Exception as e:
        print(f"変換エラー: {e}")

def color_image_callback(msg):
    bridge = CvBridge()
    try:
        # ROS ImageメッセージをOpenCV画像に変換
        color_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # カラー画像を表示
        cv2.imshow("Color Image", color_image)
        cv2.waitKey(1)
        
    except Exception as e:
        print(f"変換エラー: {e}")

def main():
    rospy.init_node('image_listener')
    rospy.Subscriber('/camera/depth/image_rect_raw', Image, depth_image_callback)
    rospy.Subscriber('/camera/color/image_raw', Image, color_image_callback)
    rospy.spin()

    # 終了時にウィンドウを閉じる
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
