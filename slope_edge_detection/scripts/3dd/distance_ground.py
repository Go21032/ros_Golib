import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

# コールバック関数
def image_callback(msg):
    bridge = CvBridge()
    try:
        # ROS ImageメッセージをOpenCV画像に変換
        depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
        # 中央のピクセルの距離を取得
        height, width = depth_image.shape
        center_distance = depth_image[height // 2, width // 2] /1000
        
        print(f"中央の距離: {center_distance:.2f} メートル")
        
    except Exception as e:
        print(f"変換エラー: {e}")

def main():
    rospy.init_node('depth_listener')
    rospy.Subscriber('/camera/depth/image_rect_raw', Image, image_callback)
    rospy.spin()

if __name__ == '__main__':
    main()

# 単に地面との距離測定
# import cv2
# import numpy as np
# import pyrealsense2 as rs
# from ultralytics import YOLO

# #カメラの起動コマンドを起動すると動かないよ
# # パイプラインのセットアップ
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# # ストリーム開始
# pipeline.start(config)

# try:
#     while True:
#         # フレームを取得
#         frames = pipeline.wait_for_frames()
#         depth_frame = frames.get_depth_frame()
        
#         if not depth_frame:
#             continue
        
#         # 深度データを取得
#         depth_image = np.asanyarray(depth_frame.get_data())
        
#         # 中央のピクセルの距離を取得
#         height, width = depth_image.shape
#         center_distance = depth_frame.get_distance(width // 2, height // 2)
        
#         print(f"中央の距離: {center_distance:.2f} メートル")
        
# finally:
#     # クリーンアップ
#     pipeline.stop()
