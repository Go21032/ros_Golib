#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import rospy
import torch  # torchをインポート
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
        # GPUが利用可能か確認
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)  # まずはモデルをロード
        # モデルを指定したデバイスに移動
        self.model.model.to(self.device)  # モデルをGPUに移動

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
        
    def images_callback(self, msg_info, msg_color, msg_depth):
        try:
            img_color = self.bridge.imgmsg_to_cv2(msg_color, 'bgr8')
            img_depth = self.bridge.imgmsg_to_cv2(msg_depth, 'passthrough').copy()
                     
        except CvBridgeError as e:
            rospy.logwarn(str(e))
            return
        except Exception as e:
            print(f"変換エラー: {e}")
        
        if img_color.shape[0:2] != img_depth.shape[0:2]:
            rospy.logwarn('カラーと深度の画像サイズが異なる')
            return
            
        # Use YOLO model to detect slope
        pil_img = PilImage.fromarray(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        results = self.model.predict(source=pil_img, device=self.device, verbose=False)  # GPUを指定して予測

        if results and results[0].masks:
            self.process_segmentation(results[0], img_color, img_depth, msg_info, img_depth[img_color.shape[0] // 2, img_color.shape[1] // 2] / 1000)

    def process_segmentation(self, results, img_color, img_depth, msg_info, center_distance):
        # 予測結果がない場合の処理
        if not results or not results.masks:
            rospy.logwarn("予測結果が存在しません")
            return
        
        # 信頼度の取得と表示
        mask_confidences = results.boxes.conf
        print("信頼度:", mask_confidences)
        
        # 信頼度が90%以上か確認
        if results.boxes and results.boxes.conf[0] > 0.8 and center_distance < 4.5:
            masks = results.masks.to(self.device)  # マスクをGPUに送る
            x_numpy = masks[0].data.to('cpu').detach().numpy().copy()

            name = results.names
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

            # y座標が高い順にソート
            point = sorted(point, key=lambda x: x[1], reverse=True)

            # 上位50個のy座標が高い座標を取得
            top_points = point[:70]

            # 座標の表示
            for i in range(len(top_points)):
                (u, v) = (int(top_points[i][0]), int(top_points[i][1]))
                # 画像内に指定したクラス(results[0]の境界線を赤点で描画
                cv2.circle(img_color, (u, v), 10, (0, 0, 255), -1)
            
            # y座標のピクセル値が高い順に上位50個を選ぶ
            top_points_y_sorted = sorted(top_points, key=lambda p: p[1], reverse=True)[:70]

            # x座標の中央値を計算するために、上位50個のうちx座標を絞る
            median_x_value = np.median([p[0] for p in top_points_y_sorted])
            median_x_candidates = [p for p in top_points_y_sorted if abs(p[0] - median_x_value) < 70]  # 50は調整可能

            # x座標とy座標の中央値をそれぞれ算出
            median_x = int(np.median([p[0] for p in median_x_candidates]))
            median_y = int(np.median([p[1] for p in top_points_y_sorted]))

            # 中央の点を描画
            cv2.circle(img_color, (median_x, median_y), 10, (255, 0, 0), -1)
                        
            #映像出力rosbag playでやるときのみ外す
            cv2.imshow('color', img_color)
            cv2.waitKey(1)  # ウィンドウを更新するためのキー入力を待つ

if __name__ == '__main__':
    rospy.init_node('slope_detection')
    model_path = "/home/carsim05/slope_ws/src/ros_Golib/slope_edge_detection/scripts/best.pt"
    node = SlopeDetection(model_path)
    rospy.spin()

