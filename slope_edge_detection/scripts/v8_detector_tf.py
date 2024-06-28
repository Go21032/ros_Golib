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

from yolov8_detector import Detectorv8, parse_opt

#、yolov8_detector.py内のModelクラスの__init__メソッドを確認し、deviceというキーワード引数を受け入れるように修正する必要があります
class ObjectDetectionv8:

    def __init__(self, **args):
        #目標物によってtarget_nameを変更していく
        self.target_name = 'bottle'
        self.frame_id = 'target'

        self.detector = Detectorv8(**args)

        self.bridge = CvBridge()

        # sub_infoからsub_depthはmessage_filtersのSubscriberクラスのインスタンスを生成している
        self.sub_info = Subscriber('camera/aligned_depth_to_color/camera_info', CameraInfo)
        self.sub_color = Subscriber('camera/color/image_raw', Image)
        self.sub_depth = Subscriber('camera/aligned_depth_to_color/image_raw', Image)
        
        # 同期処理のためのApproximateTimeSynchronizerクラスのインスタンス生成
        self.ts = ApproximateTimeSynchronizer([self.sub_info, self.sub_color, self.sub_depth], 10, 0.1)
        
        # すべてのトピックのメッセージが揃ったときに呼ばれるコールバック
        self.ts.registerCallback(self.images_callback)
        self.broadcaster = TransformBroadcaster(self)
        #時間取得        
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        # ファイル名に現在の時刻を加える
        file_name = f'object_distance_{current_time}.csv'
        
        # CSVファイルを開き、ライターを初期化
        self.csv_file = open(file_name, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        # CSVヘッダーを書き込む
        self.csv_writer.writerow(['name', 'distance'])


    def images_callback(self, msg_info, msg_color, msg_depth):
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

        if target is not None:
            u1 = round(target.u1)
            u2 = round(target.u2)
            v1 = round(target.v1)
            v2 = round(target.v2)
            u = round((target.u1 + target.u2) / 2)
            v = round((target.v1 + target.v2) / 2)
            depth = np.median(img_depth[v1:v2+1, u1:u2+1])
            if depth != 0:
                z = depth * 1e-3
                fx = msg_info.K[0]
                fy = msg_info.K[4]
                cx = msg_info.K[2]
                cy = msg_info.K[5]
                x = z / fx * (u - cx)
                y = z / fy * (v - cy)
                #loginfoでprit
                #rospy.loginfo(f'{target.name} ({x:.3f}, {y:.3f}, {z:.3f})')
                
                #3次元空間における二点間の距離公式https://mathwords.net/nitennokyori
                dis_x = x ** 2
                dis_y = y ** 2
                dis_z = z ** 2
                dis = np.sqrt(dis_x + dis_y + dis_z)
                rospy.loginfo(f'{target.name} ({dis:.3f})')
                
                # CSVファイルに検出された対象の情報を書き込み
                self.csv_writer.writerow([target.name, f'{dis:.3f}'])
                
                ts = TransformStamped()
                ts.header = msg_depth.header
                ts.child_frame_id = self.frame_id
                ts.transform.translation.x = x
                ts.transform.translation.y = y
                ts.transform.translation.z = z
                self.broadcaster.sendTransform((x, y, z), (0, 0, 0, 1), rospy.Time.now(), self.frame_id, msg_depth.header.frame_id)

        img_depth *= 16
        if target is not None:
            pt1 = (int(target.u1), int(target.v1))
            pt2 = (int(target.u2), int(target.v2))
            cv2.rectangle(img_depth, pt1=pt1, pt2=pt2, color=0xffff)

        cv2.imshow('depth', img_depth)
        cv2.waitKey(1)

    def __del__(self):
        # オブジェクトが削除される際にCSVファイルを閉じる
        self.csv_file.close()
        
if __name__ == '__main__':
    rospy.init_node('detection_tf')
    opt = parse_opt(args=sys.argv)
    node = ObjectDetectionv8(**vars(opt))
    try:
        rospy.spin()
    except KeyboardInterrupt:
        # CSVファイルを安全に閉じる
        if node.csv_file:
            node.csv_file.close()
        pass
    rospy.signal_shutdown('KeyboardInterrupt')

