#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import rospy
import torch
import time
import open3d as o3d
import GPUtil
import csv  # CSVライブラリをインポート
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ultralytics import YOLO
from datetime import datetime  # datetimeモジュールをインポート

class SR:
    def __init__(self, model_path):
        rospy.init_node('SR_seiko_node')  # ノードの初期化をここに移動
        self.color_image = None
        self.depth_image = None
        self.intrinsics = None
        self.previous_b = None  # 前の b の値を保持するための変数
        self.previous_c = None  # 前の c の値を保持するための変数
        self.bridge = CvBridge()
        
        rospy.Subscriber('/camera/color/image_raw', Image, self.callback, callback_args='color')
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.callback, callback_args='depth')
        rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, self.callback, callback_args='info')
        
        # YOLOモデルの初期化
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)
        self.model.model.to(self.device)

        # 現在の時間を取得し、フォーマットを指定してCSVファイル名を生成
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f'processing_time_{current_time}.csv'
        
        # CSVファイルの初期化
        self.csv_file = open(csv_filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['処理名', '時間（秒）', '成功率（%）'])  # ヘッダー行を書き込み

        # 成功した平面検出のカウンタと全体のフレーム数
        self.successful_plane_detections = 0
        self.total_frames = 0

        # ウィンドウの作成と位置設定
        cv2.namedWindow('slope_detection')  # ウィンドウを事前に作成
        cv2.moveWindow('slope_detection', 710, 850)  # ウィンドウの位置を設定
        cv2.namedWindow('bgr')  # bgrウィンドウも作成
        cv2.moveWindow('bgr', 710, 0)  # bgrウィンドウの位置を設定

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window('PCD', width=640, height=480, left=0, top=550)
        self.pointcloud = o3d.geometry.PointCloud()
        self.geom_added = False
        self.rate = rospy.Rate(10)

    def callback(self, msg, msg_type):
        if msg_type == 'color':
            self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        elif msg_type == 'depth':
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        elif msg_type == 'info':
            self.intrinsics = msg

    def get_gpu_usage(self):
        gpus = GPUtil.getGPUs()
        if gpus:
            return gpus[0].load * 100
        return 0

    def process_pointcloud(self):
        try:
            last_ransac_time = time.time()  # 最後のRANSAC処理の時間を初期化
            while not rospy.is_shutdown():
                if self.color_image is None or self.depth_image is None or self.intrinsics is None:
                    rospy.loginfo("Waiting for images and camera info...")
                    rospy.sleep(0.1)
                    continue

                o3d_color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
                img_depth = o3d.geometry.Image(self.depth_image)
                img_color = o3d.geometry.Image(o3d_color_image)

                # PyTorchを使用して深度画像をGPUで処理
                depth_tensor = torch.tensor(self.depth_image, dtype=torch.float32, device='cuda')  # GPUに転送し、Float32に変換
                filtered_depth = depth_tensor[depth_tensor < 1000]  # 例: 深度フィルタリング

                # Open3Dでポイントクラウドを生成
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    img_color, img_depth, convert_rgb_to_intensity=False)

                pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    self.intrinsics.width, self.intrinsics.height,
                    self.intrinsics.K[0], self.intrinsics.K[4],
                    self.intrinsics.K[2], self.intrinsics.K[5])

                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)

                # 以下の処理は前と同様
                pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                points = np.asarray(pcd.points)
                pass_x = (points[:, 0] > -1.0) & (points[:, 0] < 1.0)
                pass_y = (points[:, 1] > -1.0) & (points[:, 1] < 1.0)
                filtered_indices = np.where(pass_x & pass_y)[0]
                filtered_pcd = pcd.select_by_index(filtered_indices)
                filtered_pcd = filtered_pcd.voxel_down_sample(voxel_size=0.0015)

                # 平面セグメンテーション
                # RANSAC処理を3秒ごとに実行
                current_time = time.time()
                if current_time - last_ransac_time >= 0.0:  # 3秒経過したら処理を実行
                    start_pcd = time.time() #計測開始
                    plane_model, inliers = filtered_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
                    inlier_cloud = filtered_pcd.select_by_index(inliers)
                    a, b, c, d = plane_model
                    normal = np.array([a, b, c])

                    angle_degrees = 0
                    
                    if np.abs(normal[1]) < 0.9:
                        inlier_cloud.paint_uniform_color([1.0, 0, 0])
                        angle_with_vertical = np.arccos(np.abs(normal[1]))
                        angle_degrees = np.degrees(angle_with_vertical)
                        rospy.loginfo(f"angle: {angle_degrees:.2f} degrees, omomi( a: {normal[0]:.2f}, b: {normal[1]:.2f}, c: {normal[2]:.2f}, d: {d:.2f})")
                        plane_segments = [inlier_cloud]
                        self.successful_plane_detections += 1  # 成功したカウントを増やす
                    else:
                        plane_segments = []
                    
                    end_pcd = time.time() #計測終了
                    pcd_time = end_pcd - start_pcd
                    rospy.loginfo(f"平面処理時間: {pcd_time:.4f}秒")
                    self.csv_writer.writerow(['平面', pcd_time, ''])  # 初期値として空の成功率を記入

                    self.total_frames += 1  # フレームカウントを増やす
                    # 成功率を計算
                    if self.total_frames > 0:
                        success_rate = (self.successful_plane_detections / self.total_frames) * 100
                    else:
                        success_rate = 0.0
                    
                    # 成功率をCSVに書き込む
                    self.csv_writer.writerow(['成功率', '', success_rate])
                    # YOLOでスロープを検出
                    pil_img = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
                    results = self.model.predict(source=pil_img, device=self.device, verbose=False)

                    if results:
                        if results[0].masks:
                            rospy.loginfo("マスクがあります")
                            mask_confidences = results[0].boxes.conf
                            rospy.loginfo(f"信頼度: {mask_confidences[0]}")
                            if results[0].boxes and mask_confidences[0] > 0.8:
                                rospy.loginfo("信頼度が条件を満たしています")
                                # 法線ベクトルの b の値を取得（仮に normal[1] とします）
                                b = normal[1]  # normal は事前に計算されていると仮定
                                c = normal[2]
                                # 前の b の値が存在する場合に変動をチェック
                                if self.previous_b is not None:
                                    if self.previous_c is not None:
                                        #前後の差分の絶対値が0.01以上ならそこからスロープありと判定
                                        if abs(b - self.previous_b) > 0.01 and abs(c - self.previous_c) > 0.02:
                                            rospy.loginfo(f"法線ベクトルの b と c の変動が条件を満たしています: {b:.4f}")
                                            self.process_segmentation(results[0], self.color_image, self.depth_image, inliers)

                                # 現在の b と c の値を保存
                                self.previous_b = b
                                self.previous_c = c
                                
                    else:
                        rospy.logwarn("予測結果が存在しません")


                    if plane_segments:
                        combined_points = np.vstack([np.asarray(plane.points) for plane in plane_segments])
                        combined_colors = np.vstack([np.asarray(plane.colors) for plane in plane_segments])
                        self.pointcloud.points = o3d.utility.Vector3dVector(combined_points)
                        self.pointcloud.colors = o3d.utility.Vector3dVector(combined_colors)
                        if not self.geom_added:
                            self.vis.add_geometry(self.pointcloud)
                            self.geom_added = True
                        else:
                            self.vis.update_geometry(self.pointcloud)

                    self.vis.poll_events()
                    self.vis.update_renderer()

                    # GPU使用率を表示
                    gpu_usage = self.get_gpu_usage()
                    #rospy.loginfo(f"GPU Usage: {gpu_usage:.2f}%")
                    self.color_image_copy = self.color_image.copy()  # 元のカラー画像のコピーを作成
                    cv2.imshow('bgr', self.color_image_copy)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
                    self.rate.sleep()

        finally:
            o3d.io.write_point_cloud("output2.ply", self.pointcloud)
            cv2.destroyAllWindows()
            self.vis.destroy_window()

    def process_segmentation(self, results, img_color, img_depth, inliers):
        if not results or not results.masks:
            rospy.logwarn("予測結果が存在しません")
            return
        start_seg = time.time()
        # 信頼度の取得と表示
        mask_confidences = results.boxes.conf
        print("信頼度:", mask_confidences)
        
        # GPU使用率を表示
        gpu_usage = self.get_gpu_usage()
        #rospy.loginfo(f"GPU Usage: {gpu_usage:.2f}%")

        # RANSACで検出した平面のインデックスを用いてマスクを適用
        masks = results.masks.to(self.device)
        x_numpy = masks[0].data.to('cpu').detach().numpy().copy()
        point = masks[0].xy
        point = np.array(point)

        # ３次元を２次元に変換する
        point = point.reshape(-1, point.shape[-1])  # 配列を2次元に変換

        # y座標が高い順にソート
        point = sorted(point, key=lambda x: x[1], reverse=True)

        # 上位70個のy座標が高い座標を取得
        top_points = point[:70]
        
        # ここでRANSACのインライヤーを使用してセグメンテーションをフィルタリング
        filtered_points = [p for p in top_points if int(p[0]) in inliers]  # ここでインライヤー条件を適用
        
        # 座標の表示
        for p in filtered_points:
            (u, v) = (int(p[0]), int(p[1]))
            # 画像内に指定したクラス(results[0]の境界線を赤点で描画
            cv2.circle(img_color, (u, v), 10, (0, 0, 255), -1)
        
        # y座標のピクセル値が高い順に上位70個を選ぶ
        top_points_y_sorted = sorted(filtered_points, key=lambda p: p[1], reverse=True)[:70]

        # x座標の中央値を計算するために、上位70個のうちx座標を絞る
        median_x_value = np.median([p[0] for p in top_points_y_sorted])
        median_x_candidates = [p for p in top_points_y_sorted if abs(p[0] - median_x_value) < 70]  # 70は調整可能

        # median_x_candidatesが空でないかを確認
        if median_x_candidates:
            median_x = int(np.median([p[0] for p in median_x_candidates]))
            median_y = int(np.median([p[1] for p in top_points_y_sorted]))

            # 中央の点を描画
            cv2.circle(img_color, (median_x, median_y), 10, (255, 0, 0), -1)
        else:
            rospy.logwarn("中央値を計算するための候補がありません")
        end_seg = time.time()
        seg_time = end_seg - start_seg
        rospy.loginfo(f"セマンティックセグメンテーションの処理時間: {seg_time:.4f}秒")
        self.csv_writer.writerow(['セグメンテーション', seg_time])

        # 映像出力rosbag playでやるときのみ外す
        cv2.imshow('slope_detection', img_color)
        cv2.waitKey(1)  # ウィンドウを更新するためのキー入力を待つ

if __name__ == '__main__':
    model_path = "/home/carsim05/slope_ws/src/ros_Golib/slope_edge_detection/scripts/best.pt"
    node = SR(model_path)
    node.process_pointcloud()  # メインループを開始
