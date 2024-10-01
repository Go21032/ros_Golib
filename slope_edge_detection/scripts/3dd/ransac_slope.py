# #!/usr/bin/env python3
# import rospy
# from sensor_msgs.msg import Image, CameraInfo
# from cv_bridge import CvBridge, CvBridgeError
# import numpy as np
# import open3d as o3d
# import cv2
# from message_filters import Subscriber, ApproximateTimeSynchronizer
# from tf import TransformBroadcaster
# from geometry_msgs.msg import TransformStamped
# from ultralytics import YOLO
# from datetime import datetime
# #import csv
# from PIL import Image as PilImage
# import sys

# class SlopeDetection:

#     def __init__(self, model_path):
#         self.color_image = None
#         self.depth_image = None
#         self.intrinsics = None
#         self.model = YOLO(model_path)
#         self.bridge = CvBridge()
#         self.frame_id = 'slope'
        
#         # トピックの購読と同期
#         self.sub_info = Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo)
#         self.sub_color = Subscriber('/camera/color/image_raw', Image)
#         self.sub_depth = Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        
#         self.ts = ApproximateTimeSynchronizer([self.sub_info, self.sub_color, self.sub_depth], 20, 0.1)
#         self.ts.registerCallback(self.images_callback)
        
#         self.broadcaster = TransformBroadcaster()
        
#         # CSV file
#         # current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
#         # file_name = f'slope_3次元座標{current_time}.csv'
#         # self.csv_file = open(file_name, 'w', newline='')
#         # self.csv_writer = csv.writer(self.csv_file)
#         # self.csv_writer.writerow(['name', 'X', 'Y', 'Z', 'median_x', 'median_y'])

#     def images_callback(self, msg_info, msg_color, msg_depth):
#         global img_color, img_depth, intrinsics
#         try:
#             img_color = self.bridge.imgmsg_to_cv2(msg_color, "bgr8")
#             img_depth = self.bridge.imgmsg_to_cv2(msg_depth, "16UC1")
#             intrinsics = msg_info
#         except CvBridgeError as e:
#             rospy.logwarn(str(e))
#             return
#         except Exception as e:
#             print(f"変換エラー: {e}")
        
#         if img_color.shape[0:2] != img_depth.shape[0:2]:
#             rospy.logwarn('カラーと深度の画像サイズが異なる')
#             return
#         if img_color is not None:
#             print(f"img_color type: {type(img_color)}, shape: {img_color.shape}")
#         else:
#             print("img_color is None")
#         # 画像変換を事前に行う
#         o3d_img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
#         img_depth = o3d.geometry.Image(img_depth)
#         img_color = o3d.geometry.Image(o3d_img_color)
#         rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth, convert_rgb_to_intensity=False)

#         pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
#             intrinsics.width, intrinsics.height, intrinsics.K[0], intrinsics.K[4], intrinsics.K[2], intrinsics.K[5])

#         pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
#         pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

#         # フィルタリングのインデックスを毎回計算
#         points = np.asarray(pcd.points)
#         pass_x = (points[:, 0] > -1.0) & (points[:, 0] < 1.0)
#         pass_y = (points[:, 1] > -1.0) & (points[:, 1] < 1.0)
#         filtered_indices = np.where(pass_x & pass_y)[0]

#         # フィルタリング後の点群を更新
#         filtered_pcd = pcd.select_by_index(filtered_indices)
#         filtered_pcd = filtered_pcd.voxel_down_sample(voxel_size=0.005)  # 点群をダウンサンプリング

#         # 地面の平面を検出
#         plane_model, inliers = filtered_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=100)
#         inlier_cloud = filtered_pcd.select_by_index(inliers)

#         # 法線が特定の閾値を持つ場合に平面として認識
#         a, b, c, d = plane_model
#         normal = np.array([a, b, c])
        
#         # 地面の傾きを計算
#         if np.abs(normal[1]) < 0.9:  # 法線が水平すぎない場合
#             inlier_cloud.paint_uniform_color([1.0, 0, 0])

#             # 傾きを計算
#             angle_with_vertical = np.arccos(np.abs(normal[1]))  # y軸との角度
#             angle_degrees = np.degrees(angle_with_vertical)
#             rospy.loginfo(f"Ground plane tilt angle: {angle_degrees:.2f} degrees")

#             plane_segments = [inlier_cloud]
#         else:
#             plane_segments = []

#         # 点群を更新
#         if plane_segments:  # plane_segmentsが空でない場合のみ処理
#             combined_points = np.vstack([np.asarray(plane.points) for plane in plane_segments])
#             combined_colors = np.vstack([np.asarray(plane.colors) for plane in plane_segments])

#             pointcloud.points = o3d.utility.Vector3dVector(combined_points)
#             pointcloud.colors = o3d.utility.Vector3dVector(combined_colors)

#             if not geom_added:
#                 vis.add_geometry(pointcloud)
#                 geom_added = True
#             else:
#                 vis.update_geometry(pointcloud)

#         # 信頼度の取得と表示
#         mask_confidences = results[0].boxes.conf
#         print("信頼度:", mask_confidences)
           
#         if angle_degrees >= 40.0 and results and results[0].boxes and results[0].boxes.conf[0] > 0.8:
#             # YOLOによるスロープ検出
#             pil_img = PilImage.fromarray(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
#             results = self.model.predict(source=pil_img)
#             # 予測結果がない場合の処理
#             if not results or not results[0].masks:
#                 rospy.logwarn("予測結果が存在しません")
#                 return
                
#             masks = results[0].masks
#             x_numpy = masks[0].data.to('cpu').detach().numpy().copy()

#             name = results[0].names
#             point = masks[0].xy
#             point = np.array(point)

#             # ３次元を２次元に変換する
#             result = []
#             for i in range(len(point)):
#                 for j in range(len(point[i])):
#                     my_list = []
#                     for k in range(len(point[i][j])):
#                         my_list.append(point[i][j][k])
#                     result.append(my_list)
#             point = np.array(result)

#             #y座標が高い順にソート
#             point = sorted(point, key=lambda x: x[1], reverse=True)

#             # 上位50個のy座標が高い座標を取得
#             top_points = point[:70]

#             # 座標の表示
#             for i in range(len(top_points)):
#                 (u, v) = (int(top_points[i][0]), int(top_points[i][1]))
#                 # 画像内に指定したクラス(results[0]の境界線を赤点で描画
#                 cv2.circle(img_color, (u, v), 10, (0, 0, 255), -1)
            
#             #右より
#             # y座標のピクセル値が高い順に上位50個を選ぶ
#             top_points_y_sorted = sorted(top_points, key=lambda p: p[1], reverse=True)[:70]

#             # x座標の中央値を計算するために、上位50個のうちx座標を絞る
#             # 絞る条件を少し緩めるために、中央値に近い範囲を考慮
#             median_x_value = np.median([p[0] for p in top_points_y_sorted])
#             median_x_candidates = [p for p in top_points_y_sorted if abs(p[0] - median_x_value) < 70]  # 50は調整可能

#             # y座標が高い2つの点を選ぶ
#             (u1, v1) = (int(top_points_y_sorted[0][0]), int(top_points_y_sorted[0][1]))
#             (u2, v2) = (int(top_points_y_sorted[1][0]), int(top_points_y_sorted[1][1]))

#             # x座標とy座標の中央値をそれぞれ算出
#             median_x = int(np.median([p[0] for p in median_x_candidates]))
#             median_y = int(np.median([p[1] for p in top_points_y_sorted]))

#             # 中央の点を描画
#             cv2.circle(img_color, (median_x, median_y), 10, (255, 0, 0), -1)                 
                        
#             #映像出力rosbag playでやるときのみ外す
#             # cv2.imshow('color', img_color)
#             # if cv2.waitKey(1) & 0xFF == ord('q'):
#             #     rospy.signal_shutdown('closed')
#             #     return
#             # Write to CSV
#             # self.csv_writer.writerow([self.frame_id, x, y, z, median_x, median_y])

#             # Broadcast transform
#             ts = TransformStamped()
#             ts.header = msg_depth.header
#             ts.child_frame_id = self.frame_id
#             ts.transform.translation.x = x
#             ts.transform.translation.y = y
#             ts.transform.translation.z = z
#             self.broadcaster.sendTransform((x, y, z), (0, 0, 0, 1), rospy.Time.now(), self.frame_id, msg_depth.header.frame_id)

#     # def __del__(self):
#     #     self.csv_file.close()

# if __name__ == '__main__':
#     rospy.init_node('slope_detection')
#     model_path = "/home/carsim/gakuhari_ws/src/ros_Golib/slope_edge_detection/scripts/best.pt"
#     node = SlopeDetection(model_path)
#     vis = o3d.visualization.Visualizer()
#     vis.create_window('PCD', width=960, height=720)
#     pointcloud = o3d.geometry.PointCloud()
#     geom_added = False
#     rate = rospy.Rate(10)  # 10Hzに設定
    
#     try:
#         while not rospy.is_shutdown():
#             if img_color is None or img_depth is None or intrinsics is None:
#                 rospy.loginfo("Waiting for images and camera info...")
#                 rospy.sleep(0.1)
#                 continue

#             vis.poll_events()
#             vis.update_renderer()
            
#             # ここで型を確認
#             if isinstance(img_color, np.ndarray):
#                 cv2.imshow('bgr', img_color)
#             else:
#                 rospy.logwarn("img_color is not a valid numpy array")

#             key = cv2.waitKey(1)
#             if key == ord('q'):
#                 break
#             rate.sleep()

#     finally:
#         o3d.io.write_point_cloud("output2.ply", pointcloud)
#         cv2.destroyAllWindows()
#         vis.destroy_window()
