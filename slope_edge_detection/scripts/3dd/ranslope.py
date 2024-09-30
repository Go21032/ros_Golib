import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import open3d as o3d
import cv2
from ultralytics import YOLO
from PIL import Image as PilImage
from datetime import datetime
from geometry_msgs.msg import TransformStamped
from tf import TransformBroadcaster

class PointCloudAndSlopeProcessor:
    def __init__(self, model_path):
        # 初期化
        self.bridge = CvBridge()
        self.model = YOLO(model_path)
        self.color_image = None
        self.depth_image = None
        self.intrinsics = None
        self.frame_id = 'slope'
        
        # ROSノードとサブスクライバの設定
        rospy.init_node('pointcloud_and_slope_node')
        rospy.Subscriber('/camera/color/image_raw', Image, self.callback, callback_args='color')
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.callback, callback_args='depth')
        rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, self.callback, callback_args='info')
        
        # Open3Dの設定
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window('PCD', width=640, height=480)
        self.pointcloud = o3d.geometry.PointCloud()
        self.geom_added = False
        
        # TFブロードキャスタの設定
        self.broadcaster = TransformBroadcaster()
        
        # CSVファイルの設定
        # current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        # file_name = f'slope_3次元座標{current_time}.csv'
        # self.csv_file = open(file_name, 'w', newline='')
        # self.csv_writer = csv.writer(self.csv_file)
        # self.csv_writer.writerow(['name', 'X', 'Y', 'Z', 'median_x', 'median_y'])
        
        self.rate = rospy.Rate(10)
        
    def callback(self, msg, msg_type):
        # コールバック関数
        try:
            if msg_type == 'color':
                self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            elif msg_type == 'depth':
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            elif msg_type == 'info':
                self.intrinsics = msg
        except CvBridgeError as e:
            rospy.logwarn(str(e))

    def process_pointcloud_and_slope(self):
        # メイン処理ループ
        try:
            while not rospy.is_shutdown():
                if self.color_image is None or self.depth_image is None or self.intrinsics is None:
                    rospy.loginfo("Waiting for images and camera info...")
                    rospy.sleep(0.1)
                    continue
                
                # Open3D用の画像変換
                o3d_color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
                img_depth = o3d.geometry.Image(self.depth_image)
                img_color = o3d.geometry.Image(o3d_color_image)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    img_color, img_depth, convert_rgb_to_intensity=False)
                
                pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    self.intrinsics.width, self.intrinsics.height,
                    self.intrinsics.K[0], self.intrinsics.K[4],
                    self.intrinsics.K[2], self.intrinsics.K[5])
                
                # PointCloud生成
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
                pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                
                # 平面検出
                plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=100)
                a, b, c, d = plane_model
                normal = np.array([a, b, c])
                angle_with_vertical = np.arccos(np.abs(normal[1]))
                angle_degrees = np.degrees(angle_with_vertical)
                
                # セマンティックセグメンテーション
                pil_img = PilImage.fromarray(cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB))
                results = self.model.predict(source=pil_img)

                if results and results[0].boxes and results[0].boxes.conf[0] > 0.8:
                    # 信頼度が80%以上のオブジェクトがある
                    if angle_degrees > 40:
                        # 角度が40度以上の場合
                        masks = results[0].masks
                        point = masks[0].xy
                        point = np.array(point)

                        point = sorted(point, key=lambda x: x[1], reverse=True)
                        top_points = point[:70]

                        top_points_y_sorted = sorted(top_points, key=lambda p: p[1], reverse=True)[:70]
                        median_x_value = np.median([p[0] for p in top_points_y_sorted])
                        median_x_candidates = [p for p in top_points_y_sorted if abs(p[0] - median_x_value) < 70]

                        median_x = int(np.median([p[0] for p in median_x_candidates]))
                        median_y = int(np.median([p[1] for p in top_points_y_sorted]))

                        # 結果の出力
                        # x, y, z = 0, 0, 0  # 実際の計算に置き換える
                        # self.csv_writer.writerow([self.frame_id, x, y, z, median_x, median_y])

                        ts = TransformStamped()
                        ts.header.stamp = rospy.Time.now()
                        ts.header.frame_id = self.intrinsics.header.frame_id
                        ts.child_frame_id = self.frame_id
                        ts.transform.translation.x = x
                        ts.transform.translation.y = y
                        ts.transform.translation.z = z
                        self.broadcaster.sendTransform(ts)

                self.rate.sleep()
        finally:
            # self.csv_file.close()
            o3d.io.write_point_cloud("output2.ply", self.pointcloud)
            cv2.destroyAllWindows()
            self.vis.destroy_window()

if __name__ == '__main__':
    model_path = "/home/carsim/gakuhari_ws/src/ros_Golib/slope_edge_detection/scripts/best.pt"
    processor = PointCloudAndSlopeProcessor(model_path)
    processor.process_pointcloud_and_slope()
