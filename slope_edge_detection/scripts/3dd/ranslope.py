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

    def process_pointcloud(self):
        try:
            while not rospy.is_shutdown():
                if self.color_image is None or self.depth_image is None or self.intrinsics is None:
                    rospy.loginfo("Waiting for images and camera info...")
                    rospy.sleep(0.1)
                    continue

                # OpenCVで画像を処理
                o3d_color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
                img_depth = o3d.geometry.Image(self.depth_image)
                img_color = o3d.geometry.Image(o3d_color_image)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    img_color, img_depth, convert_rgb_to_intensity=False)
                pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    self.intrinsics.width, self.intrinsics.height,
                    self.intrinsics.K[0], self.intrinsics.K[4],
                    self.intrinsics.K[2], self.intrinsics.K[5])
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
                pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                
                # スロープ認識を追加
                self.process_image(self.color_image, self.depth_image, msg_depth=None)
                
                # 平面検出
                points = np.asarray(pcd.points)
                pass_x = (points[:, 0] > -1.0) & (points[:, 0] < 1.0)
                pass_y = (points[:, 1] > -1.0) & (points[:, 1] < 1.0)
                filtered_indices = np.where(pass_x & pass_y)[0]
                filtered_pcd = pcd.select_by_index(filtered_indices)
                filtered_pcd = filtered_pcd.voxel_down_sample(voxel_size=0.005)
                plane_model, inliers = filtered_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=100)
                inlier_cloud = filtered_pcd.select_by_index(inliers)
                a, b, c, d = plane_model
                normal = np.array([a, b, c])
                if np.abs(normal[1]) < 0.9:
                    inlier_cloud.paint_uniform_color([1.0, 0, 0])
                    angle_with_vertical = np.arccos(np.abs(normal[1]))
                    angle_degrees = np.degrees(angle_with_vertical)
                    rospy.loginfo(f"Ground plane tilt angle: {angle_degrees:.2f} degrees")
                    plane_segments = [inlier_cloud]
                else:
                    plane_segments = []
                
                # 結果の表示
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
                cv2.imshow('bgr', self.color_image)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                self.rate.sleep()
        finally:
            o3d.io.write_point_cloud("output2.ply", self.pointcloud)
            cv2.destroyAllWindows()
            self.vis.destroy_window()

    def process_image(self, img_color, img_depth, msg_depth):
        pil_img = PilImage.fromarray(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        results = self.model.predict(source=pil_img)

        if not results or not results[0].masks:
            rospy.logwarn("予測結果が存在しません")
            return
        
        mask_confidences = results[0].boxes.conf
        print("信頼度:", mask_confidences)
        
        if results[0].boxes and results[0].boxes.conf[0] > 0.8 and angle_degrees > 35:
            masks = results[0].masks
            x_numpy = masks[0].data.to('cpu').detach().numpy().copy()

            name = results[0].names
            point = masks[0].xy
            point = np.array(point)

            result = []
            for i in range(len(point)):
                for j in range(len(point[i])):
                    my_list = []
                    for k in range(len(point[i][j])):
                        my_list.append(point[i][j][k])
                    result.append(my_list)
            point = np.array(result)

            point = sorted(point, key=lambda x: x[1], reverse=True)
            top_points = point[:70]

            for i in range(len(top_points)):
                (u, v) = (int(top_points[i][0]), int(top_points[i][1]))
                cv2.circle(img_color, (u, v), 10, (0, 0, 255), -1)
            
            top_points_y_sorted = sorted(top_points, key=lambda p: p[1], reverse=True)[:70]

            median_x_value = np.median([p[0] for p in top_points_y_sorted])
            median_x_candidates = [p for p in top_points_y_sorted if abs(p[0] - median_x_value) < 70]

            (u1, v1) = (int(top_points_y_sorted[0][0]), int(top_points_y_sorted[0][1]))
            (u2, v2) = (int(top_points_y_sorted[1][0]), int(top_points_y_sorted[1][1]))

            median_x = int(np.median([p[0] for p in median_x_candidates]))
            median_y = int(np.median([p[1] for p in top_points_y_sorted]))

            cv2.circle(img_color, (median_x, median_y), 10, (255, 0, 0), -1)

            # Assume x, y, z are obtained from some calculation
            # x, y, z = 0, 0, 0  # Replace with actual calculation
            # self.csv_writer.writerow([self.frame_id, x, y, z, median_x, median_y])

            ts = TransformStamped()
            ts.header = msg_depth.header
            ts.child_frame_id = self.frame_id
            ts.transform.translation.x = x
            ts.transform.translation.y = y
            ts.transform.translation.z = z
            self.broadcaster.sendTransform((x, y, z), (0, 0, 0, 1), rospy.Time.now(), self.frame_id, msg_depth.header.frame_id)

if __name__ == '__main__':
    model_path = "/home/carsim/gakuhari_ws/src/ros_Golib/slope_edge_detection/scripts/best.pt"
    processor = PointCloudAndSlopeProcessor(model_path)
    processor.process_pointcloud()
