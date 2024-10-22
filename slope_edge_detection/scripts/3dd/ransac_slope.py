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
from message_filters import ApproximateTimeSynchronizer, Subscriber

class SlopeDetection:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        self.frame_id = 'slope'

        self.color_image = None  
        self.depth_image = None  
        self.intrinsics = None 
        
        self.sub_info = Subscriber('camera/aligned_depth_to_color/camera_info', CameraInfo)
        self.sub_color = Subscriber('camera/color/image_raw', Image)
        self.sub_depth = Subscriber('camera/aligned_depth_to_color/image_raw', Image)
        
        self.ts = ApproximateTimeSynchronizer([self.sub_info, self.sub_color, self.sub_depth], 20, 0.1)
        self.ts.registerCallback(self.callback)
        
        self.broadcaster = TransformBroadcaster()
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window('PCD', width=640, height=480)
        self.pointcloud = o3d.geometry.PointCloud()
        self.geom_added = False
                
        self.rate = rospy.Rate(10)
        
    def callback(self, msg_info, msg_color, msg_depth):
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg_color, 'bgr8')
            self.depth_image = self.bridge.imgmsg_to_cv2(msg_depth, 'passthrough').copy()
        except CvBridgeError as e:
            rospy.logwarn(str(e))
            return
        except Exception as e:
            print(f"変換エラー: {e}")
            return

        if self.color_image.shape[0:2] != self.depth_image.shape[0:2]:
            rospy.logwarn('カラーと深度の画像サイズが異なる')
            return

        pil_img = PilImage.fromarray(cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB))
        results = self.model.predict(source=pil_img)

        if results and results[0].masks:
            self.process_segmentation(results[0], self.color_image, self.depth_image, msg_depth)
        else:
            rospy.logwarn("予測結果またはマスクが見つかりません")


    def process_pointcloud(self):
        try:
            while not rospy.is_shutdown():
                if self.color_image is None or self.depth_image is None or self.intrinsics is None:
                    rospy.loginfo("Waiting for images and camera info...")
                    rospy.sleep(0.1)
                    continue

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
                
                rospy.loginfo(f"PointCloud Points: {len(pcd.points)}")
                rospy.loginfo(f"PointCloud Colors: {len(pcd.colors)}")
                if len(pcd.points) == 0:
                    rospy.logwarn("PointCloudが空です。")
                    continue
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
                
                angle_degrees = 0
                if np.abs(normal[1]) < 0.9:
                    inlier_cloud.paint_uniform_color([1.0, 0, 0])
                    angle_with_vertical = np.arccos(np.abs(normal[1]))
                    angle_degrees = np.degrees(angle_with_vertical)
                    rospy.loginfo(f"Ground plane tilt angle: {angle_degrees:.2f} degrees")
                    plane_segments = [inlier_cloud]
                else:
                    plane_segments = []
                
                pil_img = PilImage.fromarray(cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB))
                results = self.model.predict(source=pil_img)

                if results:
                    if results[0].masks:
                        rospy.loginfo("マスクがあります")
                        mask_confidences = results[0].boxes.conf
                        rospy.loginfo(f"信頼度: {mask_confidences[0]}")
                        if results[0].boxes and mask_confidences[0] > 0.8:
                            rospy.loginfo("信頼度が条件を満たしています")
                            if angle_degrees > 31:
                                rospy.loginfo(f"角度が条件を満たしています:{angle_degrees:.2f} degrees")
                                if self.depth_image is not None:
                                    self.process_segmentation(results[0], self.color_image, self.depth_image)
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

                cv2.imshow('Color Image', self.color_image)
                key = cv2.waitKey(1)
                
                if key == ord('q'):
                    break
                self.rate.sleep()
        finally:
            o3d.io.write_point_cloud("output2.ply", self.pointcloud)
            cv2.destroyAllWindows()
            self.vis.destroy_window()

    def process_segmentation(self, result, color_image, depth_image, msg_depth):
        pil_img = PilImage.fromarray(cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB))
        results = self.model.predict(source=pil_img)
        if not result.masks:
            rospy.logwarn("マスクが見つかりません")
            return

        masks = result.masks
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

        # トップ70ポイントを取得
        point = sorted(point, key=lambda x: x[1], reverse=True)
        top_points = point[:70]

        for i in range(len(top_points)):
            (u, v) = (int(top_points[i][0]), int(top_points[i][1]))
            cv2.circle(color_image, (u, v), 10, (0, 0, 255), -1)

        top_points_y_sorted = sorted(top_points, key=lambda p: p[1], reverse=True)[:70]

        if len(top_points_y_sorted) == 0:
            rospy.logwarn("トップポイントが見つかりません")
            return

        median_x_value = np.median([p[0] for p in top_points_y_sorted])
        median_x_candidates = [p for p in top_points_y_sorted if abs(p[0] - median_x_value) < 70]

        if len(median_x_candidates) == 0:
            rospy.logwarn("中央値候補が見つかりません")
            return

        median_x = int(np.median([p[0] for p in median_x_candidates]))
        median_y = int(np.median([p[1] for p in top_points_y_sorted]))

        rospy.loginfo(f"中央値の座標: ({median_x}, {median_y})")
        cv2.circle(color_image, (median_x, median_y), 10, (255, 0, 0), -1)

        # ウィンドウ名が異なる可能性を考慮して、ウィンドウを再作成
        cv2.namedWindow('Slope Segmentation', cv2.WINDOW_NORMAL)
        cv2.imshow('Slope Segmentation', color_image)
        cv2.waitKey(1)

        ts = TransformStamped()
        ts.header = msg_depth.header
        ts.child_frame_id = self.frame_id
        ts.transform.translation.x = 0  # 適切な値を設定
        ts.transform.translation.y = 0  # 適切な値を設定
        ts.transform.translation.z = 0  # 適切な値を設定
        self.broadcaster.sendTransform((0, 0, 0), (0, 0, 0, 1), rospy.Time.now(), self.frame_id, msg_depth.header.frame_id)


if __name__ == '__main__':
    rospy.init_node('slope_detection')
    model_path = "/home/carsim05/slope_ws/src/ros_Golib/slope_edge_detection/scripts/best.pt"
    node = SlopeDetection(model_path)
    node.process_pointcloud() 
    rospy.spin()

    git config --local user.name [Go21032] git config --local user.email [a221032@g.kogakuin.jp]

