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
        self.bridge = CvBridge()
        self.model = YOLO(model_path)
        self.color_image = None
        self.depth_image = None
        self.intrinsics = None
        self.depth_header = None
        self.frame_id = 'slope'
        
        rospy.init_node('pointcloud_and_slope_node')
        rospy.Subscriber('/camera/color/image_raw', Image, self.callback, callback_args='color')
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.callback, callback_args='depth')
        rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, self.callback, callback_args='info')
        
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window('PCD', width=640, height=480)
        self.pointcloud = o3d.geometry.PointCloud()
        self.geom_added = False
        
        self.broadcaster = TransformBroadcaster()
        
        self.rate = rospy.Rate(10)
        
    def callback(self, msg, msg_type):
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
                    rospy.loginfo("予測結果があります")
                    if results[0].masks:
                        rospy.loginfo("マスクがあります")
                        mask_confidences = results[0].boxes.conf
                        rospy.loginfo(f"信頼度: {mask_confidences[0]}")
                        if results[0].boxes and mask_confidences[0] > 0.8:
                            rospy.loginfo("信頼度が条件を満たしています")
                            if angle_degrees > 32:
                                rospy.loginfo("角度が条件を満たしています")
                                if self.depth_header is not None:
                                    self.process_segmentation(results[0], self.color_image, self.depth_header)
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

    def process_segmentation(self, result, img_color, depth_header):
        if not result.masks:
            rospy.logwarn("マスクが見つかりません")
            return
        
        masks = result.masks
        x_numpy = masks[0].data.to('cpu').detach().numpy().copy()

        name = result.names
        point = masks[0].xy
        point = np.array(point)

        result_list = []
        for i in range(len(point)):
            for j in range(len(point[i])):
                my_list = []
                for k in range(len(point[i][j])):
                    my_list.append(point[i][j][k])
                result_list.append(my_list)
        point = np.array(result_list)

        point = sorted(point, key=lambda x: x[1], reverse=True)
        top_points = point[:70]

        for i in range(len(top_points)):
            (u, v) = (int(top_points[i][0]), int(top_points[i][1]))
            cv2.circle(img_color, (u, v), 10, (0, 0, 255), -1)

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
        cv2.circle(img_color, (median_x, median_y), 10, (255, 0, 0), -1)
        
        cv2.imshow('Slope Segmentation', img_color)
        cv2.waitKey(1)

        ts = TransformStamped()
        ts.header = depth_header
        ts.child_frame_id = self.frame_id
        ts.transform.translation.x = 0  # 適切な値を設定
        ts.transform.translation.y = 0  # 適切な値を設定
        ts.transform.translation.z = 0  # 適切な値を設定
        self.broadcaster.sendTransform((0, 0, 0), (0, 0, 0, 1), rospy.Time.now(), self.frame_id, depth_header.frame_id)

if __name__ == '__main__':
    model_path = "/home/carsim/gakuhari_ws/src/ros_Golib/slope_edge_detection/scripts/best.pt"
    processor = PointCloudAndSlopeProcessor(model_path)
    processor.process_pointcloud()
