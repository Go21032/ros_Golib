import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d
import cv2

class PointCloudProcessor:
    def __init__(self):
        self.color_image = None
        self.depth_image = None
        self.intrinsics = None
        self.bridge = CvBridge()
        rospy.init_node('pointcloud_node')
        rospy.Subscriber('/camera/color/image_raw', Image, self.callback, callback_args='color')
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.callback, callback_args='depth')
        rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, self.callback, callback_args='info')
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window('PCD', width=640, height=480)
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
                if np.abs(normal[1]) < 0.9:
                    inlier_cloud.paint_uniform_color([1.0, 0, 0])
                    angle_with_vertical = np.arccos(np.abs(normal[1]))
                    angle_degrees = np.degrees(angle_with_vertical)
                    rospy.loginfo(f"Ground plane tilt angle: {angle_degrees:.2f} degrees")
                    plane_segments = [inlier_cloud]
                else:
                    plane_segments = []
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
if __name__ == '__main__':
    processor = PointCloudProcessor()
    processor.process_pointcloud()