import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d
import cv2
import datetime

# グローバル変数の定義
color_image = None
depth_image = None
intrinsics = None

# コールバック関数
def color_callback(msg):
    global color_image
    color_image = bridge.imgmsg_to_cv2(msg, "bgr8")

def depth_callback(msg):
    global depth_image
    depth_image = bridge.imgmsg_to_cv2(msg, "16UC1")

def camera_info_callback(msg):
    global intrinsics
    intrinsics = msg

# ROSノードの初期化
rospy.init_node('pointcloud_node')
bridge = CvBridge()

# トピックの購読
rospy.Subscriber('/camera/color/image_raw', Image, color_callback)
rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, depth_callback)
rospy.Subscriber('/camera/color/camera_info', CameraInfo, camera_info_callback)

# Open3Dの設定
vis = o3d.visualization.Visualizer()
vis.create_window('PCD', width=960, height=720)
pointcloud = o3d.geometry.PointCloud()
geom_added = False

# 深度スケール設定
depth_scale = 0.001  # 深度画像をメートルに変換するスケール

try:
    while not rospy.is_shutdown():
        if color_image is None or depth_image is None or intrinsics is None:
            rospy.sleep(0.1)
            continue

        dt0 = datetime.datetime.now()

        # 深度画像のスケールを適用
        depth_image_scaled = depth_image * depth_scale

        o3d_color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        img_depth = o3d.geometry.Image(depth_image_scaled.astype(np.float32))
        img_color = o3d.geometry.Image(o3d_color_image)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            img_color, img_depth, depth_scale=5.0, depth_trunc=5.0, convert_rgb_to_intensity=False)

        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height, intrinsics.K[0], intrinsics.K[4], intrinsics.K[2], intrinsics.K[5])

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        pointcloud.points = pcd.points
        pointcloud.colors = pcd.colors

        if not geom_added:
            vis.add_geometry(pointcloud)
            geom_added = True
        else:
            vis.update_geometry(pointcloud)

        vis.poll_events()
        vis.update_renderer()

        cv2.imshow('bgr', color_image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        process_time = datetime.datetime.now() - dt0
finally:
    o3d.io.write_point_cloud("output2.ply", pointcloud)
    cv2.destroyAllWindows()
    vis.destroy_window()
