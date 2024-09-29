import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d
import datetime
import cv2

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
vis.create_window('PCD', width=640, height=480)
pointcloud = o3d.geometry.PointCloud()
geom_added = False

try:
    while not rospy.is_shutdown():
        if color_image is None or depth_image is None or intrinsics is None:
            continue

        dt0 = datetime.datetime.now()

        o3d_color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        img_depth = o3d.geometry.Image(depth_image)
        img_color = o3d.geometry.Image(o3d_color_image)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth, convert_rgb_to_intensity=False)

        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height, intrinsics.K[0], intrinsics.K[4], intrinsics.K[2], intrinsics.K[5])

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # NumPy配列に変換
        points = np.asarray(pcd.points)

        # x座標でのフィルタリング
        pass_x = (points[:, 0] > -10.0) & (points[:, 0] < 10.0)
        cloud_passthrough_x = points[pass_x]

        # y座標でのフィルタリング
        pass_y = (cloud_passthrough_x[:, 1] > -10.0) & (cloud_passthrough_x[:, 1] < 10.0)
        cloud_passthrough = cloud_passthrough_x[pass_y]

        # フィルタリング後の点群を更新
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(cloud_passthrough)
        filtered_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[pass_x][pass_y])

        pointcloud.points = filtered_pcd.points
        pointcloud.colors = filtered_pcd.colors

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
