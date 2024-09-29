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
rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, camera_info_callback)

# Open3Dの設定
vis = o3d.visualization.Visualizer()
vis.create_window('PCD', width=640, height=480)
device = o3d.core.Device("CUDA:0")  # デバイス指定
pointcloud = o3d.t.geometry.PointCloud(device=device)  # デバイス指定
geom_added = False

# ループ内の処理を制限
rate = rospy.Rate(10)  # 10Hzに設定

try:
    while not rospy.is_shutdown():
        if color_image is None or depth_image is None or intrinsics is None:
            rospy.loginfo("Waiting for images and camera info...")
            rospy.sleep(0.1)
            continue

        dt0 = datetime.datetime.now()

        # 画像変換を事前に行う
        o3d_color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        img_depth = o3d.geometry.Image(depth_image)
        img_color = o3d.geometry.Image(o3d_color_image)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth, convert_rgb_to_intensity=False)

        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height, intrinsics.K[0], intrinsics.K[4], intrinsics.K[2], intrinsics.K[5])

        # Tensor-based PointCloudでGPUを利用
        t_rgbd = o3d.t.geometry.RGBDImage.from_legacy(rgbd, device)
        t_intrinsic = o3d.core.Tensor(pinhole_camera_intrinsic.intrinsic_matrix, device=device)
        t_pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(t_rgbd, t_intrinsic)

        # フィルタリングのインデックスを一度計算
        points = t_pcd.point.positions.numpy()
        pass_x = (points[:, 0] > -1.0) & (points[:, 0] < 1.0)
        pass_y = (points[:, 1] > -1.0) & (points[:, 1] < 1.0)
        filtered_indices = np.where(pass_x & pass_y)[0]

        # フィルタリング後の点群を更新
        mask = o3d.core.Tensor(filtered_indices, dtype=o3d.core.Dtype.Bool, device=device)
        filtered_pcd = t_pcd.select_by_mask(mask)

        # 平面検出の実行
        if len(filtered_pcd.point.positions) > 0:
            plane_model, inliers = filtered_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
            inlier_cloud = filtered_pcd.select_by_mask(inliers)
            inlier_cloud.paint_uniform_color([1.0, 0, 0])
            outlier_cloud = filtered_pcd.select_by_mask(inliers, invert=True)

            pointcloud.point.positions = filtered_pcd.point.positions
            pointcloud.point.colors = filtered_pcd.point.colors

            if not geom_added:
                vis.add_geometry(pointcloud.to_legacy())
                geom_added = True
            else:
                vis.update_geometry(pointcloud.to_legacy())

            vis.poll_events()
            vis.update_renderer()

        process_time = datetime.datetime.now() - dt0
        rate.sleep()

finally:
    o3d.io.write_point_cloud("output2.ply", pointcloud.to_legacy())
    cv2.destroyAllWindows()
    vis.destroy_window()
