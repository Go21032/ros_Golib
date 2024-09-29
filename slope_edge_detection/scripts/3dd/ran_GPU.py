import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d
import datetime
import cv2
# CUDAモジュールをインポート
import cv2.cuda as cv2_cuda

# グローバル変数の定義
color_image = None
depth_image = None
intrinsics = None
filtered_indices = None

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
pointcloud = o3d.geometry.PointCloud()
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

        # CUDAを利用して画像を処理
        gpu_color_image = cv2_cuda.GpuMat()
        gpu_color_image.upload(color_image)
        gpu_rgb_image = cv2_cuda.cvtColor(gpu_color_image, cv2.COLOR_BGR2RGB)
        o3d_color_image = gpu_rgb_image.download()

        # Open3Dの処理
        img_depth = o3d.geometry.Image(depth_image)
        img_color = o3d.geometry.Image(o3d_color_image)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth, convert_rgb_to_intensity=False)

        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height, intrinsics.K[0], intrinsics.K[4], intrinsics.K[2], intrinsics.K[5])

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # フィルタリングのインデックスを一度計算
        points = np.asarray(pcd.points)
        pass_x = (points[:, 0] > -1.0) & (points[:, 0] < 1.0)
        pass_y = (points[:, 1] > -1.0) & (points[:, 1] < 1.0)
        filtered_indices = np.where(pass_x & pass_y)[0]

        # フィルタリング後の点群を更新
        filtered_pcd = pcd.select_by_index(filtered_indices)

        # 平面検出の実行
        if len(filtered_pcd.points) > 0:
            plane_model, inliers = filtered_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
            # 平面モデルの係数を出力
            [a, b, c, d] = plane_model

            # # 平面の法線ベクトル
            # normal_vector = np.array([a, b, c])
            # # 傾きの計算（Z軸に対する角度）
            # z_axis = np.array([0, 0, 1])
            # angle = np.arccos(np.dot(normal_vector, z_axis) / np.linalg.norm(normal_vector))
            # angle_degrees = np.degrees(angle)
            # rospy.loginfo(f"平面の傾き: {angle_degrees:.2f}度")

            # インライアの点を抽出して色を付ける
            inlier_cloud = filtered_pcd.select_by_index(inliers)
            inlier_cloud.paint_uniform_color([1.0, 0, 0])
            # 平面以外の点を抽出
            outlier_cloud = filtered_pcd.select_by_index(inliers, invert=True)

            pointcloud.points = filtered_pcd.points
            pointcloud.colors = filtered_pcd.colors

            # リアルタイムで可視化
            if not geom_added:
                vis.add_geometry(pointcloud)
                geom_added = True
            else:
                vis.update_geometry(pointcloud)

            vis.poll_events()
            vis.update_renderer()

        process_time = datetime.datetime.now() - dt0
        rate.sleep()

finally:
    o3d.io.write_point_cloud("output2.ply", pointcloud)
    cv2.destroyAllWindows()
    vis.destroy_window()
