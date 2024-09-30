import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d
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

        # 画像変換を事前に行う
        o3d_color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        img_depth = o3d.geometry.Image(depth_image)
        img_color = o3d.geometry.Image(o3d_color_image)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth, convert_rgb_to_intensity=False)

        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height, intrinsics.K[0], intrinsics.K[4], intrinsics.K[2], intrinsics.K[5])

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # フィルタリングのインデックスを毎回計算
        points = np.asarray(pcd.points)
        pass_x = (points[:, 0] > -1.0) & (points[:, 0] < 1.0)
        pass_y = (points[:, 1] > -1.0) & (points[:, 1] < 1.0)
        filtered_indices = np.where(pass_x & pass_y)[0]

        # フィルタリング後の点群を更新
        filtered_pcd = pcd.select_by_index(filtered_indices)
        filtered_pcd = filtered_pcd.voxel_down_sample(voxel_size=0.005)  # 点群をダウンサンプリング

        # 地面の平面を検出
        plane_model, inliers = filtered_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=100)
        inlier_cloud = filtered_pcd.select_by_index(inliers)

        # 法線が特定の閾値を持つ場合に平面として認識
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        
        # 地面の傾きを計算
        if np.abs(normal[1]) < 0.9:  # 法線が水平すぎない場合
            inlier_cloud.paint_uniform_color([1.0, 0, 0])

            # 傾きを計算
            angle_with_vertical = np.arccos(np.abs(normal[1]))  # y軸との角度
            angle_degrees = np.degrees(angle_with_vertical)
            rospy.loginfo(f"Ground plane tilt angle: {angle_degrees:.2f} degrees")

            plane_segments = [inlier_cloud]
        else:
            plane_segments = []

        # 点群を更新
        if plane_segments:  # plane_segmentsが空でない場合のみ処理
            combined_points = np.vstack([np.asarray(plane.points) for plane in plane_segments])
            combined_colors = np.vstack([np.asarray(plane.colors) for plane in plane_segments])

            pointcloud.points = o3d.utility.Vector3dVector(combined_points)
            pointcloud.colors = o3d.utility.Vector3dVector(combined_colors)

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
        rate.sleep()

finally:
    o3d.io.write_point_cloud("output2.ply", pointcloud)
    cv2.destroyAllWindows()
    vis.destroy_window()



"""
# 地面一つだけ
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d
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

        # 画像変換を事前に行う
        o3d_color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        img_depth = o3d.geometry.Image(depth_image)
        img_color = o3d.geometry.Image(o3d_color_image)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth, convert_rgb_to_intensity=False)

        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height, intrinsics.K[0], intrinsics.K[4], intrinsics.K[2], intrinsics.K[5])

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # フィルタリングのインデックスを毎回計算
        points = np.asarray(pcd.points)
        pass_x = (points[:, 0] > -1.0) & (points[:, 0] < 1.0)
        pass_y = (points[:, 1] > -1.0) & (points[:, 1] < 1.0)
        filtered_indices = np.where(pass_x & pass_y)[0]

        # フィルタリング後の点群を更新
        filtered_pcd = pcd.select_by_index(filtered_indices)
        filtered_pcd = filtered_pcd.voxel_down_sample(voxel_size=0.005)  # 点群をダウンサンプリング

        # 平面検出の実行
        if len(filtered_pcd.points) > 0:
            plane_model, inliers = filtered_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=100)
            # 平面モデルの係数を出力
            [a, b, c, d] = plane_model
            inlier_cloud = filtered_pcd.select_by_index(inliers)
            # インライアの点を抽出して色を付ける
            inlier_cloud.paint_uniform_color([1.0, 0, 0])
            # 平面以外の点を抽出
            outlier_cloud = filtered_pcd.select_by_index(inliers, invert=True)

            # インライアとアウトライアの点群を結合
            combined_points = np.vstack((np.asarray(inlier_cloud.points), np.asarray(outlier_cloud.points)))
            combined_colors = np.vstack((np.asarray(inlier_cloud.colors), np.asarray(outlier_cloud.colors)))

            # 点群を更新
            pointcloud.points = o3d.utility.Vector3dVector(combined_points)
            pointcloud.colors = o3d.utility.Vector3dVector(combined_colors)

            if not geom_added:
                vis.add_geometry(pointcloud)
                geom_added = True
            else:
                vis.update_geometry(pointcloud)

            vis.poll_events()
            vis.update_renderer()

        rate.sleep()

finally:
    o3d.io.write_point_cloud("output2.ply", pointcloud)
    cv2.destroyAllWindows()
    vis.destroy_window()
"""