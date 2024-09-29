import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import datetime

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
align = rs.align(rs.stream.color)

try:
    while True:
        dt0 = datetime.datetime.now()
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        profile = frames.get_profile()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        o3d_color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        img_depth = o3d.geometry.Image(depth_image)
        img_color = o3d.geometry.Image(o3d_color_image)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth, convert_rgb_to_intensity=False)

        intrinsics = profile.as_video_stream_profile().get_intrinsics()
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # 引数を設定して、平面を推定
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        # 平面モデルの係数を出力
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        # インライアの点を抽出して色を付ける
        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])

        # 平面以外の点を抽出
        outlier_cloud = pcd.select_by_index(inliers, invert=True)

        # 可視化
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                          zoom=0.8,
                                          front=[-0.4999, -0.1659, -0.8499],
                                          lookat=[2.1813, 2.0619, 2.0999],
                                          up=[0.1204, -0.9852, 0.1215])

        process_time = datetime.datetime.now() - dt0
        print("FPS: " + str(1 / process_time.total_seconds()))
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
