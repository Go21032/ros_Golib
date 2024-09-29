#!/usr/bin/env python3
import pyrealsense2 as rs
import open3d as o3d
import numpy as np

# RealSenseのパイプラインを設定
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# ストリーム開始
pipeline.start(config)

try:
    # フレームを取得
    for _ in range(30):  # 最初の数フレームをスキップ
        frames = pipeline.wait_for_frames()
    
    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        print("深度フレームが取得できませんでした。")
        exit(1)

    # 深度フレームをnumpy配列に変換
    depth_image = np.asanyarray(depth_frame.get_data())

    # 点群を生成
    pc = rs.pointcloud()
    points = pc.calculate(depth_frame)
    vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    
    # Open3DのPointCloudオブジェクトを作成
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vtx)

    # RANSACを使って平面を検出
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

finally:
    # ストリームを停止
    pipeline.stop()

# import numpy as np
# import matplotlib.pyplot as plt
# import open3d as o3d

# # 点群の読み込み
# dataset = o3d.data.PCDPointCloud()
# pcd = o3d.io.read_point_cloud(dataset.path)
# # 引数を設定して、平面を推定
# plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
# # 平面モデルの係数を出力
# [a, b, c, d] = plane_model
# print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# # インライアの点を抽出して色を付ける
# inlier_cloud = pcd.select_by_index(inliers)
# inlier_cloud.paint_uniform_color([1.0, 0, 0])

# # 平面以外の点を抽出
# outlier_cloud = pcd.select_by_index(inliers, invert=True)

# # 可視化はdraw_geometriesではなくdraw_plotly
# o3d.visualization.draw_plotly([inlier_cloud, outlier_cloud],
#                                   zoom=0.8,
#                                   front=[-0.4999, -0.1659, -0.8499],
#                                   lookat=[2.1813, 2.0619, 2.0999],
#                                   up=[0.1204, -0.9852, 0.1215])