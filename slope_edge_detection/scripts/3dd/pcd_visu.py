import open3d as o3d

def visualize_ply(file_path):
    # PLYファイルを読み込む
    file_path = "/home/carsim05/slope_ws/src/ros_Golib/slope_edge_detection/scripts/3dd/ransac3seco.ply"
    pcd = o3d.io.read_point_cloud(file_path)

    # ビジュアライザを作成
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    # 表示ループ
    vis.run()
    vis.destroy_window()

# PLYファイルのパスを指定
visualize_ply("output.ply")