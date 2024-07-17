import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image

# カメラの内部パラメータ（例として）
fx = 525.0  # 焦点距離 x
fy = 525.0  # 焦点距離 y
cx = 319.5  # 光学中心 x
cy = 239.5  # 光学中心 y

# 深度マップを読み込む（ここではサンプルとしてランダムな深度マップを使用）
# 実際には深度カメラから取得したデータを使用
depth_map = np.random.rand(640, 640) * 1000  # 単位はミリメートル

# 2次元画像座標から3次元空間座標への変換関数
def pixel_to_camera_coordinates(u, v, depth, fx, fy, cx, cy):
    z = depth * 1e-3
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z])

# YOLOモデルの読み込み
model = YOLO("C:/Users/gakuh/OneDrive/ドキュメント/KGI/研究/YOLOv8/mask_3d/pl2_best.pt")

# 画像の読み込み
img = Image.open(r"C:\Users\gakuh\OneDrive\ドキュメント\KGI\研究\YOLOv8\mask_3d\1.png")
results = model.predict(source=img, save=True)

# セマンティックセグメンテーションマスクの取得
masks = results[0].masks
points = masks[0].xy
points = np.array(points)

# 3次元座標を格納するリスト
points_3d = []
points_uv = []

# 各点について3次元座標に変換
for i in range(len(points)):
    for j in range(len(points[i])):
        u, v = int(points[i][j][0]), int(points[i][j][1])
        depth = depth_map[v, u]  # 深度マップから深度を取得
        point_3d = pixel_to_camera_coordinates(u, v, depth, fx, fy, cx, cy)
        points_3d.append(point_3d)
        points_uv.append((u, v))

points_3d = np.array(points_3d)
points_uv = np.array(points_uv)

# カメラから各点までの距離を計算
distances = np.linalg.norm(points_3d, axis=1)

# カメラからの距離を表示
for idx, distance in enumerate(distances):
    print(f"Point {points_uv[idx]} -> Distance: {distance} mm")

# カメラとの距離が近い順にソート
sorted_indices = np.argsort(distances)

# 距離が近い座標10個を取得
nearest_points_3d = points_3d[sorted_indices[:10]]
nearest_points_uv = points_uv[sorted_indices[:10]]

print("Nearest 3D Points: ", nearest_points_3d)
print("Nearest UV Points: ", nearest_points_uv)

# 画像に近い座標を描画
img_np = np.array(img)
img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

for (u, v) in nearest_points_uv:
    cv2.circle(img_bgr, (u, v), 5, (0, 255, 0), -1)

# 画像を表示＆保存
cv2.imwrite('nearest_points.jpg', img_bgr)
cv2.imshow('Nearest Points', img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
