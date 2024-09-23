from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import csv
import os

model = YOLO("/home/go/slope_ws/src/ros_Golib/slope_edge_detection/scripts/best.pt")

img = Image.open("/home/go/slope_ws/src/ros_Golib/slope_edge_detection/scripts/2dd/kennsyou/slope_39.jpg")
results = model.predict(source=img, save=True)

masks = results[0].masks
x_numpy = masks[0].data.to('cpu').detach().numpy().copy()
print(x_numpy.shape)

name = results[0].names
print(name)   
point = masks[0].xy
point = np.array(point)

# ３次元を２次元に変換する
result = []
for i in range(len(point)):
    for j in range(len(point[i])):
        my_list = []
        for k in range(len(point[i][j])):
            my_list.append(point[i][j][k])
        result.append(my_list)
result = np.array(result)
point = result

# PIL Imageをnumpy配列に変換
img = np.array(img)

# OpenCVで使用するためにRGBからBGRに変換
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#y座標が高い順にソート
'''
sorted() は、リストを特定の順序でソートするためのPythonの組み込み関数。
point はソートしたいリストです。このリストは、各要素が座標（[x, y] の形）を表している。
key=lambda x: x[1] は、ソートの基準となるキーを指定します。ここでは、各要素（座標）のy座標（インデックス1の値）を基準にソートしている。
x は point リストの各要素を指します。各要素は座標であり、今回は[u, v] という形。
x: x[1] は、リストの各要素 x のインデックス1の値（つまり y 座標）を返す。
'''
point = sorted(point, key=lambda x: x[1], reverse=True)

# 上位50個のy座標が高い座標を取得
top_points = point[:50]

# 座標の表示
for i in range(len(top_points)):
    (u, v) = (int(top_points[i][0]), int(top_points[i][1]))
    print((u, v))
    # 画像内に指定したクラス(results[0]の境界線を赤点で描画
    cv2.circle(img, (u, v), 10, (0, 0, 255), -1)

# 上位50個の座標の中央値を算出
median_x = int(np.median([p[0] for p in top_points]))
median_y = int(np.median([p[1] for p in top_points]))

# 中央値の座標を表示
print("Median coordinate:", (median_x, median_y))

# 中央値の座標を青点で描画
cv2.circle(img, (median_x, median_y), 10, (255, 0, 0), -1)

# 画像を表示＆保存
cv2.imwrite('top_mask10.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# CSVファイルに座標を出力
csv_file_path = 'coordinates.csv'

# ファイル書き込みをデバッグするためのメッセージ
print(f"Attempting to write to {csv_file_path}")

try:
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['X', 'Y'])  # ヘッダーを書き込む
        for point in top_points:
            writer.writerow(point)
    print(f"Coordinates have been written to {csv_file_path}")
except Exception as e:
    print(f"An error occurred while writing to the file: {e}")
