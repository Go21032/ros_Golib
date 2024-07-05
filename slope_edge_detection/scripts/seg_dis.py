#参考動画https://www.youtube.com/watch?v=l8nv1giyfsM&t=1s
from ultralytics import YOLO
from PIL import Image
import cv2
import csv
import math
csv_filename = 'sample2.csv'

#出力するデータ(ここを３次元の座標に変換すれば距離求められる)
header = ['Class', 'Label', 'Scores', 'x1', 'y1', 'x2', 'y2', 'distance_pixels']

#CSVファイルにデータを書き込む
file = open(csv_filename, mode='w', newline='', encoding='utf-8')
writer = csv.writer(file)
writer.writerow(header)

model = YOLO("/home/go/slope_ws/src/ros_Golib/slope_edge_detection/scripts/best.pt")

im1 = Image.open("/home/go/slope_ws/src/ros_Golib/slope_edge_detection/scripts/IMG_9448.jpg")
results = model.predict(source=im1, save=True)

items = results[0]

for item in items:
    cls = int(item.boxes.cls)
    label = item.names[int(cls)]
    score = item.boxes.conf.cpu().numpy()[0]
    x1,y1,x2,y2 = item.boxes.xyxy.cpu().numpy()[0]
        
    # 距離を計算 (ピクセル単位)
    distance_pixels = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    csv_data = [str(cls),str(label),str(score),str(x1),str(y1),str(x2),str(y2),str(distance_pixels)]
    writer.writerow(csv_data)
    
# CSVファイルを読み取るためにopenする
with open('sample2.csv', 'r') as file:
    reader = csv.DictReader(file)
    
    #距離算出
    # 各行をループで処理
    for row in reader:
        # 'slope'クラスのデータだけ処理
        if row['Label'] == 'slope':
            # 座標を取得
            x1 = float(row['x1'])
            y1 = float(row['y1'])
            x2 = float(row['x2']) 
            y2 = float(row['y2'])
            
            # 距離を計算 (ピクセル単位)
            distance_pixels = float(row['distance_pixels'])

file.close()
items.show()