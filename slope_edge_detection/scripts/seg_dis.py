#参考動画https://www.youtube.com/watch?v=l8nv1giyfsM&t=1s
from ultralytics import YOLO
from PIL import Image
import cv2
import csv
csv_filename = 'sample.csv'

#出力するデータ(ここを３次元の座標に変換すれば距離求められる)
header = ['Class', 'Label', 'Scores', 'x1', 'y1', 'x2', 'y2']

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
    
    csv_data = [str(cls),str(label),str(score),str(x1),str(y1),str(x2),str(y2)]
    writer.writerow(csv_data)
    
items.show()

file.close()
    