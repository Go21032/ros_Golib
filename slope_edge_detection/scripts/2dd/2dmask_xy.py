from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

model = YOLO("/home/go/slope_ws/src/ros_Golib/slope_edge_detection/scripts/best.pt")

img = Image.open("/home/go/slope_ws/src/ros_Golib/slope_edge_detection/scripts/IMG_9448.jpg")
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

#座標の表示
for i in range(len(point)):
    (u, v) = (int(point[i][0]), int(point[i][1]))
    print((u, v))
    #画像内に指定したクラス(results[0]の境界線を赤点で描画
    cv2.circle(img, (u, v), 10, (0, 0, 255), -1)

# 画像を表示＆保存
cv2.imwrite('seg_mask2.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()