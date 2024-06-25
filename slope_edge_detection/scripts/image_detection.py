import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

class Slope:
    # Realsenseパイプラインを設定
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # パイプラインを開始
    pipeline.start(config)

    # YOLOモデルを読み込む
    model = YOLO('best.pt')

    try:
        while True:
            # フレームを待ち、取得
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            # フレームデータをnumpy配列に変換
            color_image = np.asanyarray(color_frame.get_data())
            
            # YOLOモデルでオブジェクト検出
            result = model.predict(color_image)
            img_annotated = result[0].plot()
            
            # ウィンドウに画像表示
            cv2.imshow('Realsense YOLO', img_annotated)

            # qで終了
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        # パイプラインを停止
        pipeline.stop()
        cv2.destroyAllWindows()

# #写真でテスト   
# from ultralytics import YOLO
# import cv2

# model = YOLO("/home/go/slope_ws/src/ros_Golib/slope_edge_detection/scripts/best.pt")

# im2 = cv2.imread("/home/go/slope_ws/src/ros_Golib/slope_edge_detection/scripts/IMG_9448.jpg")
# results = model.predict(source=im2, save=True)