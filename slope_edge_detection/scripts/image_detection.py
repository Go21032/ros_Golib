# import cv2

# #カメラのキャプチャ開始
# cap = cv2.VideoCapture(0)

# # カメラがオープンできたかチェック
# if not cap.isOpened():
#     print("カメラが開かない")
#     exit()
    
# while True:
#     # フレームごとに画像キャプチャ
#     ret, frame = cap.read()
    
#     # 正常にフレームが読み込まれたかチェック
#     if not ret:
#         print("フレームを取得できない")
#         break
    
#     # ウィンドウに画像表示
#     cv2.imshow('Camera', frame)
    
#     # qで終了
#     if cv2.waitKey(1) == ord('q'):
#         break
    
# # キャプチャをリリースし、ウィンドウズを閉じる
# cap.release()
# cv2.destroyAllWindows()
 
# #写真でテスト   
# from ultralytics import YOLO
# import cv2

# model = YOLO("/home/go/slope_ws/src/ros_Golib/slope_edge_detection/scripts/last.pt")

# im2 = cv2.imread("/home/go/slope_ws/src/ros_Golib/slope_edge_detection/scripts/IMG_9447.jpg")
# results = model.predict(source=im2, save=True)