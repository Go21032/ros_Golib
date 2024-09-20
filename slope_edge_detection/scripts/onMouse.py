import cv2

def onMouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)

img = cv2.imread('/home/go/slope_ws/src/ros_Golib/slope_edge_detection/scripts/2dd/top_mask2.jpg')
resized_img = cv2.resize(img, (640, 640))  # 画像を1280x720にリサイズ
cv2.imshow('sample', resized_img)
cv2.setMouseCallback('sample', onMouse)
cv2.waitKey(0)
cv2.destroyAllWindows()  # ウィンドウを閉じる