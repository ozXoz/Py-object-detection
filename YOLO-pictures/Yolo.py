from ultralytics import YOLO

import cv2 # for using that iw Detection

# model =YOLO-pictures('https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt')
model =YOLO('https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt')
result =model('/Users/oz/PycharmProjects/Object-Detection-Of-Yolo/YOLO-pictures/images/3.png',show=True)
cv2.waitKey(0)