from ultralytics import  YOLO
import cv2
import cvzone
import math

# Creating Webcam Object

cap =cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush","watch","headphone","airpods case"
              ]


model =YOLO('https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt')

while True:
    success, img = cap.read()
    results=model(img,stream=True)
    #I was just assuming that we implementaion for convert stream to integer
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1, y1, x2, y2 =box.xyxy[0]
            # print((x1,x2,y1,y2)) # show tensor , # below  code i will do convert tensor to int
            x1,y1,x2,y2 =int(x1),int(y1),int(x2),int(y2)
            print(x1,y1,x2,y2)
            #either use rectangle or cvz.cornerRect
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h =x2-x1,y2-y1
            cvzone.cornerRect(img, [x1, y1, w, h])

            # Confidence Value and class name
            conf =math.ceil((box.conf[0]*100))/100
            # print(conf)
            #Class name
            cls =int(box.cls[0])
            #we shows confidence like accurracy

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
    cv2.imshow("Image",img)
    cv2.waitKey(1)