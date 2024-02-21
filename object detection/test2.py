from ultralytics import YOLO
import cv2
import numpy as np
model = YOLO("yolov8n.pt")
results = model(source="https://ultralytics.com/images/bus.jpg")
boxes = results[0].boxes.xyxy.tolist()
classes = results[0].boxes.cls.tolist()
names = results[0].names
confidences = results[0].boxes.conf.tolist()
image = results[0].orig_img
for box, label in zip(boxes, classes):
    if label == 0:
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255,0), 2)
cv2.imshow('Image', image)
cv2.imwrite('test.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
