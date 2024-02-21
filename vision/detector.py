from ultralytics import YOLO
import cv2


class Detector():
    def __init__(self, model_path='yolov8n.pt', target='cup'):
        self.model = YOLO(model_path)
        self.target = target

    def detect(self, image):  # returns list of tuples (x,y) - x,y coords of bboxes center pixels
        results = self.model(image)
        boxes = results[0].boxes.xyxy.tolist()
        classes = results[0].boxes.cls.tolist()
        names = results[0].names
        target_id = list(names.values()).index(self.target)
        confidences = results[0].boxes.conf.tolist()
        centers = []
        for box, label in zip(boxes, classes):
            if label == target_id:
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                centers.append((int(center_x), int(center_y)))
        return centers

    def show_detections(self, image, save=False,
                        filename='test'):  # shows original image with target objects detected and their center pixels
        results = self.model(image)
        boxes = results[0].boxes.xyxy.tolist()
        classes = results[0].boxes.cls.tolist()
        names = results[0].names
        target_id = list(names.values()).index(self.target)
        for box, label in zip(boxes, classes):
            if label == target_id:
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        if save:
            cv2.imwrite(filename, image)
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
