# BackEnd.py

from ultralytics import YOLO
import cv2

class SpeedDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = r"C:\Users\abdul\OneDrive\Documents\GitHub\computer-vision-week-project-rasid\Model\object_Detection.pt"
        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def process_frame(self, frame):
        try:
            results = self.model(frame, imgsz=640, conf=0.4, verbose=False)[0]
            if results.boxes is not None and len(results.boxes) > 0:
                best_box = results.boxes[0]
                cls_id = int(best_box.cls.item())
                label = self.class_names[cls_id]
                conf = float(best_box.conf.item())
                x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
                return label, conf, (x1, y1, x2, y2)
            return None, None, None
        except Exception as e:
            print(f"Error in process_frame: {e}")
            return None, None, None
