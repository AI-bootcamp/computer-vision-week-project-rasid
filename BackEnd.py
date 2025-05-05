# BackEnd.py

import torch
from ultralytics import YOLO
import numpy as np

class SpeedDetector:
    def __init__(self, model_path=None):
        # Use default model path if not provided
        if model_path is None:
            model_path = r"Model\object_Detection.pt"
        
        self.model = YOLO(model_path)
        self.model.fuse()  # Optional: optimize for inference

        # Extract class names from model
        self.class_names = self.model.names  # Dict like {0: 'speed_30', ...}


    def process_frame(self, frame):
        """
        Takes a frame (numpy RGB) and returns (speed, confidence, bbox)
        """
        
        try:
            # Run inference
            results = self.model(frame, imgsz=640, conf=0.4, verbose=False)[0]

            # If detections exist
            if results.boxes is not None and len(results.boxes) > 0:
                best_box = results.boxes[0]  # Pick top prediction
                cls_id = int(best_box.cls.item())
                conf = float(best_box.conf.item())
                label = self.class_names[cls_id]
                
                # Extract bbox
                x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
                
                # Extract speed from label (e.g., "speed_100" -> "100")
                speed = ''.join(filter(str.isdigit, label))
                
                return speed, conf, (x1, y1, x2, y2)
            
            return None, None, None
        except Exception as e:
            print(f"Detection error: {str(e)}")
            return None, None, None
