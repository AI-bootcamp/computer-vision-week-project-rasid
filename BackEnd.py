import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import time
import warnings
import os
import torch  

warnings.filterwarnings('ignore')

class SpeedDetector:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            # Load YOLOv8 model
            self.object_detector = YOLO('Model/object_Detection.pt')
            
            # Load classification model
            self.classifier = tf.keras.models.load_model('Model/classfaction_model.h5')
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")

        # Class names
        self.class_names = ['10', '100', '110', '120', '140', '20', '30', '40', '50', '60', '70', '80', '90']
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Speed tracking
        self.current_speed = None
        self.speed_history = []

    def detect_speed_sign(self, frame):
        """Detect speed sign in frame using YOLOv8"""
        try:
            # Run inference
            results = self.object_detector(frame)
            
            # Process results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if result.names[int(box.cls)] == 'speed_limit':  # Adjust class name
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        return (x1, y1, x2, y2)
            return None
        except Exception as e:
            print(f"Detection error: {e}")
            return None

    def classify_speed(self, cropped_image):
        """Classify speed from cropped image"""
        try:
            img = Image.fromarray(cropped_image)
            img = self.transform(img).unsqueeze(0)
            
            # Predict
            predictions = self.classifier.predict(img.numpy())
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions)
            
            return self.class_names[predicted_class], confidence
        except Exception as e:
            print(f"Classification error: {e}")
            return None, 0.0

    def process_frame(self, frame):
        """Complete frame processing pipeline"""
        try:
            # Step 1: Detect speed sign
            bbox = self.detect_speed_sign(frame)
            if bbox is None:
                return None, None, None
                
            # Step 2: Crop detected region
            x1, y1, x2, y2 = bbox
            cropped = frame[y1:y2, x1:x2]
            
            # Step 3: Classify speed
            speed, confidence = self.classify_speed(cropped)
            
            return speed, confidence, bbox
        except Exception as e:
            print(f"Frame processing error: {e}")
            return None, None, None

    def process_video(self, video_path, callback=None):
        """Process complete video file"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                if frame_count % int(fps) != 0:  # Process 1 frame per second
                    continue
                    
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                speed, confidence, bbox = self.process_frame(frame_rgb)
                
                if speed is not None:
                    if self.current_speed is None or abs(int(speed) - int(self.current_speed)) >= 5:
                        self.current_speed = speed
                        self.speed_history.append(int(speed))
                        if callback:
                            callback(speed, frame_count/fps, bbox)
                            
            cap.release()
            return self.speed_history
            
        except Exception as e:
            print(f"Video processing error: {e}")
            return []