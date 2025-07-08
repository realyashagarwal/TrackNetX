from ultralytics import YOLO
import cv2
import numpy as np

class PlayerDetector:
    def __init__(self, model_path):
        """Initialize the player detector with YOLOv11 model"""
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ Model loaded successfully from {self.model_path}")
            print(f"🏷️  Model classes: {self.model.names}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def detect(self, frame, conf_threshold=0.5):
        """Detect players in a frame"""
        if self.model is None:
            raise ValueError("Model not loaded!")
        
        # Run detection
        results = self.model(frame, conf=conf_threshold)
        
        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name
                    }
                    detections.append(detection)
        
        return detections
    
    def visualize_detections(self, frame, detections):
        """Draw bounding boxes on frame"""
        vis_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(vis_frame, label, (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return vis_frame

if __name__ == "__main__":
    print("🔍 Detection module ready!")