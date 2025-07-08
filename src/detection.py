from ultralytics import YOLO
import cv2
import numpy as np
import torch
from typing import List, Dict, Union
from dataclasses import dataclass

@dataclass
class DetectionBox:
    """Represents a detection bounding box"""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str
    
    @property
    def center(self) -> tuple:
        """Get center point of bounding box"""
        return (int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2))
    
    @property
    def width(self) -> int:
        """Get width of bounding box"""
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        """Get height of bounding box"""
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        """Get area of bounding box"""
        return self.width * self.height

class PlayerDetector:
    def __init__(self, model_path: str, img_size: tuple = (256, 448)):
        """
        Initialize PlayerDetector
        
        Args:
            model_path: Path to YOLO model file
            img_size: Input image size (height, width)
        """
        self.model_path = model_path
        self.img_size = img_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the YOLO model with optimizations"""
        try:
            self.model = YOLO(self.model_path)
            
            # Apply optimizations for CUDA
            if self.device.type == 'cuda':
                # Move model to GPU and enable half precision
                self.model.model = self.model.model.half().to(self.device)
                print(f"✅ Model loaded with GPU optimization (FP16)")
            else:
                print(f"✅ Model loaded on CPU")
            
            print(f"🏷️  Model classes: {self.model.model.names}")
            print(f"📏 Input size: {self.img_size}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Resize and normalize image for model input
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Preprocessed tensor ready for model
        """
        # Resize image to model input size
        img = cv2.resize(image, (self.img_size[1], self.img_size[0]))
        
        # Convert BGR to RGB and transpose to CHW format
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        
        # Convert to tensor
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.device.type == 'cuda' else img.float()
        img /= 255.0  # Normalize to [0, 1]
        
        return img.unsqueeze(0)  # Add batch dimension
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5, 
               return_format: str = 'dict') -> Union[List[Dict], List[DetectionBox]]:
        """
        Detect players in a frame with optimizations
        
        Args:
            frame: Input frame in BGR format
            conf_threshold: Confidence threshold for detections
            return_format: 'dict' or 'dataclass' for output format
            
        Returns:
            List of detections in specified format
        """
        if frame is None or frame.size == 0:
            return []
        
        # Handle different color depths
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        
        try:
            # Preprocess image
            img = self.preprocess(frame)
            
            # Run detection
            with torch.no_grad():
                results = self.model(img, conf=conf_threshold, verbose=False)
            
            # Extract detections
            detections = []
            original_height, original_width = frame.shape[:2]
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.cpu().numpy()
                    
                    for box in boxes:
                        # Get coordinates and scale back to original image size
                        x1, y1, x2, y2 = box.xyxy[0]
                        
                        # Scale coordinates back to original image size
                        x1 = int(x1 * original_width / self.img_size[1])
                        y1 = int(y1 * original_height / self.img_size[0])
                        x2 = int(x2 * original_width / self.img_size[1])
                        y2 = int(y2 * original_height / self.img_size[0])
                        
                        # Ensure coordinates are within image bounds
                        x1 = max(0, min(x1, original_width - 1))
                        y1 = max(0, min(y1, original_height - 1))
                        x2 = max(x1 + 1, min(x2, original_width))
                        y2 = max(y1 + 1, min(y2, original_height))
                        
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.model.model.names[class_id]
                        
                        if return_format == 'dataclass':
                            detections.append(DetectionBox(
                                x1=x1, y1=y1, x2=x2, y2=y2,
                                confidence=confidence,
                                class_id=class_id,
                                class_name=class_name
                            ))
                        else:
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': class_name
                            })
            
            return detections
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return []
    
    def detect_players(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[DetectionBox]:
        """
        Detect only players and return as DetectionBox objects
        
        Args:
            frame: Input frame
            conf_threshold: Confidence threshold
            
        Returns:
            List of DetectionBox objects for player detections
        """
        all_detections = self.detect(frame, conf_threshold, return_format='dataclass')
        
        # Filter for players only
        player_detections = [det for det in all_detections if det.class_name == 'player']
        
        return player_detections
    
    def visualize_detections(self, frame: np.ndarray, detections: List[Union[Dict, DetectionBox]], 
                           show_confidence: bool = True, show_class: bool = True) -> np.ndarray:
        """
        Visualize detections on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            show_confidence: Whether to show confidence scores
            show_class: Whether to show class names
            
        Returns:
            Frame with visualized detections
        """
        output_frame = frame.copy()
        
        for detection in detections:
            if isinstance(detection, DetectionBox):
                x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2
                confidence = detection.confidence
                class_name = detection.class_name
            else:
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                class_name = detection['class_name']
            
            # Choose color based on class
            color = (0, 255, 0) if class_name == 'player' else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label_parts = []
            if show_class:
                label_parts.append(class_name)
            if show_confidence:
                label_parts.append(f"{confidence:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                
                # Get text size
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Draw label background
                cv2.rectangle(output_frame, 
                             (x1, y1 - text_height - 10),
                             (x1 + text_width, y1),
                             color, -1)
                
                # Draw label text
                cv2.putText(output_frame, label, (x1, y1 - 5),
                           font, font_scale, (255, 255, 255), thickness)
        
        return output_frame
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_path': self.model_path,
            'device': str(self.device),
            'input_size': self.img_size,
            'classes': self.model.model.names if self.model else {},
            'num_classes': len(self.model.model.names) if self.model else 0
        }

if __name__ == "__main__":
    print("🔍 Enhanced detection module ready!")
    
    # Test with dummy data if run directly
    try:
        # This will fail without actual model, but shows the interface
        detector = PlayerDetector("dummy_model.pt")
        print("Detector initialized successfully!")
    except Exception as e:
        print(f"Expected error in standalone mode: {e}")