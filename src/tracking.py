"""
TrackNetX - Enhanced Player Tracking Module
Advanced tracking with Hungarian Algorithm and improved cost function
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math
from scipy.optimize import linear_sum_assignment

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
    def center(self) -> Tuple[int, int]:
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

@dataclass
class TrackedPlayer:
    """Represents a tracked player with history"""
    track_id: int
    class_name: str
    current_box: DetectionBox
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    age: int = 0
    hits: int = 0
    hit_streak: int = 0
    time_since_update: int = 0
    color: Tuple[int, int, int] = field(default_factory=lambda: tuple(np.random.randint(0, 255, 3).tolist()))
    
    # Motion model parameters
    velocity: Tuple[float, float] = (0.0, 0.0)
    acceleration: Tuple[float, float] = (0.0, 0.0)
    
    # Additional fields for better re-identification
    last_seen_frame: Optional[np.ndarray] = None
    appearance_features: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Initialize after creation"""
        self.history.append(self.current_box.center)
        # Initialize hits to 1 since we start with one detection
        self.hits = 1
        self.hit_streak = 1
    
    def update(self, detection: DetectionBox):
        """Update tracked player with new detection"""
        old_center = self.current_box.center
        new_center = detection.center
        
        # Update velocity and acceleration
        if len(self.history) >= 1:  # Changed from >= 2 to >= 1
            prev_velocity = self.velocity
            self.velocity = (
                (new_center[0] - old_center[0]),
                (new_center[1] - old_center[1])
            )
            self.acceleration = (
                self.velocity[0] - prev_velocity[0],
                self.velocity[1] - prev_velocity[1]
            )
        
        self.current_box = detection
        self.history.append(detection.center)
        self.age += 1
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
    
    def predict_next_position(self) -> Tuple[int, int]:
        """Predict next position using Kalman-like motion model"""
        if len(self.history) < 2:
            return self.current_box.center
        
        # Use velocity and acceleration for prediction
        current_pos = self.current_box.center
        
        # Predict with velocity and damped acceleration
        predicted_x = current_pos[0] + self.velocity[0] + 0.5 * self.acceleration[0]
        predicted_y = current_pos[1] + self.velocity[1] + 0.5 * self.acceleration[1]
        
        return (int(predicted_x), int(predicted_y))
    
    def get_predicted_box(self) -> DetectionBox:
        """Get predicted bounding box for next frame"""
        predicted_center = self.predict_next_position()
        current_box = self.current_box
        
        # Maintain current box dimensions
        half_width = current_box.width // 2
        half_height = current_box.height // 2
        
        return DetectionBox(
            x1=predicted_center[0] - half_width,
            y1=predicted_center[1] - half_height,
            x2=predicted_center[0] + half_width,
            y2=predicted_center[1] + half_height,
            confidence=current_box.confidence,
            class_id=current_box.class_id,
            class_name=current_box.class_name
        )

class PlayerTracker:
    """Enhanced tracking class with Hungarian Algorithm"""
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 100, 
                 iou_threshold: float = 0.1, confidence_threshold: float = 0.5):
        """
        Initialize tracker
        
        Args:
            max_disappeared: Maximum frames a player can be missing before deletion
            max_distance: Maximum distance for association (pixels)
            iou_threshold: Minimum IoU for valid association
            confidence_threshold: Minimum confidence for detections
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.next_id = 1
        self.tracked_players: Dict[int, TrackedPlayer] = {}
        
        # Re-identification features
        self.lost_players: Dict[int, TrackedPlayer] = {}
        self.reidentification_threshold = 0.6  # Lowered for better re-identification
        
        # Cost function weights (adjusted for better stability)
        self.weight_distance = 0.3
        self.weight_iou = 0.5  # Increased IoU weight for better matching
        self.weight_confidence = 0.1
        self.weight_size = 0.1
        
        # Jitter tolerance
        self.jitter_tolerance = 20  # pixels
        
        # Stable matching parameters
        self.min_hit_streak_for_stable = 3  # Minimum hits before considering stable
        self.stable_track_bonus = 0.8  # Cost reduction for stable tracks
        
    def calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def calculate_iou(self, box1: DetectionBox, box2: DetectionBox) -> float:
        """Calculate Intersection over Union (IoU) between two boxes"""
        # Calculate intersection area
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        area1 = box1.area
        area2 = box2.area
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_size_similarity(self, box1: DetectionBox, box2: DetectionBox) -> float:
        """Calculate size similarity between two boxes"""
        area1 = box1.area
        area2 = box2.area
        
        if area1 == 0 or area2 == 0:
            return 0.0
        
        # Size ratio similarity (closer to 1 is better)
        ratio = min(area1, area2) / max(area1, area2)
        return ratio
    
    def calculate_comprehensive_cost(self, tracked_player: TrackedPlayer, 
                                   detection: DetectionBox) -> float:
        """
        Calculate comprehensive cost combining multiple factors
        Enhanced for ID consistency and jitter robustness
        
        Args:
            tracked_player: Existing tracked player
            detection: New detection
            
        Returns:
            Cost value (lower is better, inf means invalid)
        """
        # Skip if different class
        if detection.class_name != tracked_player.class_name:
            return float('inf')
        
        # Skip if confidence too low
        if detection.confidence < self.confidence_threshold:
            return float('inf')
        
        # Get predicted position and box
        predicted_pos = tracked_player.predict_next_position()
        predicted_box = tracked_player.get_predicted_box()
        
        # Calculate distance cost (normalized)
        distance = self.calculate_distance(predicted_pos, detection.center)
        
        # Enhanced distance threshold with jitter tolerance
        effective_max_distance = self.max_distance + self.jitter_tolerance
        if distance > effective_max_distance:
            return float('inf')
        
        distance_cost = distance / effective_max_distance
        
        # Calculate IoU cost (1 - IoU for cost minimization)
        iou_current = self.calculate_iou(tracked_player.current_box, detection)
        iou_predicted = self.calculate_iou(predicted_box, detection)
        iou = max(iou_current, iou_predicted)
        
        # More lenient IoU threshold for jitter tolerance
        effective_iou_threshold = max(0.05, self.iou_threshold - 0.05) if distance < self.jitter_tolerance else self.iou_threshold
        
        if iou < effective_iou_threshold:
            return float('inf')
        
        iou_cost = 1.0 - iou
        
        # Calculate confidence cost (1 - confidence for cost minimization)
        confidence_cost = 1.0 - detection.confidence
        
        # Calculate size similarity cost
        size_similarity = self.calculate_size_similarity(tracked_player.current_box, detection)
        size_cost = 1.0 - size_similarity
        
        # Combine costs with weights
        total_cost = (
            self.weight_distance * distance_cost +
            self.weight_iou * iou_cost +
            self.weight_confidence * confidence_cost +
            self.weight_size * size_cost
        )
        
        # Apply stability bonus for tracks with good hit streak
        if tracked_player.hit_streak >= self.min_hit_streak_for_stable:
            total_cost *= self.stable_track_bonus
        
        # Apply jitter tolerance bonus for very close detections
        if distance < self.jitter_tolerance:
            total_cost *= 0.5  # Significantly prefer close matches
        
        return total_cost
    
    def associate_detections_hungarian(self, detections: List[DetectionBox], 
                                     frame: np.ndarray) -> Dict[int, int]:
        """
        Associate detections with existing tracks using Hungarian Algorithm
        
        Args:
            detections: List of detection boxes
            frame: Current frame for appearance features
            
        Returns:
            Dict mapping track_id to detection_index
        """
        if not self.tracked_players or not detections:
            return {}
        
        track_ids = list(self.tracked_players.keys())
        
        # Create cost matrix
        cost_matrix = np.full((len(track_ids), len(detections)), float('inf'))
        
        # Fill cost matrix
        for i, track_id in enumerate(track_ids):
            tracked_player = self.tracked_players[track_id]
            
            for j, detection in enumerate(detections):
                cost = self.calculate_comprehensive_cost(tracked_player, detection)
                cost_matrix[i, j] = cost
        
        # Apply Hungarian algorithm
        # Replace inf with a large number for the algorithm
        cost_matrix_finite = np.where(cost_matrix == float('inf'), 1e6, cost_matrix)
        
        try:
            row_indices, col_indices = linear_sum_assignment(cost_matrix_finite)
            
            # Create associations dictionary, excluding invalid matches
            associations = {}
            for i, j in zip(row_indices, col_indices):
                if cost_matrix[i, j] != float('inf'):
                    track_id = track_ids[i]
                    associations[track_id] = j
            
            return associations
            
        except Exception as e:
            print(f"Hungarian algorithm failed: {e}")
            return {}
    
    def extract_appearance_features(self, frame: np.ndarray, box: DetectionBox) -> np.ndarray:
        """
        Extract appearance features for re-identification
        Enhanced color histogram with spatial information
        """
        # Extract region of interest
        roi = frame[max(0, box.y1):min(frame.shape[0], box.y2), 
                   max(0, box.x1):min(frame.shape[1], box.x2)]
        
        if roi.size == 0:
            return np.zeros(96)  # Return zero features if invalid ROI
        
        # Resize ROI to standard size for consistent features
        roi_resized = cv2.resize(roi, (64, 64))
        
        # Calculate color histogram for each channel
        hist_b = cv2.calcHist([roi_resized], [0], None, [16], [0, 256])
        hist_g = cv2.calcHist([roi_resized], [1], None, [16], [0, 256])
        hist_r = cv2.calcHist([roi_resized], [2], None, [16], [0, 256])
        
        # Calculate spatial histograms (top and bottom half)
        h_mid = roi_resized.shape[0] // 2
        roi_top = roi_resized[:h_mid, :]
        roi_bottom = roi_resized[h_mid:, :]
        
        hist_top = cv2.calcHist([roi_top], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist_bottom = cv2.calcHist([roi_bottom], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        # Normalize all histograms
        hist_b = cv2.normalize(hist_b, hist_b).flatten()
        hist_g = cv2.normalize(hist_g, hist_g).flatten()
        hist_r = cv2.normalize(hist_r, hist_r).flatten()
        hist_top = cv2.normalize(hist_top, hist_top).flatten()[:16]  # Take first 16 values
        hist_bottom = cv2.normalize(hist_bottom, hist_bottom).flatten()[:16]  # Take first 16 values
        
        # Concatenate features
        features = np.concatenate([hist_b, hist_g, hist_r, hist_top, hist_bottom])
        
        return features
    
    def compare_appearance(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compare two appearance feature vectors using multiple metrics"""
        if features1.size == 0 or features2.size == 0:
            return 0.0
        
        # Use correlation coefficient
        correlation = np.corrcoef(features1, features2)[0, 1]
        correlation = correlation if not np.isnan(correlation) else 0.0
        
        # Use cosine similarity
        dot_product = np.dot(features1, features2)
        norm_product = np.linalg.norm(features1) * np.linalg.norm(features2)
        cosine_sim = dot_product / norm_product if norm_product > 0 else 0.0
        
        # Use chi-squared distance (for histograms)
        chi_squared = 0.0
        for i in range(len(features1)):
            if features1[i] + features2[i] > 0:
                chi_squared += ((features1[i] - features2[i]) ** 2) / (features1[i] + features2[i])
        chi_squared_sim = 1.0 / (1.0 + chi_squared)
        
        # Combine metrics with weights
        similarity = 0.4 * correlation + 0.4 * cosine_sim + 0.2 * chi_squared_sim
        return max(0.0, similarity)
    
    def update(self, detections: List[DetectionBox], frame: np.ndarray) -> List[TrackedPlayer]:
        """
        Update tracker with new detections using Hungarian Algorithm
        
        Args:
            detections: List of detection boxes
            frame: Current frame for appearance features
            
        Returns:
            List of currently tracked players
        """
        # Filter detections by confidence
        valid_detections = [d for d in detections if d.confidence >= self.confidence_threshold]
        
        # Associate detections with existing tracks using Hungarian algorithm
        associations = self.associate_detections_hungarian(valid_detections, frame)
        
        # Update associated tracks
        updated_tracks = set()
        for track_id, detection_idx in associations.items():
            self.tracked_players[track_id].update(valid_detections[detection_idx])
            updated_tracks.add(track_id)
        
        # Handle unassociated detections (create new tracks or re-identify)
        associated_detections = set(associations.values())
        for i, detection in enumerate(valid_detections):
            if i not in associated_detections:
                # Enhanced re-identification with lost players
                reidentified = False
                best_similarity = 0.0
                best_lost_id = None
                best_cost = float('inf')
                
                for lost_id, lost_player in self.lost_players.items():
                    if (detection.class_name == lost_player.class_name and
                        lost_player.time_since_update <= self.max_disappeared):
                        
                        # Calculate comprehensive cost for re-identification
                        cost = self.calculate_comprehensive_cost(lost_player, detection)
                        
                        # More lenient constraints for re-identification
                        if cost != float('inf'):
                            # Calculate appearance similarity
                            new_features = self.extract_appearance_features(frame, detection)
                            old_features = self.extract_appearance_features(frame, lost_player.current_box)
                            appearance_similarity = self.compare_appearance(new_features, old_features)
                            
                            # Combined score (lower cost + higher appearance similarity)
                            combined_score = cost * (1.0 - appearance_similarity)
                            
                            if (appearance_similarity > self.reidentification_threshold and
                                combined_score < best_cost):
                                best_cost = combined_score
                                best_similarity = appearance_similarity
                                best_lost_id = lost_id
                
                if best_lost_id is not None:
                    # Re-identify player
                    lost_player = self.lost_players[best_lost_id]
                    lost_player.update(detection)
                    lost_player.time_since_update = 0
                    self.tracked_players[best_lost_id] = lost_player
                    del self.lost_players[best_lost_id]
                    reidentified = True
                
                if not reidentified:
                    # Create new track
                    new_player = TrackedPlayer(
                        track_id=self.next_id,
                        class_name=detection.class_name,
                        current_box=detection
                    )
                    self.tracked_players[self.next_id] = new_player
                    self.next_id += 1
        
        # Handle unassociated tracks (increment time since update)
        for track_id, tracked_player in list(self.tracked_players.items()):
            if track_id not in updated_tracks:
                tracked_player.time_since_update += 1
                tracked_player.hit_streak = 0
                
                # Only move to lost players if disappeared for more than 1 frame
                # This gives better chance for re-identification
                if tracked_player.time_since_update > min(3, self.max_disappeared // 3):
                    # Store additional context for better re-identification
                    tracked_player.last_seen_frame = frame.copy() if frame is not None else None
                    self.lost_players[track_id] = tracked_player
                    del self.tracked_players[track_id]
        
        # Clean up very old lost players (more conservative cleanup)
        for lost_id in list(self.lost_players.keys()):
            if self.lost_players[lost_id].time_since_update > self.max_disappeared:
                del self.lost_players[lost_id]
        
        return list(self.tracked_players.values())
    
    def draw_tracks(self, frame: np.ndarray, tracked_players: List[TrackedPlayer]) -> np.ndarray:
        """
        Draw tracking results on frame with enhanced visualization
        
        Args:
            frame: Input frame
            tracked_players: List of tracked players
            
        Returns:
            Frame with tracking visualization
        """
        output_frame = frame.copy()
        
        for player in tracked_players:
            box = player.current_box
            color = player.color
            
            # Draw bounding box with thickness based on confidence
            thickness = max(1, int(box.confidence * 3))
            cv2.rectangle(output_frame, (box.x1, box.y1), (box.x2, box.y2), color, thickness)
            
            # Draw predicted position
            predicted_pos = player.predict_next_position()
            cv2.circle(output_frame, predicted_pos, 3, color, -1)
            
            # Draw track ID, class, and confidence
            label = f"ID:{player.track_id} {player.class_name} ({box.confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(output_frame, 
                         (box.x1, box.y1 - label_size[1] - 10),
                         (box.x1 + label_size[0] + 5, box.y1),
                         color, -1)
            
            # Draw label text
            cv2.putText(output_frame, label, (box.x1 + 2, box.y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw trajectory with fading effect
            if len(player.history) > 1:
                points = list(player.history)
                for i in range(1, len(points)):
                    # Calculate alpha based on position in history
                    alpha = i / len(points)
                    line_color = tuple(int(c * alpha) for c in color)
                    cv2.line(output_frame, points[i-1], points[i], line_color, 2)
            
            # Draw velocity vector
            if player.velocity != (0.0, 0.0):
                center = box.center
                vel_end = (
                    int(center[0] + player.velocity[0] * 3),
                    int(center[1] + player.velocity[1] * 3)
                )
                cv2.arrowedLine(output_frame, center, vel_end, color, 2, tipLength=0.3)
        
        return output_frame
    
    def get_statistics(self) -> Dict:
        """Get comprehensive tracking statistics"""
        active_players = list(self.tracked_players.values())
        
        return {
            'active_tracks': len(self.tracked_players),
            'lost_tracks': len(self.lost_players),
            'total_tracks_created': self.next_id - 1,
            'track_ids': list(self.tracked_players.keys()),
            'lost_track_ids': list(self.lost_players.keys()),
            'average_track_age': np.mean([p.age for p in active_players]) if active_players else 0,
            'average_hit_streak': np.mean([p.hit_streak for p in active_players]) if active_players else 0,
            'class_distribution': {
                cls: len([p for p in active_players if p.class_name == cls])
                for cls in set(p.class_name for p in active_players)
            } if active_players else {},
            'tracking_stability': {
                'stable_tracks': len([p for p in active_players if p.hit_streak >= self.min_hit_streak_for_stable]),
                'new_tracks': len([p for p in active_players if p.hit_streak < self.min_hit_streak_for_stable]),
                'reidentified_tracks': len([p for p in active_players if p.track_id in self.lost_players])
            }
        }
    
    def validate_track_consistency(self, detections: List[DetectionBox]) -> Dict:
        """Validate tracking consistency for testing purposes"""
        validation_info = {
            'id_switches': 0,
            'track_continuity': {},
            'association_quality': []
        }
        
        # Check if all detections are associated
        associations = self.associate_detections_hungarian(detections, np.zeros((480, 640, 3), dtype=np.uint8))
        validation_info['association_coverage'] = len(associations) / len(detections) if detections else 1.0
        
        # Track continuity information
        for track_id, player in self.tracked_players.items():
            validation_info['track_continuity'][track_id] = {
                'hits': player.hits,
                'hit_streak': player.hit_streak,
                'age': player.age,
                'time_since_update': player.time_since_update
            }
        
        return validation_info