import pytest
import cv2
import numpy as np
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from src.detection import PlayerDetector, DetectionBox
from src.tracking import PlayerTracker
import time
import gc
import psutil
import threading

# ============= FIXTURES =============

@pytest.fixture
def player_detector():
    """Fixture for creating a detector instance"""
    model_path = "data/models/player_detection_model.pt"
    if not Path(model_path).exists():
        pytest.skip("Model file not found")
    return PlayerDetector(model_path)

@pytest.fixture
def mock_detector():
    """Fixture for creating a mock detector for testing without actual model"""
    with patch('src.detection.YOLO') as mock_yolo:
        mock_model = Mock()
        mock_model.model.names = {0: 'player', 1: 'ball'}
        mock_model.model.half.return_value = mock_model.model
        mock_model.model.to.return_value = mock_model.model
        mock_yolo.return_value = mock_model
        
        detector = PlayerDetector("dummy_model.pt")
        detector.model = mock_model
        return detector

@pytest.fixture
def player_tracker():
    """Fixture for creating a tracker instance"""
    return PlayerTracker(
        max_disappeared=30, 
        max_distance=100, 
        iou_threshold=0.1,
        confidence_threshold=0.5
    )

@pytest.fixture
def dummy_frame():
    """Fixture for dummy black frame"""
    return np.zeros((720, 1280, 3), dtype=np.uint8)

@pytest.fixture
def small_frame():
    """Fixture for small test frame"""
    return np.zeros((100, 100, 3), dtype=np.uint8)

@pytest.fixture
def large_frame():
    """Fixture for large test frame"""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)

@pytest.fixture
def colored_frame():
    """Fixture for colored test frame"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # Blue channel
    return frame

@pytest.fixture
def temporary_test_image():
    """Fixture for temporary test image file"""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        # Create a simple test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(tmp_file.name, test_image)
        yield tmp_file.name
        os.unlink(tmp_file.name)

# ============= HELPER FUNCTIONS =============

def get_sample_frames():
    """Helper to get sample frames from outputs/frames"""
    frames_dir = Path("outputs/frames")
    if not frames_dir.exists():
        return []
    return list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))

def create_test_detections():
    """Helper to create valid test detections"""
    return [
        DetectionBox(100, 100, 200, 300, 0.9, 0, "player"),
        DetectionBox(500, 200, 600, 400, 0.85, 0, "player"),
        DetectionBox(300, 300, 400, 500, 0.75, 0, "player")
    ]

def create_overlapping_detections():
    """Helper to create overlapping detections for IoU testing"""
    return [
        DetectionBox(100, 100, 200, 200, 0.9, 0, "player"),
        DetectionBox(150, 150, 250, 250, 0.85, 0, "player"),  # Overlapping
        DetectionBox(300, 300, 400, 400, 0.8, 0, "player")
    ]

def create_mock_yolo_results(detections):
    """Helper to create mock YOLO results"""
    mock_results = []
    mock_result = Mock()
    mock_boxes = Mock()
    
    # Create mock boxes data
    boxes_data = []
    for det in detections:
        box_mock = Mock()
        box_mock.xyxy = [[det.x1, det.y1, det.x2, det.y2]]
        box_mock.conf = [det.confidence]
        box_mock.cls = [det.class_id]
        boxes_data.append(box_mock)
    
    mock_boxes.cpu.return_value.numpy.return_value = boxes_data
    mock_result.boxes = mock_boxes
    mock_results.append(mock_result)
    
    return mock_results

# ============= DETECTION BOX TESTS =============

class TestDetectionBox:
    """Test suite for DetectionBox dataclass"""
    
    def test_detection_box_creation(self):
        """Test DetectionBox creation and basic properties"""
        box = DetectionBox(10, 20, 100, 200, 0.9, 0, "player")
        assert box.x1 == 10
        assert box.y1 == 20
        assert box.x2 == 100
        assert box.y2 == 200
        assert box.confidence == 0.9
        assert box.class_id == 0
        assert box.class_name == "player"
    
    def test_detection_box_properties(self):
        """Test DetectionBox computed properties"""
        box = DetectionBox(10, 20, 100, 200, 0.9, 0, "player")
        
        # Test center calculation
        center = box.center
        assert center == (55, 110)  # ((10+100)/2, (20+200)/2)
        
        # Test width and height
        assert box.width == 90  # 100 - 10
        assert box.height == 180  # 200 - 20
        
        # Test area
        assert box.area == 16200  # 90 * 180
    
    def test_detection_box_edge_cases(self):
        """Test DetectionBox with edge cases"""
        # Zero-sized box
        box = DetectionBox(10, 10, 10, 10, 0.5, 0, "player")
        assert box.width == 0
        assert box.height == 0
        assert box.area == 0
        
        # Single pixel box
        box = DetectionBox(10, 10, 11, 11, 0.5, 0, "player")
        assert box.width == 1
        assert box.height == 1
        assert box.area == 1

# ============= PLAYER DETECTOR TESTS =============

class TestPlayerDetector:
    """Test suite for PlayerDetector class"""
    
    def test_detector_initialization(self, player_detector):
        """Test detector initialization"""
        assert player_detector is not None
        assert player_detector.model is not None
        assert player_detector.device is not None
        assert player_detector.img_size == (256, 448)
        
        # Test model info
        info = player_detector.get_model_info()
        assert 'model_path' in info
        assert 'device' in info
        assert 'input_size' in info
        assert 'classes' in info
        assert 'num_classes' in info
    
    def test_detector_initialization_with_custom_params(self):
        """Test detector initialization with custom parameters"""
        custom_size = (320, 640)
        with patch('src.detection.YOLO'):
            detector = PlayerDetector("dummy_model.pt", img_size=custom_size)
            assert detector.img_size == custom_size
    
    @patch('torch.cuda.is_available')
    def test_detector_device_selection(self, mock_cuda):
        """Test device selection logic"""
        # Test CUDA available
        mock_cuda.return_value = True
        with patch('src.detection.YOLO'):
            detector = PlayerDetector("dummy_model.pt")
            assert detector.device.type == 'cuda'
        
        # Test CUDA not available
        mock_cuda.return_value = False
        with patch('src.detection.YOLO'):
            detector = PlayerDetector("dummy_model.pt")
            assert detector.device.type == 'cpu'
    
    def test_detector_preprocess(self, mock_detector, dummy_frame):
        """Test image preprocessing"""
        with patch.object(mock_detector, 'device', torch.device('cpu')):
            tensor = mock_detector.preprocess(dummy_frame)
            
            # Check tensor properties
            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape[0] == 1  # Batch dimension
            assert tensor.shape[1] == 3  # RGB channels
            assert tensor.shape[2] == mock_detector.img_size[0]  # Height
            assert tensor.shape[3] == mock_detector.img_size[1]  # Width
            assert tensor.dtype == torch.float32
            assert tensor.min() >= 0.0 and tensor.max() <= 1.0
    
    def test_detector_on_empty_frame(self, player_detector, dummy_frame):
        """Test detector on empty frame"""
        detections = player_detector.detect(dummy_frame)
        assert isinstance(detections, list)
        
        # Test dataclass format
        detections_dc = player_detector.detect(dummy_frame, return_format='dataclass')
        assert isinstance(detections_dc, list)
        assert all(isinstance(d, DetectionBox) for d in detections_dc)
    
    def test_detector_edge_cases(self, player_detector):
        """Test detector with edge cases"""
        # Test with None
        result = player_detector.detect(None)
        assert result == []
        
        # Test with empty array
        empty_frame = np.array([])
        result = player_detector.detect(empty_frame)
        assert result == []
        
        # Test with grayscale
        gray_frame = np.zeros((100, 100), dtype=np.uint8)
        result = player_detector.detect(gray_frame)
        assert isinstance(result, list)
        
        # Test with RGBA
        rgba_frame = np.zeros((100, 100, 4), dtype=np.uint8)
        result = player_detector.detect(rgba_frame)
        assert isinstance(result, list)
    
    @pytest.mark.parametrize("conf_threshold", [0.1, 0.5, 0.9])
    def test_detector_confidence_thresholds(self, player_detector, dummy_frame, conf_threshold):
        """Test detector with different confidence thresholds"""
        detections = player_detector.detect(dummy_frame, conf_threshold=conf_threshold)
        assert isinstance(detections, list)
        
        # All detections should meet confidence threshold
        for det in detections:
            assert det['confidence'] >= conf_threshold
    
    @pytest.mark.parametrize("size", [(640, 480), (1280, 720), (1920, 1080), (100, 100)])
    def test_detector_image_sizes(self, player_detector, size):
        """Test detector with various image sizes"""
        dummy_frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        detections = player_detector.detect(dummy_frame)
        assert isinstance(detections, list)
    
    def test_detector_coordinate_scaling(self, mock_detector):
        """Test coordinate scaling back to original image size"""
        # Create a frame and mock detection results
        original_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock YOLO results
        mock_detection = Mock()
        mock_detection.x1, mock_detection.y1 = 50, 50
        mock_detection.x2, mock_detection.y2 = 100, 100
        mock_detection.confidence = 0.9
        mock_detection.class_id = 0
        
        mock_results = create_mock_yolo_results([mock_detection])
        
        with patch.object(mock_detector.model, '__call__', return_value=mock_results):
            detections = mock_detector.detect(original_frame)
            
            if detections:
                # Check that coordinates are scaled properly
                bbox = detections[0]['bbox']
                assert all(coord >= 0 for coord in bbox)
                assert bbox[2] <= original_frame.shape[1]  # x2 <= width
                assert bbox[3] <= original_frame.shape[0]  # y2 <= height
    
    def test_detector_player_only_method(self, player_detector, dummy_frame):
        """Test the detect_players method"""
        players = player_detector.detect_players(dummy_frame)
        assert isinstance(players, list)
        assert all(isinstance(p, DetectionBox) for p in players)
        assert all(p.class_name == 'player' for p in players)
    
    def test_detector_visualization(self, player_detector, dummy_frame):
        """Test detection visualization"""
        detections = create_test_detections()
        
        # Test with DetectionBox objects
        viz_frame = player_detector.visualize_detections(dummy_frame, detections)
        assert viz_frame.shape == dummy_frame.shape
        assert not np.array_equal(viz_frame, dummy_frame)  # Should be different
        
        # Test with dict format
        dict_detections = [
            {
                'bbox': [det.x1, det.y1, det.x2, det.y2],
                'confidence': det.confidence,
                'class_name': det.class_name
            }
            for det in detections
        ]
        viz_frame2 = player_detector.visualize_detections(dummy_frame, dict_detections)
        assert viz_frame2.shape == dummy_frame.shape
    
    def test_detector_visualization_options(self, player_detector, dummy_frame):
        """Test visualization with different options"""
        detections = create_test_detections()
        
        # Test without confidence
        viz_frame = player_detector.visualize_detections(
            dummy_frame, detections, show_confidence=False
        )
        assert viz_frame.shape == dummy_frame.shape
        
        # Test without class names
        viz_frame = player_detector.visualize_detections(
            dummy_frame, detections, show_class=False
        )
        assert viz_frame.shape == dummy_frame.shape
        
        # Test with neither
        viz_frame = player_detector.visualize_detections(
            dummy_frame, detections, show_confidence=False, show_class=False
        )
        assert viz_frame.shape == dummy_frame.shape

# ============= TRACKER TESTS =============

class TestPlayerTracker:
    """Test suite for PlayerTracker class"""
    
    def test_tracker_initialization(self, player_tracker):
        """Test tracker initialization"""
        assert player_tracker.max_disappeared == 30
        assert player_tracker.max_distance == 100
        assert player_tracker.next_id == 1
        assert len(player_tracker.tracked_players) == 0
        assert player_tracker.iou_threshold == 0.1
        assert player_tracker.confidence_threshold == 0.5
    
    def test_tracker_basic_functionality(self, player_tracker, dummy_frame):
        """Test basic tracker functionality"""
        detections = create_test_detections()
        
        # First update
        tracked_players = player_tracker.update(detections, dummy_frame)
        assert len(tracked_players) == 3  # Should filter high confidence detections
        assert all(p.hits == 1 for p in tracked_players)
        assert all(p.track_id in [1, 2, 3] for p in tracked_players)
        
        # Second update with same detections
        tracked_players = player_tracker.update(detections, dummy_frame)
        assert len(tracked_players) == 3
        assert all(p.hits == 2 for p in tracked_players)
        assert all(p.time_since_update == 0 for p in tracked_players)
    
    def test_tracker_no_detections(self, player_tracker, dummy_frame):
        """Test tracker with no detections"""
        detections = create_test_detections()
        
        # Initialize with detections
        tracked_players = player_tracker.update(detections, dummy_frame)
        assert len(tracked_players) == 3
        
        # Update with no detections
        tracked_players = player_tracker.update([], dummy_frame)
        assert len(tracked_players) == 3  # Should still track for a while
        assert all(p.time_since_update == 1 for p in tracked_players)
    
    def test_tracker_id_consistency(self, player_tracker, dummy_frame):
        """Test ID consistency on identical frames"""
        detections = create_test_detections()
        
        first = player_tracker.update(detections, dummy_frame)
        second = player_tracker.update(detections, dummy_frame)
        
        ids1 = sorted([p.track_id for p in first])
        ids2 = sorted([p.track_id for p in second])
        assert ids1 == ids2, "Track IDs should not change on identical input"
    
    def test_tracker_confidence_filtering(self, player_tracker, dummy_frame):
        """Test confidence threshold filtering"""
        low_conf_detections = [
            DetectionBox(100, 100, 200, 200, 0.3, 0, "player"),  # Below threshold
            DetectionBox(300, 300, 400, 400, 0.7, 0, "player"),  # Above threshold
        ]
        
        tracked = player_tracker.update(low_conf_detections, dummy_frame)
        assert len(tracked) == 1  # Only high confidence detection
        assert len([d for d in low_conf_detections if d.confidence >= player_tracker.confidence_threshold]) == len(tracked)
    
    def test_tracker_jitter_robustness(self, player_tracker, dummy_frame):
        """Test tracker robustness to small jitter"""
        original = create_test_detections()
        
        # Initial update
        player_tracker.update(original, dummy_frame)
        
        # Add small jitter (±5 pixels)
        jittered = [
            DetectionBox(
                det.x1 + np.random.randint(-5, 6),
                det.y1 + np.random.randint(-5, 6),
                det.x2 + np.random.randint(-5, 6),
                det.y2 + np.random.randint(-5, 6),
                det.confidence,
                det.class_id,
                det.class_name
            ) for det in original
        ]
        
        tracked = player_tracker.update(jittered, dummy_frame)
        assert len(tracked) == 3
        assert all(p.hits >= 2 for p in tracked), "Tracks should persist under small jitter"
    
    def test_tracker_movement_tracking(self, player_tracker, dummy_frame):
        """Test tracking with gradual movement"""
        detections = create_test_detections()
        
        # Initial position
        players1 = player_tracker.update(detections, dummy_frame)
        
        # Move detections slightly
        moved_detections = [
            DetectionBox(det.x1 + 10, det.y1 + 10, det.x2 + 10, det.y2 + 10,
                        det.confidence, det.class_id, det.class_name)
            for det in detections
        ]
        
        players2 = player_tracker.update(moved_detections, dummy_frame)
        
        # Should maintain same IDs
        assert len(players1) == len(players2)
        ids1 = sorted([p.track_id for p in players1])
        ids2 = sorted([p.track_id for p in players2])
        assert ids1 == ids2
    
    def test_tracker_temporary_occlusion(self, player_tracker, dummy_frame):
        """Test re-identification after temporary occlusion"""
        det = [DetectionBox(200, 200, 300, 400, 0.9, 0, "player")]
        
        # Frame 1: see player
        first = player_tracker.update(det, dummy_frame)
        original_id = first[0].track_id
        
        # Frame 2-3: miss player (but not too long)
        player_tracker.update([], dummy_frame)
        player_tracker.update([], dummy_frame)
        
        # Frame 4: player re-appears
        reappeared = player_tracker.update(det, dummy_frame)
        
        # Should either re-identify or create new track
        assert len(reappeared) >= 1
        reappeared_ids = [p.track_id for p in reappeared]
        assert original_id in reappeared_ids or max(reappeared_ids) > original_id
    
    def test_tracker_max_distance_threshold(self, player_tracker, dummy_frame):
        """Test max distance threshold enforcement"""
        # Start with detection at center (150, 150)
        det1 = [DetectionBox(100, 100, 200, 200, 0.9, 0, "player")]
        tracked1 = player_tracker.update(det1, dummy_frame)
        original_id = tracked1[0].track_id
        
        # Move very far (center moves from 150,150 to 450,450 = ~424px distance)
        det2 = [DetectionBox(400, 400, 500, 500, 0.9, 0, "player")]
        tracked2 = player_tracker.update(det2, dummy_frame)
        
        # Should create new track due to distance
        if len(tracked2) > 0:
            new_ids = [p.track_id for p in tracked2]
            # Either no tracks with original ID or new ID assigned
            assert original_id not in new_ids or len(new_ids) > 1
    
    def test_tracker_max_disappeared_cleanup(self, player_tracker, dummy_frame):
        """Test cleanup of disappeared tracks"""
        det = [DetectionBox(100, 100, 200, 200, 0.9, 0, "player")]
        tracked = player_tracker.update(det, dummy_frame)
        
        # Update with no detections for many frames
        for i in range(player_tracker.max_disappeared + 5):
            tracked = player_tracker.update([], dummy_frame)
        
        # Should eventually clean up
        assert len(tracked) == 0
    
    def test_tracker_statistics(self, player_tracker, dummy_frame):
        """Test tracker statistics"""
        detections = create_test_detections()
        
        # Initial state
        stats = player_tracker.get_statistics()
        assert stats['active_tracks'] == 0
        assert stats['total_tracks_created'] == 0
        
        # After adding tracks
        player_tracker.update(detections, dummy_frame)
        stats = player_tracker.get_statistics()
        assert stats['active_tracks'] == 3
        assert stats['total_tracks_created'] == 3
        assert len(stats['track_ids']) == 3

    # Complete frame exit/re-entry
    def test_tracker_full_reentry(self,player_tracker, dummy_frame):
        # Initial appearance
        det1 = [DetectionBox(100, 100, 200, 200, 0.9, 0, "player")]
        tracked = player_tracker.update(det1, dummy_frame)
        original_id = tracked[0].track_id
        
        # Complete disappearance (beyond frame bounds)
        for _ in range(10):
            player_tracker.update([], dummy_frame)
            
        # Reappearance at different location
        det2 = [DetectionBox(50, 50, 150, 150, 0.9, 0, "player")]
        tracked = player_tracker.update(det2, dummy_frame)
        
        assert tracked[0].track_id == original_id  # Critical requirement
    
    def test_tracker_validation(self, player_tracker, dummy_frame):
        """Test tracker validation"""
        detections = create_test_detections()
        player_tracker.update(detections, dummy_frame)
        
        validation = player_tracker.validate_track_consistency(detections)
        assert 'association_coverage' in validation
        assert 'track_continuity' in validation
        assert validation['association_coverage'] >= 0

    # ENHANCED occlusion test (goal scenario)
    def test_goal_event_reidentification(self,player_tracker, dummy_frame):
        # Player near goal
        goal_det = [DetectionBox(600, 300, 700, 400, 0.9, 0, "player")]
        player_tracker.update(goal_det, dummy_frame)
        
        # Player disappears (crowd occlusion)
        for _ in range(15):  # Within max_disappeared
            player_tracker.update([], dummy_frame)
            
        # Reappearance after goal event
        reappear = [DetectionBox(620, 280, 720, 380, 0.9, 0, "player")]
        tracked = player_tracker.update(reappear, dummy_frame)
        
        assert len(tracked) == 1 and tracked[0].hits > 1

# ============= INTEGRATION TESTS =============

class TestIntegration:
    """Integration tests between detector and tracker"""
    
    def test_detector_tracker_integration(self, player_detector, player_tracker, dummy_frame):
        """Test integration between detector and tracker"""
        # Get detections from detector
        detections_dict = player_detector.detect(dummy_frame)
        
        # Convert to DetectionBox format for tracker
        detections_box = []
        for det in detections_dict:
            if det['class_name'] == 'player':
                bbox = det['bbox']
                detections_box.append(DetectionBox(
                    x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3],
                    confidence=det['confidence'],
                    class_id=det['class_id'],
                    class_name=det['class_name']
                ))
        
        # Track detections
        tracked = player_tracker.update(detections_box, dummy_frame)
        assert isinstance(tracked, list)
    
    def test_detector_player_method_integration(self, player_detector, player_tracker, dummy_frame):
        """Test integration using detect_players method"""
        # Get player detections directly
        player_detections = player_detector.detect_players(dummy_frame)
        
        # Track them
        tracked = player_tracker.update(player_detections, dummy_frame)
        assert isinstance(tracked, list)
        assert all(isinstance(p, type(tracked[0])) for p in tracked) if tracked else True
    
    def test_full_pipeline_simulation(self, player_detector, player_tracker):
        """Test full pipeline with simulated video frames"""
        frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        
        all_tracks = []
        for frame in frames:
            detections = player_detector.detect_players(frame)
            tracked = player_tracker.update(detections, frame)
            all_tracks.append(tracked)
        
        # Should have consistent tracking across frames
        assert len(all_tracks) == 10
        assert all(isinstance(tracks, list) for tracks in all_tracks)

# ============= PERFORMANCE TESTS =============

class TestPerformance:
    """Performance tests for detection and tracking"""
    
    @pytest.mark.performance
    def test_detection_performance(self, player_detector, benchmark):
        """Test detection performance"""
        sample_frames = get_sample_frames()
        if not sample_frames:
            pytest.skip("No sample frames available in outputs/frames/")
        
        image = cv2.imread(str(sample_frames[0]))
        if image is None:
            pytest.skip("Could not load the first sample frame")
        
        # Warmup
        for _ in range(3):
            _ = player_detector.detect(image)
        
        # Benchmark
        result = benchmark(player_detector.detect, image)
        
        # Performance assertions
        mean_time = benchmark.stats['mean']
        fps = 1 / mean_time if mean_time > 0 else float('inf')
        
        # More realistic performance expectations
        if torch.cuda.is_available():
            assert mean_time < 0.5, f"Detection too slow on GPU: {mean_time:.4f}s"
            assert fps > 2, f"FPS too low on GPU: {fps:.1f}"
        else:
            assert mean_time < 2.0, f"Detection too slow on CPU: {mean_time:.4f}s"
            assert fps > 0.5, f"FPS too low on CPU: {fps:.1f}"
    
    @pytest.mark.performance
    def test_tracking_performance(self, player_tracker, dummy_frame, benchmark):
        """Test tracking performance"""
        detections = create_test_detections()
        
        # Warmup
        for _ in range(10):
            player_tracker.update(detections, dummy_frame)
        
        # Benchmark
        result = benchmark(player_tracker.update, detections, dummy_frame)
        
        # Tracking should be very fast
        mean_time = benchmark.stats['mean']
        assert mean_time < 0.01, f"Tracking too slow: {mean_time:.4f}s"
    
    @pytest.mark.performance
    def test_memory_usage(self, player_tracker, dummy_frame):
        """Test memory usage during tracking"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        detections = create_test_detections()
        
        # Run many tracking updates
        for i in range(1000):
            player_tracker.update(detections, dummy_frame)
            
            # Check memory every 100 iterations
            if i % 100 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                
                # Should not increase memory significantly
                assert memory_increase < 100, f"Memory usage increased by {memory_increase:.1f}MB"
    
    @pytest.mark.performance
    def test_concurrent_detection(self, player_detector):
        """Test concurrent detection calls"""
        frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        results = []
        threads = []
        
        def detect_frame(frame):
            result = player_detector.detect(frame)
            results.append(result)
        
        # Start concurrent detections
        for frame in frames:
            thread = threading.Thread(target=detect_frame, args=(frame,))
            threads.append(thread)
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        # All should complete successfully
        assert len(results) == 5
        assert all(isinstance(r, list) for r in results)