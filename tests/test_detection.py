import pytest
import cv2
import numpy as np
import torch
from pathlib import Path
from src.detection import PlayerDetector
from src.tracking import PlayerTracker, DetectionBox

# Fixture for creating a detector instance
@pytest.fixture
def player_detector():
    model_path = "data/models/player_detection_model.pt"
    if not Path(model_path).exists():
        pytest.skip("Model file not found")
    return PlayerDetector(model_path)

# Fixture for creating a tracker instance
@pytest.fixture
def player_tracker():
    return PlayerTracker(max_disappeared=30, max_distance=100)

# Fixture for dummy black frame
@pytest.fixture
def dummy_frame():
    return np.zeros((720, 1280, 3), dtype=np.uint8)

# Helper to get sample frames from outputs/frames
def get_sample_frames():
    frames_dir = Path("outputs/frames")
    if not frames_dir.exists():
        return []
    return list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))

# Test detector initialization
def test_detector_initialization(player_detector):
    assert player_detector is not None
    assert player_detector.model is not None

# Test detector on all sample frames
@pytest.mark.parametrize("frame_path", get_sample_frames())
def test_detector_on_sample_frame(player_detector, frame_path):
    image = cv2.imread(str(frame_path))
    if image is None:
        pytest.skip(f"Could not load image: {frame_path}")
    
    detections = player_detector.detect(image)
    
    assert isinstance(detections, list)
    assert len(detections) >= 0  # Accepts 0 or more

    # Filter for players
    player_detections = [d for d in detections if d['class_name'] == 'player']
    if player_detections:
        for detection in player_detections:
            assert 'bbox' in detection
            assert 'confidence' in detection
            assert detection['confidence'] > 0.3

# Test detector on an empty frame
def test_detector_on_empty_frame(player_detector, dummy_frame):
    detections = player_detector.detect(dummy_frame)
    assert isinstance(detections, list)
    assert all(d['confidence'] < 0.3 for d in detections)

# Test tracker with dummy detections
def test_tracker(player_tracker, dummy_frame):
    detections = [
        DetectionBox(100, 100, 200, 300, 0.9, 0, "player"),
        DetectionBox(500, 200, 600, 400, 0.85, 0, "player")
    ]
    
    tracked_players = player_tracker.update(detections, dummy_frame)
    assert len(tracked_players) == 2

    tracked_players = player_tracker.update(detections, dummy_frame)
    assert len(tracked_players) == 2
    assert all(player.hits > 1 for player in tracked_players)

    tracked_players = player_tracker.update([], dummy_frame)
    assert len(tracked_players) == 2
    assert all(player.time_since_update == 1 for player in tracked_players)

# Performance test
def test_detection_performance(player_detector, benchmark):
    sample_frames = get_sample_frames()
    if not sample_frames:
        pytest.skip("No sample frames available in outputs/frames/")
    
    image = cv2.imread(str(sample_frames[0]))
    if image is None:
        pytest.skip("Could not load the first sample frame")

    _ = player_detector.detect(image)  # Warmup
    benchmark(player_detector.detect, image)
    
    stats = benchmark.stats
    if torch.cuda.is_available():
        assert stats['mean'] < 0.1
    else:
        assert stats['mean'] < 1.5

# Test tracking with movement
def test_tracking_with_movement(player_tracker, dummy_frame):
    detections = [
        DetectionBox(100, 100, 200, 300, 0.9, 0, "player"),
        DetectionBox(500, 200, 600, 400, 0.85, 0, "player")
    ]
    
    players1 = player_tracker.update(detections, dummy_frame)

    moved_detections = [
        DetectionBox(110, 110, 210, 310, 0.9, 0, "player"),
        DetectionBox(510, 210, 610, 410, 0.85, 0, "player")
    ]

    players2 = player_tracker.update(moved_detections, dummy_frame)

    assert len(players1) == len(players2)
    assert all(p1.track_id == p2.track_id for p1, p2 in zip(players1, players2))

#Additional
# 1. ID consistency on identical frames
def test_tracker_id_consistency(player_tracker, dummy_frame):
    dets = [
        DetectionBox(100, 100, 200, 300, 0.9, 0, "player"),
        DetectionBox(500, 200, 600, 400, 0.85, 0, "player")
    ]
    first = player_tracker.update(dets, dummy_frame)
    second = player_tracker.update(dets, dummy_frame)

    ids1 = sorted([p.track_id for p in first])
    ids2 = sorted([p.track_id for p in second])
    assert ids1 == ids2, "Track IDs should not shuffle on identical input"

# 2. Robustness to small jitter
def test_tracker_jitter(player_tracker, dummy_frame):
    original = [
        DetectionBox(150, 150, 250, 350, 0.9, 0, "player"),
        DetectionBox(600, 250, 700, 450, 0.85, 0, "player")
    ]
    player_tracker.update(original, dummy_frame)

    # Jitter by ±5 pixels
    jittered = [
        DetectionBox(ov.x1 + 5, ov.y1 - 3, ov.x2 - 2, ov.y2 + 4, ov.confidence, ov.class_id, ov.class_name)
        for ov in original
    ]
    tracked = player_tracker.update(jittered, dummy_frame)
    assert all(p.hits > 1 for p in tracked), "Tracks should persist under small shifts"

# 3. Temporary occlusion and re‑identification
def test_tracker_temporary_occlusion(player_tracker, dummy_frame):
    dets = [DetectionBox(200, 200, 300, 400, 0.9, 0, "player")]
    # Frame 1: see player
    first = player_tracker.update(dets, dummy_frame)
    pid = first[0].track_id

    # Frame 2: miss player completely
    player_tracker.update([], dummy_frame)

    # Frame 3: player re‑appears at same location
    reappeared = player_tracker.update(dets, dummy_frame)
    assert any(p.track_id == pid for p in reappeared), "Player should be re‑identified after brief disappearance"
