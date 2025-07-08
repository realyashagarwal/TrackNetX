import cv2
import argparse
from pathlib import Path
import sys
import os
import time
from typing import List

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import create_output_dirs, get_video_info, extract_sample_frames
from detection import PlayerDetector
from tracking import PlayerTracker, DetectionBox, TrackedPlayer

def setup_project():
    """Initial project setup"""
    print("🚀 Setting up TrackNetX project...")
    create_output_dirs()
    
    # Check if model exists
    model_path = "data/models/player_detection_model.pt"
    if not Path(model_path).exists():
        print(f"⚠️  Model not found at {model_path}")
        print("📥 Please download the model from the provided link and place it in data/models/")
        return False
    
    # Check if video exists
    video_path = "data/videos/15sec_input_720p.mp4"
    if not Path(video_path).exists():
        print(f"⚠️  Video not found at {video_path}")
        print("📥 Please place your input video in data/videos/")
        return False
    
    return True

def convert_detections_to_tracking_format(detections: List[dict]) -> List[DetectionBox]:
    """Convert detection format to tracking format"""
    tracking_detections = []
    
    for detection in detections:
        # Assuming detection format: {'bbox': [x1, y1, x2, y2], 'confidence': float, 'class_name': str, 'class_id': int}
        bbox = detection['bbox']
        detection_box = DetectionBox(
            x1=int(bbox[0]),
            y1=int(bbox[1]), 
            x2=int(bbox[2]),
            y2=int(bbox[3]),
            confidence=detection['confidence'],
            class_id=detection.get('class_id', 0),
            class_name=detection['class_name']
        )
        tracking_detections.append(detection_box)
    
    return tracking_detections

def test_detection():
    """Test detection on sample frames"""
    print("\n🔍 Testing detection capabilities...")
    
    video_path = "data/videos/15sec_input_720p.mp4"
    model_path = "data/models/player_detection_model.pt"
    
    # Get video info
    video_info = get_video_info(video_path)
    print(f"📹 Video Info:")
    print(f"   Resolution: {video_info['width']}x{video_info['height']}")
    print(f"   FPS: {video_info['fps']:.2f}")
    print(f"   Duration: {video_info['duration']:.2f} seconds")
    print(f"   Total frames: {video_info['total_frames']}")
    
    # Extract sample frames
    print("\n📸 Extracting sample frames...")
    sample_frames = extract_sample_frames(video_path, num_samples=5)
    
    # Initialize detector
    detector = PlayerDetector(model_path)
    
    # Test detection on each sample frame
    total_detections = 0
    for i, frame_path in enumerate(sample_frames):
        print(f"\n🔍 Processing sample frame {i+1}...")
        
        # Load frame
        frame = cv2.imread(frame_path)
        
        # Detect players
        detections = detector.detect(frame)
        total_detections += len(detections)
        
        print(f"   Found {len(detections)} detections:")
        for j, detection in enumerate(detections):
            print(f"     • Detection {j+1}: {detection['class_name']} "
                  f"(confidence: {detection['confidence']:.2f})")
        
        # Visualize detections
        vis_frame = detector.visualize_detections(frame, detections)
        
        # Save annotated frame
        output_path = f"outputs/frames/detected_frame_{i:03d}.jpg"
        cv2.imwrite(output_path, vis_frame)
        print(f"   💾 Saved annotated frame: {output_path}")
    
    print(f"\n📊 Summary:")
    print(f"   Total detections across all frames: {total_detections}")
    print(f"   Average detections per frame: {total_detections/len(sample_frames):.1f}")

def test_tracking():
    """Test tracking on sample frames"""
    print("\n🎯 Testing tracking capabilities...")
    
    video_path = "data/videos/15sec_input_720p.mp4"
    model_path = "data/models/player_detection_model.pt"
    
    # Initialize detector and tracker
    detector = PlayerDetector(model_path)
    tracker = PlayerTracker(max_disappeared=30, max_distance=100)
    
    # Extract sample frames for tracking test
    sample_frames = extract_sample_frames(video_path, num_samples=5)
    
    print(f"🔄 Processing {len(sample_frames)} frames for tracking...")
    
    all_tracked_players = []
    
    for i, frame_path in enumerate(sample_frames):
        print(f"\n🔍 Processing frame {i+1} for tracking...")
        
        # Load frame
        frame = cv2.imread(frame_path)
        
        # Detect players
        detections = detector.detect(frame)
        
        # Convert to tracking format
        tracking_detections = convert_detections_to_tracking_format(detections)
        
        # Update tracker
        tracked_players = tracker.update(tracking_detections, frame)
        all_tracked_players.extend(tracked_players)
        
        print(f"   🎯 Active tracks: {len(tracked_players)}")
        for player in tracked_players:
            print(f"     • Track ID {player.track_id}: {player.class_name} "
                  f"(age: {player.age}, hits: {player.hits})")
        
        # Draw tracking results
        tracked_frame = tracker.draw_tracks(frame, tracked_players)
        
        # Save tracked frame
        output_path = f"outputs/frames/tracked_frame_{i:03d}.jpg"
        cv2.imwrite(output_path, tracked_frame)
        print(f"   💾 Saved tracked frame: {output_path}")
    
    # Get tracking statistics
    stats = tracker.get_statistics()
    print(f"\n📊 Tracking Summary:")
    print(f"   Total unique tracks created: {stats['total_tracks_created']}")
    print(f"   Currently active tracks: {stats['active_tracks']}")
    print(f"   Lost tracks: {stats['lost_tracks']}")
    print(f"   Active track IDs: {stats['track_ids']}")

def process_full_video(video_path: str, output_path: str = None):
    """Process full video with detection and tracking"""
    print(f"\n🎬 Processing full video: {video_path}")
    
    model_path = "data/models/player_detection_model.pt"
    
    # Initialize detector and tracker
    detector = PlayerDetector(model_path)
    tracker = PlayerTracker(max_disappeared=30, max_distance=100)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup output video writer
    if output_path is None:
        output_path = "outputs/tracked_video.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    print("🔄 Processing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect players
        detections = detector.detect(frame)
        
        # Convert to tracking format
        tracking_detections = convert_detections_to_tracking_format(detections)
        
        # Update tracker
        tracked_players = tracker.update(tracking_detections, frame)
        
        # Draw tracking results
        tracked_frame = tracker.draw_tracks(frame, tracked_players)
        
        # Write frame
        out.write(tracked_frame)
        
        # Progress update
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            elapsed = time.time() - start_time
            fps_processing = frame_count / elapsed
            print(f"   Progress: {progress:.1f}% ({frame_count}/{total_frames}) "
                  f"- Processing FPS: {fps_processing:.1f}")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Final statistics
    stats = tracker.get_statistics()
    elapsed = time.time() - start_time
    
    print(f"\n✅ Video processing complete!")
    print(f"   📁 Output saved to: {output_path}")
    print(f"   ⏱️  Processing time: {elapsed:.2f} seconds")
    print(f"   🎯 Total unique tracks: {stats['total_tracks_created']}")
    print(f"   📊 Final active tracks: {stats['active_tracks']}")

def main():
    """Main function"""
    print("🎯 TrackNetX - Sports Player Tracking System")
    print("=" * 50)
    
    parser = argparse.ArgumentParser(description='TrackNetX: Sports Player Tracking System')
    parser.add_argument('--setup', action='store_true', help='Run initial setup')
    parser.add_argument('--test', action='store_true', help='Test detection on sample frames')
    parser.add_argument('--test-tracking', action='store_true', help='Test tracking on sample frames')
    parser.add_argument('--video', type=str, help='Path to input video for full processing')
    parser.add_argument('--output', type=str, help='Path to output video (optional)')
    
    args = parser.parse_args()
    
    if args.setup or not any(vars(args).values()):
        if not setup_project():
            return
    
    if args.test:
        test_detection()
    
    if args.test_tracking:
        test_tracking()
    
    if args.video:
        if Path(args.video).exists():
            process_full_video(args.video, args.output)
        else:
            print(f"❌ Video file not found: {args.video}")
            return
    
    print("\n✅ TrackNetX operations complete!")
    print("📋 Available commands:")
    print("   --setup: Initialize project structure")
    print("   --test: Test detection capabilities")
    print("   --test-tracking: Test tracking capabilities")
    print("   --video [path]: Process full video with tracking")
    print("   --output [path]: Specify output video path (optional)")

if __name__ == "__main__":
    main()