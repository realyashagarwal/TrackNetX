import cv2
import numpy as np
import os
from pathlib import Path

def create_output_dirs():
    """Create necessary output directories"""
    dirs = ['outputs/frames', 'outputs/results', 'data/videos', 'data/models']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✅ Output directories created successfully!")

def get_video_info(video_path):
    """Get basic video information"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    cap.release()
    
    return {
        'fps': fps,
        'width': width,
        'height': height,
        'total_frames': total_frames,
        'duration': duration
    }

def extract_sample_frames(video_path, num_samples=5):
    """Extract sample frames from video for analysis"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices for sampling
    frame_indices = np.linspace(0, total_frames-1, num_samples, dtype=int)
    
    sample_frames = []
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Save frame
            output_path = f'outputs/frames/sample_frame_{i:03d}.jpg'
            cv2.imwrite(output_path, frame)
            sample_frames.append(output_path)
            print(f"📸 Saved sample frame {i+1}/{num_samples}: {output_path}")
    
    cap.release()
    return sample_frames

if __name__ == "__main__":
    create_output_dirs()
    print("🔧 Utils module ready!")