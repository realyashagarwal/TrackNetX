# 🎯 TrackNetX

![Python](https://img.shields.io/badge/python-3.9-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**TrackNetX** is a real-time multi-object tracking system with re-identification capabilities, purpose-built for football analytics. Powered by YOLOv11 and advanced computer vision techniques, it consistently detects and tracks players, goalkeepers, referees, and the ball—even when they leave and re-enter the frame—ensuring persistent identity throughout the match.

## 🚀 Overview

TrackNetX automates player detection, tracking, and re-identification specifically for football videos. Whether it's analyzing key moments, tactical sequences, or full match footage, it enables consistent identity tracking of all field participants even through player clustering, occlusions, or brief exits from the camera view.

## ✨ Key Features

- 🕵️ **Real-time Detection**: Player and ball detection using YOLOv11
- 🔁 **Re-identification**: Tracks objects across frames and occlusions
- 🎥 **High Performance**: Real-time processing on 720p sports videos
- 🧠 **Multi-class Support**: 4 distinct classes: **player**, **ball**, **goalkeeper**, **referee**
- 📈 **Analytics**: Logs processing progress and summary statistics

## 🛠️ Setup Guide

### 1. Clone the Repository

```bash
git clone https://github.com/realyashagarwal/TrackNetX
cd TrackNetX
```

### 2. Create and Activate Environment

**Option A: Using Conda (Recommended)**

```bash
conda create -n tracknetx python=3.9 -y
conda activate tracknetx
```

**Option B: Using venv**

```bash
python -m venv player-tracking-env
source player-tracking-env/bin/activate  # Linux/Mac
# Windows: player-tracking-env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup the Project

```bash
python src/main.py --setup
```

### 5. Add Required Files

- Download the YOLOv11 model and place it at: `data/models/player_detection_model.pt`
- Add your video to: `data/videos/15sec_input_720p.mp4`

## 🚀 Quick Start

### Test Installation

```bash
python src/main.py --test
```

### Process a Video

```bash
python src/main.py --video data/videos/15sec_input_720p.mp4
```

### Custom Output Location

```bash
python src/main.py --video data/videos/your_clip.mp4 --output outputs/custom_result.mp4
```

## 📁 Project Structure

```
TrackNetX/
├── data/
│   ├── videos/          # Input videos
│   └── models/          # Model weights
├── src/
│   ├── detection.py     # Detection logic
│   ├── tracking.py      # Tracking and re-ID
│   ├── main.py          # Entry point
│   └── utils.py         # Helper functions
├── outputs/
│   ├── frames/          # Intermediate visualizations
│   └── results/         # Final processed videos
├── tests/               # Unit/integration tests
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## 🤖 Model Information

- **Base Model**: YOLOv11 (Ultralytics)
- **Fine-tuned for**: Sports analytics
- **Tracked Classes**: player, ball, goalkeeper, referee
- **Input Format**: 720p videos
- **Performance**: ~17 FPS on RTX 3070 (FP16 enabled)

## 📦 Usage Examples

### Basic Usage

```bash
# 1. Install environment
conda create -n tracknetx python=3.9
conda activate tracknetx
pip install -r requirements.txt

# 2. Add your video
cp your_game_video.mp4 data/videos/

# 3. Run the tracker
python src/main.py --video data/videos/your_game_video.mp4

# 4. View results in outputs/tracked_video.mp4
```

### Advanced Usage

```bash
# Custom output path
python src/main.py --video data/videos/game_clip.mp4 --output outputs/my_tracking.mp4

# Test mode
python src/main.py --test

# Setup mode
python src/main.py --setup
```

## 📊 Expected Output

When running the system, you can expect:

- ✅ Model initialized with **GPU + FP16** acceleration
- ✅ Detection of 4 object classes
- ✅ Video properties analysis (resolution, FPS, frame count)
- ✅ Unique identity tracking across frames
- ✅ Final processed video saved to specified output path

**Example processing log:**
```
✅ Model initialized with GPU + FP16
✅ 4 object classes detected
✅ Detected video properties: 1280×720, 25 FPS, 375 frames
✅ Tracked 119 unique identities
✅ Final active tracks: 3
📍 Output saved at: outputs/tracked_video.mp4
```

## 🔧 Troubleshooting

### Common Issues

1. **Model file not found**: Ensure `player_detection_model.pt` is in `data/models/`
2. **Video format issues**: Use MP4 format for best compatibility
3. **Performance issues**: Enable GPU acceleration and FP16 for better performance

### System Requirements

- Python 3.9+
- CUDA-compatible GPU (recommended)
- OpenCV-compatible video formats

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- YOLOv11 by Ultralytics
- OpenCV community

---

**Happy Tracking! 🎯**