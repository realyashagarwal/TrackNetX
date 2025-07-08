# TrackNetX 🎯

**TrackNetX is a real-time multi-object tracking system with re-identification capabilities, purpose-built for sports analytics. Powered by YOLOv11 and advanced computer vision techniques, the system can detect and consistently track players across frames—even when they leave and re-enter the scene—ensuring persistent identity over time.**

## 🚀 Overview

TrackNetX provides automated player tracking and re-identification capabilities for sports video analysis, enabling consistent player identification even when players temporarily leave the frame.

## ✨ Key Features

- Real-time player detection using YOLOv11
- Advanced re-identification algorithms
- Robust tracking across occlusions


## 🛠️ Setup

### 1. Clone Repository

```bash
git clone https://github.com/realyashagarwal/TrackNetX
cd TrackNetX
```

### 2. Create Environment

```bash
# Using conda (recommended)
conda create -n player-tracking python=3.9 -y
conda activate player-tracking

# Or using venv
python -m venv player-tracking-env
source player-tracking-env/bin/activate  # On Windows: player-tracking-env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Project

```bash
python src/main.py --setup
```

### 5. Add Required Files

- Download the YOLOv11 model and place it in `data/models/player_detection_model.pt`
- Add your input video to `data/videos/15sec_input_720p.mp4`

### 6. Test Detection

```bash
python src/main.py --test
```

## 📁 Project Structure

```
TrackNetX/
├── data/
│   ├── videos/          # Input videos
│   └── models/          # YOLO model files
├── src/
│   ├── detection.py     # Player detection module
│   ├── tracking.py      # Player tracking module
│   ├── main.py          # Main application
│   └── utils.py         # Utility functions
├── outputs/
│   ├── frames/          # Sample and processed frames
│   └── results/         # Final output videos
├── tests/               # Test files
├── requirements.txt     # Python dependencies
└── README.md           # This file
```


### Basic Detection Test

```bash
python src/main.py --test
```

### Process Full Video

```bash
python src/main.py --video data/videos/15sec_input_720p.mp4
```


## 🤖 Model Information

- **Model**: Fine-tuned YOLOv11 for player and ball detection
- **Classes**: Players and ball
- **Input**: 720p video (15 seconds)
- **Performance**: Real-time processing on modern GPUs

## 📋 Requirements

- Python 3.9+
- OpenCV
- Ultralytics YOLOv11
- PyTorch
- NumPy
- GPU recommended for real-time processing

## 🚀 Getting Started Quickly

### Step 1: Quick Setup (5 minutes)

```bash
git clone https://github.com/realyashagarwal/TrackNetX
cd TrackNetX
conda create -n player-tracking python=3.9 -y
conda activate player-tracking
pip install -r requirements.txt
python src/main.py --setup
```

### Step 2: Download Model and Test (10 minutes)

1. **Download the model** from the provided Google Drive link
2. **Rename it** to `player_detection_model.pt`
3. **Place it** in `data/models/` folder
4. **Add your video** to `data/videos/` folder
5. **Test the setup:**

```bash
python src/main.py --test
```

### Step 3: Commit and Push Initial Setup (5 minutes)

```bash
# Add all files to git
git add .

# Commit with a descriptive message
git commit -m "Initial project setup: dependencies, structure, and basic detection module"

# Push to GitHub
git push origin main
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Happy Tracking! 🎯**