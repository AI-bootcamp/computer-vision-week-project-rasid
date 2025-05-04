# Rasid for Speed Sign Detection System 🚗📊


A real-time speed limit sign detection system using deep learning and computer vision.


##  Features
- Real-time speed limit sign detection in video streams
- Audio alerts for detected speed limits
- Detailed speed change logging with timestamps
- Performance statistics (max/min speeds, detection counts)
- Clean and intuitive Streamlit interface


computer-vision-week-project-rasid/
├── FrontEnd.py # Streamlit UI (Main Interface)
├── BackEnd.py # Detection Engine (YOLO Model Handler)
├── classification.ipynb # Model Training Notebook
└── Model/ # Pretrained Models



##  Performance Metrics
Our model achieves excellent performance across all speed limit classes:

| Class             | Precision | Recall | mAP50 | mAP50-95 |
|-------------------|-----------|--------|-------|----------|
| **All Classes**   | 0.948     | 0.920  | 0.965 | 0.840    |
| Speed Limit 20    | 0.987     | 0.982  | 0.987 | 0.884    |
| Speed Limit 30    | 0.962     | 0.959  | 0.989 | 0.926    |
| ...               | ...       | ...    | ...   | ...      |
| Speed Limit 120   | 0.983     | 0.982  | 0.994 | 0.919    |

**Inference Speed**: 0.5ms preprocess, 3.9ms inference per image

## Model Architecture
We use a YOLOv8-based architecture with the following key components:

- **Backbone**: Custom CNN with C2f and SPPF modules
- **Neck**: Feature pyramid network with upsampling
- **Head**: Detection head with 15 output layers
- **Parameters**: 3,013,773 total parameters
- **GFLOPs**: 8.2 GFLOPs

# Sample Architecture Snippet
0: Conv(3,16,3,2)                 # Input layer
1: Conv(16,32,3,2)                # Downsample
2: C2f(32,32,1,True)              # Feature extraction
...
21: C2f(384,256,1)                # Final features
22: Detect([15,[64,128,256]])      # Detection head









