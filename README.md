# 🚦 Rasid: Traffic Signs Detection Using YOLO

Rasid is a computer-vision system that detects and classifies traffic signs in video streams using a YOLO-based pipeline. It supports real-time processing of dashcam footage or live video, generates bounding boxes.

---

## 📋 Table of Contents
1. [About the Project](#-about-the-project)  
2. [Features](#-features)  
3. [How It Works](#-how-it-works)  
4. [Project Structure](#-project-structure)  
5. [Getting Started](#-getting-started)  
6. [Technologies Used](#-technologies-used)  
7. [Example Output](#-example-output)  
8. [Team Members](#-team-members)  

---

## 📖 About the Project

Rasid processes video input (e.g., `Driving.mp4` or a webcam stream), detects traffic signs (speed limits, stop, yield, etc.) using a YOLOv5s model, and outputs annotated video frames. Optionally, it plays corresponding audio cues from the `Model/` folder (`30.mp3`, `50.mp3`, … `140.mp3`) when a sign is recognized.

---

## ✨ Features

- **Real-Time Detection**  
  Processes video at near-real time speeds using a lightweight YOLOv5s model (`yolov5s.pt`).  
- **Audio Alerts**  
  Plays pre-recorded MP3 alerts (e.g., “Speed 30”) whenever a sign is detected.  
- **Custom Dataset Tools**  
  - `combine_dataset.py`: Merge multiple annotated datasets into one.  
  - `label_extraction.py`: Extract and convert label files to YOLO format.  
  - `remove_bounding_box.py`: Clean up erroneous bounding-box annotations.  
- **End-to-End Pipeline**  
  - **BackEnd.py**: Video frame capture → inference → annotation → audio output.  
  - **FrontEnd.py**: Simple GUI for selecting input source and toggling alerts.  

---

## 🛠 How It Works

1. **Initialization**  
   - Load `yolov5s.pt` model from the `Model/` directory.  
   - Preload audio files (`30.mp3` … `140.mp3`) for each sign class.  
2. **Frame Processing Loop**  
   - Read a frame from video or webcam.  
   - Run YOLO inference to detect signs and class probabilities.  
   - Draw bounding boxes and class labels on the frame.  
   - If audio alerts enabled, play the corresponding MP3 for each detected sign.  
3. **Output**  
   - Display annotated video in a GUI window (via `BackEnd.py`).  
   - Save annotated video to disk (optional).  

---

## 📂 Project Structure

'
```
Rasid/
├── .github/ # GitHub workflows
├── Model/ # Weights & audio alerts
│ ├── yolov5s.pt # Pre-trained YOLOv5s model
│ ├── 30.mp3 # Audio for “Speed 30”
│ ├── 40.mp3 # Audio for “Speed 40”
│ └── … # up to 140.mp3
│
├── BackEnd.py # Main detection & annotation script
├── FrontEnd.py # GUI launcher & settings
│
├── combine_dataset.py # Merge multiple datasets
├── label_extraction.py # Convert labels to YOLO format
├── remove_bounding_box.py # Clean up annotation files
│
├── classfaction.ipynb # Notebook: training custom classifier
├── Driving.mp4 # Sample dashcam video
│
└── README.md # This documentation
```


---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+  
- `pip`  

### Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/AI-bootcamp/computer-vision-week-project-rasid.git
   cd computer-vision-week-project-rasid
   ```

2. **Create a venv & install dependencies**

```bash
python -m venv env
source env/bin/activate       # Linux/macOS
.\env\Scripts\activate        # Windows
pip install -r requirements.txt
```

3. Run the detection pipeline
- Backend only (CLI)
```bash
python BackEnd.py --input Driving.mp4 --output out.mp4 --audio-alerts
```

- With GUI
```bash
python FrontEnd.py
```

# 🛠 Technologies Used
- **FastAPI**: Backend API for handling requests.
- **Streamlit**: Frontend for user interaction.- PyTorch & YOLOv5: Model inference engine
- **OpenCV**: Video I/O and frame annotation
- **Jupyter Notebook (colab or Kaggle)**: Data preparation & custom training

# 📸 Example Output
- Figure: YOLO detecting a “Speed 80” sign and overlaying the label + playing 80.mp3.

# 👩‍💻 Team Members
- Abdulaziz Alhaizan
- Feras Alswaid
- Abdulaziz Alkharjy
- Waref Alyousef
- Dania Emad


