import streamlit as st
import cv2
import numpy as np
import time
import pygame
import os
import tempfile
from BackEnd import SpeedDetector

# Initialize pygame for audio
pygame.mixer.init()

# Audio files mapping
SPEED_AUDIO_MAPPING = {
    '10': '10.mp3',
    '20': '20.mp3',
    '30': '30.mp3',
    '40': '40.mp3',
    '50': '50.mp3',
    '60': '60.mp3',
    '70': '70.mp3',
    '80': '80.mp3',
    '90': '90.mp3',
    '100': '100.mp3',
    '110': '110.mp3',
    '120': '120.mp3',
    '140': '140.mp3'
}

def play_speed_audio(speed):
    """Play audio notification for speed"""
    try:
        audio_file = SPEED_AUDIO_MAPPING.get(speed)
        if audio_file and os.path.exists(audio_file):
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
    except Exception as e:
        st.error(f"Audio error: {str(e)}")

def main():
    st.set_page_config(page_title="Speed Detection System", layout="wide")
    st.title("🚗 Speed Sign Detection System")
    
    # Initialize session state
    if 'detector' not in st.session_state:
        try:
            st.session_state.detector = SpeedDetector()
            st.session_state.current_speed = None
            st.session_state.processing = False
            st.session_state.video_path = None
            st.session_state.speed_history = []
            st.session_state.frame_placeholder = None
        except Exception as e:
            st.error(f"Failed to initialize detector: {str(e)}")
            return
    
    # File uploader
    uploaded_file = st.file_uploader("Upload driving video", type=["mp4", "avi", "mov"])
    
    # Create layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Video Feed")
        frame_placeholder = st.empty()
        st.session_state.frame_placeholder = frame_placeholder
    
    with col2:
        st.subheader("Detection Info")
        info_placeholder = st.empty()
        speed_placeholder = st.empty()
        progress_placeholder = st.empty()
    
    # Process video when uploaded
    if uploaded_file is not None and not st.session_state.processing:
        if st.button("Start Processing"):
            st.session_state.processing = True
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.session_state.video_path = tmp_file.name
            
            try:
                # Define callback function
                def callback(speed, timestamp, bbox, frame):
                    # Play audio
                    play_speed_audio(speed)
                    
                    # Update UI
                    speed_placeholder.success(f"Current Speed: {speed} km/h")
                    info_placeholder.info(f"Speed changed to {speed} km/h at {timestamp:.1f}s")
                    
                    # Draw bounding box if detection exists
                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    return frame
                
                # Open video file
                cap = cv2.VideoCapture(st.session_state.video_path)
                if not cap.isOpened():
                    raise ValueError("Could not open video file")
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Process video frames
                for i in range(total_frames):
                    if not st.session_state.processing:
                        break
                        
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Convert frame to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process frame (every second)
                    if i % int(fps) == 0:
                        speed, confidence, bbox = st.session_state.detector.process_frame(frame_rgb)
                        if speed is not None:
                            if (st.session_state.current_speed is None or 
                                abs(int(speed) - int(st.session_state.current_speed)) >= 5):
                                st.session_state.current_speed = speed
                                st.session_state.speed_history.append(int(speed))
                                frame_rgb = callback(speed, i/fps, bbox, frame_rgb)
                    
                    # Display frame
                    st.session_state.frame_placeholder.image(frame_rgb, channels="RGB")
                    
                    # Update progress
                    progress = (i + 1) / total_frames
                    progress_placeholder.progress(progress)
                    
                    # Control playback speed
                    time.sleep(1/fps)
                
                # Show results
                if st.session_state.speed_history:
                    st.subheader("Speed History")
                    st.line_chart({"Speed (km/h)": st.session_state.speed_history})
                
            except Exception as e:
                st.error(f"Processing error: {str(e)}")
            finally:
                cap.release()
                if st.session_state.video_path and os.path.exists(st.session_state.video_path):
                    os.unlink(st.session_state.video_path)
                st.session_state.processing = False
    
    # Stop button
    if st.session_state.processing:
        if st.button("Stop Processing"):
            st.session_state.processing = False
            # No need for rerun - the loop will exit naturally

if __name__ == "__main__":
    main()