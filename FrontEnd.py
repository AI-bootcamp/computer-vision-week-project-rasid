import streamlit as st
import cv2
import numpy as np
import time
import pygame
import os
import tempfile
import threading
from BackEnd import SpeedDetector

# Initialize pygame for audio
pygame.mixer.init()

# Audio files mapping
SPEED_AUDIO_MAPPING = {
    '030': '30.mp3',
    '040': '40.mp3',
    '050': '50.mp3',
    '060': '60.mp3',
    '070': '70.mp3',
    '080': '80.mp3',
    '090': '90.mp3',
    '100': '100.mp3',
    '110': '110.mp3',
    '120': '120.mp3',
    '140': '140.mp3'
}

def play_speed_audio(speed):
    """Play audio notification for speed in a separate thread"""
    def _play_audio(speed):
        try:
            # Get the audio filename from mapping (without "Videos/" prefix)
            audio_file = SPEED_AUDIO_MAPPING.get(speed)
            if audio_file:
                # Construct full path if needed (assuming files are in Videos folder)
                audio_path = os.path.join("Videos", audio_file)
                if os.path.exists(audio_path):
                    pygame.mixer.music.load(audio_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                else:
                    print(f"Audio file not found: {audio_path}")
            else:
                print(f"No audio mapping for speed: {speed}")
        except Exception as e:
            print(f"Audio error: {str(e)}")
            # st.error() should only be used in main thread for Streamlit

    # Start audio in a new thread
    audio_thread = threading.Thread(target=_play_audio, args=(speed,))
    audio_thread.daemon = True
    audio_thread.start()

def format_timestamp(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def main():
    st.set_page_config(page_title="Speed Detection System", layout="wide")
    st.title("🚗 Rasid for Speed Sign Detection")
    
    # Initialize session state
    if 'detector' not in st.session_state:
        try:
            st.session_state.detector = SpeedDetector()
            st.session_state.current_speed = None
            st.session_state.processing = False
            st.session_state.video_path = None
            st.session_state.speed_history = []
            st.session_state.speed_log = []  # For storing timestamped speed changes
            st.session_state.frame_placeholder = None
        except Exception as e:
            st.error(f"Failed to initialize detector: {str(e)}")
            return
    
    # File uploader
    uploaded_file = st.file_uploader("Upload driving video", type=["mp4"])
    
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
            st.session_state.speed_log = []  # Reset log for new video
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.session_state.video_path = tmp_file.name
            
            try:
                # Define callback function
                def callback(speed, timestamp, bbox, frame):
                    # Extract real speed (remove leading zero if needed)
                    if speed in ['030','040','050','060','070','080','090']:
                        real_speed = speed[1:]
                    real_speed = speed
                    
                    # Store speed change with timestamp
                    st.session_state.speed_log.append({
                        'speed': real_speed,
                        'timestamp': timestamp,
                        'formatted_time': format_timestamp(timestamp)
                    })
                    
                    # Play audio
                    play_speed_audio(speed)
                    
                    # Update UI
                    speed_placeholder.success(f"Current Speed: {real_speed} km/h")
                    info_placeholder.info(f"Speed changed to {real_speed} km/h at {format_timestamp(timestamp)}")
                    
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
                
                # Show detailed speed log after processing completes
                if st.session_state.speed_log:
                    with st.expander("📝 Detailed Speed Log", expanded=True):
                        st.subheader("Speed Change History")
                        for entry in st.session_state.speed_log:
                            st.write(f"⏱️ {entry['formatted_time']} - 🚦 Speed: {entry['speed']} km/h")
                        
                        # Summary statistics
                        if st.session_state.speed_history:
                            st.subheader("📊 Speed Statistics")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Max Speed", f"{max(st.session_state.speed_history)} km/h")
                            with col2:
                                st.metric("Min Speed", f"{min(st.session_state.speed_history)} km/h")
                            with col3:
                                st.metric("Changes Detected", len(st.session_state.speed_history))
                
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

if __name__ == "__main__":
    main()