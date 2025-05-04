# front_end.py

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

def play_speed_audio(label):
    """Play audio for a speed label like 'speed_050'"""
    try:
        speed_number = str(int(label.split("_")[-1]))  # e.g., 'speed_050' -> '50'
        audio_file = f"{speed_number}.mp3"
        
        if os.path.exists(audio_file):
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        else:
            st.warning(f"Audio file {audio_file} not found.")
    except Exception as e:
        st.error(f"Audio error: {str(e)}")

def main():
    st.set_page_config(page_title="Speed Detection System", layout="wide")
    st.title("🚗 Speed Sign Detection System")

    if 'detector' not in st.session_state:
        try:
            st.session_state.detector = SpeedDetector()
            st.session_state.current_label = None
            st.session_state.processing = False
            st.session_state.video_path = None
            st.session_state.speed_history = []
            st.session_state.frame_placeholder = None
        except Exception as e:
            st.error(f"Failed to initialize detector: {str(e)}")
            return

    uploaded_file = st.file_uploader("Upload driving video", type=["mp4", "avi", "mov"])

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

    if uploaded_file is not None and not st.session_state.processing:
        if st.button("Start Processing"):
            st.session_state.processing = True

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.session_state.video_path = tmp_file.name

            try:
                def callback(label, timestamp, bbox, frame):
                    speed_number = str(int(label.split("_")[-1]))
                    
                    # Display info
                    speed_placeholder.success(f"Current Speed: {speed_number} km/h")
                    info_placeholder.info(f"Speed sign: {label} at {timestamp:.1f}s")
                    
                    # Draw bounding box
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Play audio after drawing
                    play_speed_audio(label)
                    
                    return frame

                cap = cv2.VideoCapture(st.session_state.video_path)
                if not cap.isOpened():
                    raise ValueError("Could not open video file")

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                for i in range(total_frames):
                    if not st.session_state.processing:
                        break

                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    if i % int(fps) == 0:
                        label, confidence, bbox = st.session_state.detector.process_frame(frame_rgb)
                        if label:
                            if (st.session_state.current_label is None or 
                                label != st.session_state.current_label):
                                st.session_state.current_label = label
                                st.session_state.speed_history.append(int(label.split("_")[-1]))
                                frame_rgb = callback(label, i/fps, bbox, frame_rgb)

                    st.session_state.frame_placeholder.image(frame_rgb, channels="RGB")

                    progress = (i + 1) / total_frames
                    progress_placeholder.progress(progress)

                    time.sleep(1/fps)

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

    if st.session_state.processing:
        if st.button("Stop Processing"):
            st.session_state.processing = False

if __name__ == "__main__":
    main()
