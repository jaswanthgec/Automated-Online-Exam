import streamlit as st
import cv2
import threading
import mediapipe as mp
import numpy as np
import os
import time
import logging

logger = logging.getLogger(__name__)

# Streamlit Page Configuration
st.set_page_config(page_title="Automated Online Proctoring", page_icon=":tada:", layout="wide")

# Check if the app is running on Streamlit Cloud
IS_STREAMLIT_CLOUD = 'STREAMLIT_SERVER' in os.environ

# Global Variables
STOP_FLAG = False

# Initialize session state variables
if 'capturing' not in st.session_state:
    st.session_state.capturing = False

# Placeholder for alerts and warnings
alert_placeholder = st.empty()

# Function to capture video
def capture_video():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        alert_placeholder.error("Error: Could not open video device.")
        return

    try:
        while st.session_state.capturing:
            success, image = cap.read()
            if not success:
                break

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
            st.image(image, channels="BGR")
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Streamlit UI
st.title("Automated Online Proctoring System")
start_button = st.button("Start the Exam")
stop_button = st.button("Stop the Exam")

if start_button:
    st.session_state.capturing = True
    video_thread = threading.Thread(target=capture_video)
    video_thread.start()

if stop_button:
    st.session_state.capturing = False