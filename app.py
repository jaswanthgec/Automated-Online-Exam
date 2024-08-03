import streamlit as st
import cv2
import threading
import mediapipe as mp
import numpy as np
import time
from scipy.ndimage import uniform_filter1d
import os

# Streamlit Page Configuration
st.set_page_config(page_title="Automated Online Proctoring", page_icon=":tada:", layout="wide")

# Check if the app is running on Streamlit Cloud
IS_STREAMLIT_CLOUD = 'STREAMLIT_SERVER' in os.environ

# Global Variables
X_AXIS_CHEAT = 0
Y_AXIS_CHEAT = 0
AUDIO_CHEAT = 0
SOUND_AMPLITUDE = 0
STOP_FLAG = False
PEAK_COUNT = 0
PEAK_THRESHOLD = 1.6
PEAK_MAX_REACHED_COUNT = 5
MULTI_FACE_COUNT = 0
NO_FACE_COUNT = 0

MULTI_FACE_THRESHOLD = 2
NO_FACE_THRESHOLD = 30

LIP_THRESHOLD = 0.005  # Define a threshold for lip distance
LIP_ALERT_COUNT = 30  # Number of times the alert should trigger

# Variables for tracking lip movement
LIP_MOVEMENT_DURATION = 3  # Duration in seconds for continuous detection
lip_alert_reset_timer = None  # Timer to reset if lip movement is not continuous

# Initialize session state variables
if 'capturing' not in st.session_state:
    st.session_state.capturing = False
if 'lip_alert_count' not in st.session_state:
    st.session_state.lip_alert_count = 0
if 'lip_alert_start_time' not in st.session_state:
    st.session_state.lip_alert_start_time = None

# Placeholder for alerts and warnings
alert_placeholder = st.empty()

# Helper Function to calculate head pose
def calculate_head_pose(face_2d, face_3d, img_w, img_h, image, nose_2d, nose_3d):
    global X_AXIS_CHEAT, Y_AXIS_CHEAT

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                           [0, focal_length, img_w / 2],
                           [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    x = angles[0] * 360
    y = angles[1] * 360

    X_AXIS_CHEAT = 1 if y < -10 or y > 10 else 0
    Y_AXIS_CHEAT = 1 if x < -5 else 0

    nose_3d_projection, _ = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
    p1 = (int(nose_2d[0]), int(nose_2d[1]))
    p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
    cv2.line(image, p1, p2, (255, 0, 0), 2)
    cv2.putText(image, f"X: {int(x)}, Y: {int(y)}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Function to get lip landmarks
def get_lip_distance(lip_landmarks):
    if len(lip_landmarks) < 2:
        return 0  # Not enough landmarks to calculate distance
    # Calculate the distance between the upper and lower lip points
    upper_lip = np.array([lip_landmarks[13].x, lip_landmarks[13].y])
    lower_lip = np.array([lip_landmarks[14].x, lip_landmarks[14].y])
    distance = np.linalg.norm(upper_lip - lower_lip)
    return distance

# Generator Function to get head pose and lip movement
def get_head_pose_and_lip_movement(video_source):
    global X_AXIS_CHEAT, Y_AXIS_CHEAT, MULTI_FACE_COUNT, NO_FACE_COUNT, STOP_FLAG
    global LIP_ALERT_COUNT, LIP_THRESHOLD
    global lip_alert_reset_timer

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_source)
    mp_drawing = mp.solutions.drawing_utils

    if not cap.isOpened():
        alert_placeholder.error("Error: Could not open video device.")
        return

    lip_alert_start_time = None
    lip_alert_count = 0

    try:
        while not STOP_FLAG:
            success, image = cap.read()
            if not success:
                break

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = face_mesh.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            img_h, img_w, _ = image.shape
            face_3d, face_2d = [], []
            face_ids = [33, 263, 1, 61, 291, 199]

            lip_landmarks = []
            if results.multi_face_landmarks:
                if len(results.multi_face_landmarks) > 1:
                    MULTI_FACE_COUNT += 1
                    NO_FACE_COUNT = 0
                    alert_placeholder.warning("Multiple faces detected!")
                else:
                    MULTI_FACE_COUNT = 0

                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None)

                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx in face_ids:
                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                            x, y = int(lm.x * img_w), int(lm.y * img_h)
                            face_2d.append([x, y])
                            face_3d.append([x, y, lm.z])

                        # Collect lip landmarks
                        if 48 <= idx < 68:  # Assuming 48-67 are lip landmarks
                            lip_landmarks.append(lm)

                    if face_2d and face_3d:
                        calculate_head_pose(face_2d, face_3d, img_w, img_h, image, nose_2d, nose_3d)

                    if lip_landmarks:
                        lip_distance = get_lip_distance(lip_landmarks)
                        cv2.putText(image, f"Lip Distance: {lip_distance:.4f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        if LIP_THRESHOLD < lip_distance < 0.0055:
                            if lip_alert_start_time is None:
                                lip_alert_start_time = time.time()
                            elif time.time() - lip_alert_start_time >= LIP_MOVEMENT_DURATION:
                                lip_alert_count += 1
                                alert_placeholder.warning(f"Lips movement detected! Alert number {lip_alert_count}.")

                                # Reset the start time and timer after alert
                                lip_alert_start_time = None
                                lip_alert_reset_timer = time.time()

                                if lip_alert_count >= LIP_ALERT_COUNT:
                                    alert_placeholder.error("Lips movement detected too many times. The application will close.")
                                    st.session_state.capturing = False
                                    STOP_FLAG = True
                                    break
                        else:
                            # Reset start time if lip movement is not continuous
                            lip_alert_start_time = None

                        # Reset the count if movement is not continuous for 3 seconds
                        if lip_alert_reset_timer and time.time() - lip_alert_reset_timer > LIP_MOVEMENT_DURATION:
                            lip_alert_reset_timer = None
                            lip_alert_count = 0
            else:
                NO_FACE_COUNT += 1
                MULTI_FACE_COUNT = 0
                alert_placeholder.warning("No face detected!")

            if MULTI_FACE_COUNT >= MULTI_FACE_THRESHOLD:
                alert_placeholder.error("Multiple faces detected too many times. The application will close.")
                st.session_state.capturing = False
                STOP_FLAG = True

            if NO_FACE_COUNT >= NO_FACE_THRESHOLD:
                alert_placeholder.error("No face detected too many times. The application will close.")
                st.session_state.capturing = False
                STOP_FLAG = True

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            yield image_rgb, X_AXIS_CHEAT,Y_AXIS_CHEAT
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Function to get audio analysis
def get_audio_analysis():
    import sounddevice as sd

    AMPLITUDE_LIST = []
    SOUND_AMPLITUDE_THRESHOLD = 0.5  # Adjust this threshold according to your needs
    SUS_COUNT = 0
    FRAMES_COUNT = 10  # Number of frames to consider for averaging
    count = 0

    def print_sound(indata, frames, time, status):
        nonlocal count, SUS_COUNT, AMPLITUDE_LIST

        global SOUND_AMPLITUDE
        global AUDIO_CHEAT

        if status:
            print(status, flush=True)
        amplitude = np.linalg.norm(indata) / np.sqrt(len(indata))
        AMPLITUDE_LIST.append(amplitude)
        if len(AMPLITUDE_LIST) > FRAMES_COUNT:
            AMPLITUDE_LIST.pop(0)
        if count == FRAMES_COUNT:
            avg_amp = sum(AMPLITUDE_LIST) / FRAMES_COUNT
            SOUND_AMPLITUDE = avg_amp
            if SUS_COUNT >= 2:
                AUDIO_CHEAT = 1
                SUS_COUNT = 0
            if avg_amp > SOUND_AMPLITUDE_THRESHOLD:
                SUS_COUNT += 1
            else:
                SUS_COUNT = 0
                AUDIO_CHEAT = 0
            count = 0

    with sd.Stream(callback=print_sound):
        while not STOP_FLAG:
            sd.sleep(1000)

# Function to update cheat probability
def update_cheat_probability(current_prob, head_x_cheat, head_y_cheat, audio_cheat):
    max_increase = 0.01
    max_decrease = 0.005
    cheat_detected = head_x_cheat or head_y_cheat or audio_cheat

    if cheat_detected:
        current_prob = min(current_prob + max_increase, PEAK_THRESHOLD)
    else:
        current_prob = max(current_prob - max_decrease, 0.0)

    return current_prob

# Function to check peak and warn user
def check_peak_and_warn(current_prob):
    global PEAK_COUNT

    if current_prob >= PEAK_THRESHOLD:
        PEAK_COUNT += 1
        alert_placeholder.warning("Don't cheat! This is warning number {}.".format(PEAK_COUNT))
        current_prob = 0.0  # Reset the cheat probability

        if PEAK_COUNT >= PEAK_MAX_REACHED_COUNT:
            alert_placeholder.error("Cheating detected too many times. The application will close.")
            st.session_state.capturing = False
            st.stop()
    return current_prob

# Streamlit Application
st.markdown("<h1 style='text-align: center;'>Automatic Online Proctoring System</h1>", unsafe_allow_html=True)

c1, c2, c3 = st.columns([8, 5, 5])

# Create buttons to start and stop capturing
if c2.button("Start the Exam"):
    st.session_state.capturing = True
    STOP_FLAG = False

if c2.button("Submit Exam"):
    st.session_state.capturing = False
    STOP_FLAG = True

col1, col2 = st.columns(2)
video_placeholder = col1.empty()
cheat_chart_placeholder = col2.empty()
cheat_probability_data = []
current_cheat_probability = 0.0

if st.session_state.capturing:
    if not IS_STREAMLIT_CLOUD:
        audio_thread = threading.Thread(target=get_audio_analysis, daemon=True)
        audio_thread.start()

    video_source = 0 if not IS_STREAMLIT_CLOUD else st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

    if video_source:
        frame_generator = get_head_pose_and_lip_movement(video_source)

        try:
            while st.session_state.capturing:
                try:
                    image, x_cheat, y_cheat = next(frame_generator)
                    video_placeholder.image(image, channels="RGB")
                    current_cheat_probability = update_cheat_probability(current_cheat_probability, x_cheat, y_cheat, AUDIO_CHEAT)
                    current_cheat_probability = check_peak_and_warn(current_cheat_probability)
                    cheat_probability_data.append(current_cheat_probability)

                    if len(cheat_probability_data) > 100:
                        cheat_probability_data.pop(0)

                    smoothed_data = uniform_filter1d(cheat_probability_data, size=5)
                    cheat_chart_placeholder.line_chart(smoothed_data)
                except StopIteration:
                    break
        finally:
            STOP_FLAG = True
            if not IS_STREAMLIT_CLOUD:
                audio_thread.join()
else:
    video_placeholder.empty()
    cheat_chart_placeholder.empty()
    alert_placeholder.empty()  # Clear the alert placeholder when not capturing