import streamlit as st
import cv2
import threading
import mediapipe as mp
import sounddevice as sd
import numpy as np
from scipy.ndimage import uniform_filter1d

# Streamlit Page Configuration
st.set_page_config(page_title="Automated Online Proctoring", page_icon=":tada:", layout="wide")

# Global Variables
X_AXIS_CHEAT = 0
Y_AXIS_CHEAT = 0
AUDIO_CHEAT = 0
SOUND_AMPLITUDE = 0
STOP_FLAG = False
PEAK_COUNT = 0
PEAK_THRESHOLD = 2
PEAK_MAX_REACHED_COUNT = 5
MULTI_FACE_COUNT = 0
NO_FACE_COUNT = 0

MULTI_FACE_THRESHOLD = 3
NO_FACE_THRESHOLD = 20

# Initialize session state variables
if 'capturing' not in st.session_state:
    st.session_state.capturing = False

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

# Generator Function to get head pose
def get_head_pose():
    global X_AXIS_CHEAT, Y_AXIS_CHEAT, MULTI_FACE_COUNT, NO_FACE_COUNT, STOP_FLAG
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    mp_drawing = mp.solutions.drawing_utils

    if not cap.isOpened():
        st.error("Error: Could not open video device.")
        return

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

            if results.multi_face_landmarks:
                if len(results.multi_face_landmarks) > 1:
                    MULTI_FACE_COUNT += 1
                    NO_FACE_COUNT = 0
                    st.warning("Multiple faces detected!")
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

                    if face_2d and face_3d:
                        calculate_head_pose(face_2d, face_3d, img_w, img_h, image, nose_2d, nose_3d)
            else:
                NO_FACE_COUNT += 1
                MULTI_FACE_COUNT = 0
                st.warning("No face detected!")

            if MULTI_FACE_COUNT >= MULTI_FACE_THRESHOLD:
                st.error("Multiple faces detected too many times. The application will close.")
                st.session_state.capturing = False
                STOP_FLAG = True

            if NO_FACE_COUNT >= NO_FACE_THRESHOLD:
                st.error("No face detected too many times. The application will close.")
                st.session_state.capturing = False
                STOP_FLAG = True

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            yield image_rgb, X_AXIS_CHEAT, Y_AXIS_CHEAT
    finally:
        cap.release()

# Function to analyze audio
def get_audio_analysis():
    global AUDIO_CHEAT, SOUND_AMPLITUDE, STOP_FLAG

    CALLBACKS_PER_SECOND = 38
    SUS_FINDING_FREQUENCY = 2
    SOUND_AMPLITUDE_THRESHOLD = 20
    FRAMES_COUNT = int(CALLBACKS_PER_SECOND / SUS_FINDING_FREQUENCY)
    AMPLITUDE_LIST = list([0] * FRAMES_COUNT)
    SUS_COUNT = 0
    count = 0

    def print_sound(indata, outdata, frames, time, status):
        nonlocal count, SUS_COUNT
        global SOUND_AMPLITUDE, AUDIO_CHEAT

        vnorm = int(np.linalg.norm(indata) * 10)
        AMPLITUDE_LIST.append(vnorm)
        count += 1
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
        st.warning("Don't cheat! This is warning number {}.".format(PEAK_COUNT))
        current_prob = 0.0  # Reset the cheat probability

        if PEAK_COUNT >= PEAK_MAX_REACHED_COUNT:
            st.error("Cheating detected too many times. The application will close.")
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
    audio_thread = threading.Thread(target=get_audio_analysis, daemon=True)
    audio_thread.start()

    frame_generator = get_head_pose()

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
        audio_thread.join()
else:
    video_placeholder.empty()
    cheat_chart_placeholder.empty()
