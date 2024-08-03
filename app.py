import streamlit as st

st.set_page_config(page_title="Automated Online Proctoring", page_icon=":tada:", layout="wide")

def check_imports():
    try:
        import cv2
        st.success("cv2 imported successfully.")
    except ImportError as e:
        st.error(f"Error importing cv2: {e}")
        return False

    try:
        import mediapipe as mp
        st.success("mediapipe imported successfully.")
    except ImportError as e:
        st.error(f"Error importing mediapipe: {e}")
        return False

    try:
        import sounddevice as sd
        st.success("sounddevice imported successfully.")
    except ImportError as e:
        st.error(f"Error importing sounddevice: {e}")
        return False

    return True

st.title("Dependency Check")

if check_imports():
    st.success("All dependencies imported successfully.")
else:
    st.error("Failed to import one or more dependencies. Please check the logs.")