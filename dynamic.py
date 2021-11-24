import streamlit as st
from streamlit_webrtc import webrtc_streamer


def detection_from_camera():
    st.title('WORK IN PROGRESS')
    webrtc_streamer(key="camera", media_stream_constraints={"video": True, "audio": False})