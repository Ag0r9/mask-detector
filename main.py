import streamlit as st
from streamlit_webrtc import webrtc_streamer

webrtc_streamer(key="camera", media_stream_constraints={"video": True, "audio": False})