from static import photo_uploader
from dynamic import detection_from_camera

import streamlit as st


choice = st.sidebar.radio('Select detection mode', ['Static', 'Dynamic'])

if choice == 'Static':
    photo_uploader()
elif choice == 'Dynamic':
    detection_from_camera()

