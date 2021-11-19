import streamlit as st
from PIL import Image
from backend import detect

def photo_uploader():
    st.title('Upload a photo')

    uploaded_file = st.file_uploader('Choose an image...', type='jpg')
    if uploaded_file is not None:
        st.write('')
        st.write('Detecting...')
        print(uploaded_file)
        image = Image.open(uploaded_file)
        image_after_detection = detect(image)
        st.image(image_after_detection, caption='Uploaded Image.', use_column_width=True)
        # label = predict(uploaded_file)
        # st.write('%s (%.2f%%)' % (label[1], label[2] * 100))


