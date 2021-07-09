import streamlit as st

from PIL import Image

st.title('Vision: Brain Tumor Detector')
st.text('Upload a brain MRI image to classify.')

uploaded_file = st.file_uploader('Choose an image...', type='jpg')

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write('\nClassifying...')