# Copyright 2021 Olman Ureña
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st

from classifier import ImageClassifier
from PIL import Image

# instance of the image classifier
image_classifier = ImageClassifier()

# display titles
st.title('Vision: Brain Tumor Detector')
st.header('Upload a brain MRI image to classify.')

# display the file uploader
uploaded_file = st.file_uploader('Choose an image...', type='jpg')

if uploaded_file:

    # get the uploaded image 
    image = Image.open(uploaded_file)

    # display the uploaded image
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write('\nClassifying...')

    # preprocess the image file to load in the model for classification
    image = image_classifier.preprocess_image(image)

    # classify the uploaded image
    inference = image_classifier.predict(image)

    # print the classification label
    if inference == 0:
        st.write('Brain Tumor')
    else:
        st.write('No Brain Tumor')
