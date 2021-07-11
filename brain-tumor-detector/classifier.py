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

import keras
import numpy as np

from PIL import Image, ImageOps

class ImageClassifier:
    '''
    TODO
    '''

    def __init__(self, model='./model.h5', image_size=(224, 224)):
        '''
        TODO
        '''
        self.model = keras.models.load_model(model)
        self.image_size = image_size

    def process_image(self, image):
        '''
        TODO
        '''
        image = image.convert('RGB')
        image = ImageOps.fit(image, self.image_size, Image.ANTIALIAS)
        image = np.asarray(image)
        image = (image.astype(np.float32) / 127.0) - 1
        image = np.expand_dims(image, axis=0)
        return image

    def inference(self, image):
        '''
        TODO
        '''
        image = self.process_image(image)
        inference = self.model.predict(image)
        return np.argmax(inference)
