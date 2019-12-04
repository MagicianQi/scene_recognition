# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import json
import base64
import requests
import numpy as np

from cv2 import resize
from PIL import Image
from settings import *


class PlacesModel(object):

    def __init__(self,
                 place_predict_url=PLACE_PREDICT_URL,
                 class_name_file=CLASS_NAME_FILE):
        self.place_predict_url = place_predict_url
        self.class_names = []
        with open(class_name_file) as class_file:
            for line in class_file:
                self.class_names.append(line.strip().split(' ')[0][3:])

    def predict_batch(self, image_batch):
        post_json = {"instances": []}
        for image in image_batch:
            image = np.array(image, dtype=np.uint8)
            image = resize(image, (224, 224))
            image = Image.fromarray(image, mode="RGB")
            image_pil = io.BytesIO()
            image.save(image_pil, format='JPEG')
            img_b64 = base64.b64encode(image_pil.getvalue()).decode('utf-8')
            post_json["instances"].append({"b64": img_b64})
        response = requests.post(self.place_predict_url, data=json.dumps(post_json))
        response.raise_for_status()
        prediction = response.json()["predictions"]
        return prediction

    def predict(self, image):
        post_json = {"instances": []}
        image = np.array(image, dtype=np.uint8)
        image = resize(image, (224, 224))
        image = Image.fromarray(image, mode="RGB")
        image_pil = io.BytesIO()
        image.save(image_pil, format='JPEG')
        img_b64 = base64.b64encode(image_pil.getvalue()).decode('utf-8')
        post_json["instances"].append({"b64": img_b64})
        response = requests.post(self.place_predict_url, data=json.dumps(post_json))
        response.raise_for_status()
        prediction = response.json()["predictions"]
        return prediction

    @staticmethod
    def get_image_from_url(image_url):
        response = requests.get(image_url)
        response = response.content
        bytes_obj = io.BytesIO(response)
        image = Image.open(bytes_obj)
        return image

    @staticmethod
    def get_image_from_base64(image_b64):
        data = base64.b64decode(image_b64)
        image = io.BytesIO(data)
        image = Image.open(image)
        return image
