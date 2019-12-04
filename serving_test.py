import io
import os
import json
import base64
import requests
import numpy as np
from PIL import Image
from cv2 import resize

TEST_IMAGE_URL = 'http://places2.csail.mit.edu/imgs/demo/6.jpg'

def get_b64(url):
    img_response = requests.get(url)
    img_response = img_response.content
    bytes_obj = io.BytesIO(img_response)
    image = Image.open(bytes_obj)
    image = np.array(image, dtype=np.uint8)
    image = resize(image, (224, 224))
    image = Image.fromarray(image, mode="RGB")
    image_pil = io.BytesIO()
    image.save(image_pil, format='JPEG')
    img_b64 = base64.b64encode(image_pil.getvalue()).decode('utf-8')
    return img_b64


post_json = {
    "instances": [
        {
            "b64": get_b64('http://places2.csail.mit.edu/imgs/demo/6.jpg')
        },
        {
            "b64": get_b64('http://places2.csail.mit.edu/imgs/demo/5.jpg')
        }
    ]
}
response = requests.post("http://172.29.100.23:8501/v1/models/places:predict", data=json.dumps(post_json))
response.raise_for_status()
prediction = response.json()["predictions"]

# load the class label
file_name = 'categories_places365.txt'
classes = []
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])

for preds in prediction:
    sum = 0
    for i in preds:
        sum += i
    print(sum)
    result = [[name, prob] for name, prob in zip(classes, preds)]
    result = sorted(result, key=lambda x:x[1], reverse=True)
    print(result)
    # top_preds = np.argsort(preds)[::-1][0:5]
    # for i in range(0, 5):
    #     print(classes[top_preds[i]])