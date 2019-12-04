# -*- coding: utf-8 -*-

import flask
import json

from backend.PlacesModel import PlacesModel

app = flask.Flask(__name__)
place_model = PlacesModel()


@app.route("/")
def homepage():
    return "Welcome to the SOUL Place Classification REST API!"


@app.route("/health")
def health_check():
    return "OK"


@app.route("/api/predict", methods=["POST", "GET"])
def predict_sensitive():
    res = {
        "code": 200,
        "message": "OK"
    }

    if flask.request.method == "POST":
        data = flask.request.data.decode('utf-8')
        data = json.loads(data)
        if "imgUrl" in data:
            prediction = place_model.predict(place_model.get_image_from_url(data["imgUrl"]))
            result = [[name, prob] for name, prob in zip(place_model.class_names, prediction[0])]
            result = sorted(result, key=lambda x: x[1], reverse=True)
            res.update({"prediction": result})
        elif "imgBase64Batch" in data:
            image_list = []
            for img_b64 in data["imgBase64Batch"]:
                image_list.append(place_model.get_image_from_base64(img_b64))
            predictions = place_model.predict_batch(image_list)
            result_list = []
            for prediction in predictions:
                result = [[name, prob] for name, prob in zip(place_model.class_names, prediction)]
                result = sorted(result, key=lambda x: x[1], reverse=True)
                result_list.append(result)
            res.update({"prediction": result_list})
        else:
            res.update({"code": 400})
            res.update({ "message": "Bad Request."})
    return flask.jsonify(res)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8585)
