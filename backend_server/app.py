from utils.app_config import AppConfig
from utils.transformer import FullTransformation, WeightedTransformation
from utils.planet import Planet
from utils.cleanup import perform_cleanup
from utils import hack_index as hack

from flask import Flask, request, jsonify, render_template, redirect

import tensorflow_hub as hub
import cv2 as cv
import json
import numpy as np

model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')

app = Flask(__name__)
app_config = AppConfig()


@app.route('/')
def index():
    perform_cleanup()
    return redirect("http://localhost:8000/templates/")


@app.route('/retrieve-model-outputs', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    first_field = data['first_field']
    second_field = data['second_field']
    response = {"first_field": second_field, "second_field": first_field}
    return jsonify(response)


@app.route("/upload-image", methods=["POST"])
def upload_image():

    if request.files:
        image = request.files["media"]
        #params = json.loads(request.data)
        params = {"T_star": 5400, "R": 2, "a": 2.5, "M_exo":1, "R_exo":1}
        image.save(f"{app_config.UPLOAD_PATH}/{image.filename}")
        planet_object = Planet(**params)
        survival = planet_object.surviveTotal
        if type(survival['cod']) == str:
            if "hot" in survival['cod']:
                style_path = app_config.fire_path
            elif "cold" in survival['cod']:
                style_path = app_config.ice_path
            else:
                style_path = app_config.fire_path
        else:
            style_path = app_config.fire_path
        full_transformer_object = FullTransformation(f"{app_config.UPLOAD_PATH}/{image.filename}",
                                                     style_path,
                                                     app_config.full_transformation_path,
                                                     planet_object.surviveTotal,
                                                     model)
        full_transformer_object.original_transformation.save(f"{app_config.original_transformation_path}/{image.filename}")
        full_transformer_object.full_transformation.save(f"{app_config.full_transformation_path}/{image.filename}")
        weighted_transformation_object = WeightedTransformation(f"{app_config.original_transformation_path}/{image.filename}",
                                                                f"{app_config.full_transformation_path}/{image.filename}",
                                                                "",
                                                                planet_object.surviveTotal)
        weighted_image = weighted_transformation_object.weighted_image
        weighted_rgb_image = np.einsum('ijk,k->ijk', weighted_image, planet_object.RGB)
        weighted_rgb_image = np.asarray(weighted_rgb_image, dtype=int)
        cv.imwrite(f"{app_config.weigted_transformation_path}/{image.filename}", weighted_rgb_image)
        cv.imwrite("templates/result/Result.jpg", weighted_rgb_image)
        hack.hack_index(hack.returnText(planet_object.surviveTotal))
        return jsonify({"status": "successful upload"})
    else:
        return jsonify({"status": "no files sent"})


if __name__ == "__main__":
    app.run(debug=True)
    # For public web serving:
    # app.run(host='0.0.0.0')
    #app.run()
