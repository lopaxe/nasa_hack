
from flask import Flask, request, jsonify

import tensorflow_hub as hub

model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')

app = Flask(__name__)


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
        return jsonify({"status": "successful upload"})
    else:
        return jsonify({"status": "no files sent"})


if __name__ == "__main__":
    app.run(debug=True)
    # For public web serving:
    # app.run(host='0.0.0.0')
    app.run()
