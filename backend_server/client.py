import requests
import tensorflow as tf
import tensorflow_hub as hub


def post_request(data):
    url = 'http://localhost:5000/upload-image'
    r = requests.post(url, json=data)
    print(r.json())


if __name__ == "__main__":
    data = {'first_field': "haha", "second_field": "haha_2"}
    post_request(data)
