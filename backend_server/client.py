import requests


def post_request(data):
    url = 'http://localhost:5000/retrieve-model-outputs'
    #r = requests.post(url, json=data)
    r = requests.post(url, json=data)
    print(r.json())


def post_image(data):
    headers = {'Content-type': 'multipart/form-data'}
    url = 'http://localhost:5000/upload-image'
    r = requests.post(url, files=data)
    print(r.text)


if __name__ == "__main__":

    #post_request({'first_field': "", 'second_field': ""})

    post_image({'media': open("/Users/agriciunas/Desktop/NASA_HACKATON_DEV/pictures/Ovidijus.jpg", "rb")})
