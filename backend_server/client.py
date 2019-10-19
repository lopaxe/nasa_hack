import requests


def post_request(data):
    url = 'http://localhost:5000/upload-image'
    #r = requests.post(url, json=data)
    r = requests.post(url, files=data)
    print(r)


if __name__ == "__main__":
    post_request({'media': open('pics/Aurimas_Griciunas.jpg', 'rb')})
