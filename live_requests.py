import requests
import json
url = "http://127.0.0.1:8000"

request_data1 = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

request_data2 = {
    "age": 38,
    "workclass": "Private",
    "fnlgt": 215646,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Divorced",
    "occupation": "Handlers-cleaners",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

response = requests.post(
    f'{url}/predict/',
    data=json.dumps(request_data1))
print(response.status_code)
print(response.json())

response = requests.post(
    f'{url}/predict/',
    data=json.dumps(request_data2))
print(response.status_code)
print(response.json())
