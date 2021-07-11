from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

# Test data 1
test_data1 = {
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

# Test data 2
test_data2 = {
    "age": 47,
    "workclass": "Private",
    "fnlgt": 51835,
    "education": "Prof-school",
    "education-num": 15,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Wife",
    "race": "White",
    "sex": "Female",
    "capital-gain": 54302,
    "capital-loss": 0,
    "hours-per-week": 60,
    "native-country": "Honduras"
}


def test_get_root():
    '''
        Test that the root returns a message
    '''
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"Greeting": "Welcome!"}


def test_post_predict_less_than_50k():
    '''
        Test that predicted value is less than 50k
    '''
    r = client.post("/predict", json=test_data1)
    assert r.status_code == 200
    assert r.json() == {'prediction': 'Salary <= 50k'}


def test_post_predict_greater_than_50k():
    '''
        Test that predicted value is greater than 50k
    '''
    r = client.post("/predict", json=test_data2)
    assert r.status_code == 200
    assert r.json() == {'prediction': 'Salary > 50k'}
