import joblib
from sklearn.datasets import load_iris

def test_model_prediction():
    model = joblib.load("iris_model.pkl")
    iris = load_iris()
    sample = iris.data[0].reshape(1, -1)
    prediction = model.predict(sample)
    assert prediction in [0, 1, 2], "Prediction out of class range"
