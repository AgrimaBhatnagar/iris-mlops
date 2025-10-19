import pytest
import joblib
import numpy as np
from sklearn.datasets import load_iris

# Load trained model
model = joblib.load("iris_model.pkl")

# -----------------------------
# Evaluation tests
# -----------------------------

def test_model_prediction_shape():
    iris = load_iris()
    X_sample = iris.data[:5]
    y_pred = model.predict(X_sample)
    assert y_pred.shape[0] == X_sample.shape[0], "Predictions should match number of samples"

def test_model_prediction_values():
    iris = load_iris()
    X_sample = iris.data[:5]
    y_pred = model.predict(X_sample)
    assert all([y in [0,1,2] for y in y_pred]), "Predicted classes must be 0, 1, or 2"

# -----------------------------
# Data validation tests
# -----------------------------

def test_input_data_shape():
    iris = load_iris()
    X = iris.data
    assert X.shape[1] == 4, "Input data must have 4 features"

def test_input_data_non_negative():
    iris = load_iris()
    X = iris.data
    assert np.all(X >= 0), "All features should be non-negative"

def test_input_data_no_missing():
    iris = load_iris()
    X = iris.data
    assert not np.isnan(X).any(), "No missing values allowed in input data"
