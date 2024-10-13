import pickle

from fastapi import status
from fastapi.testclient import TestClient
from sklearn.base import BaseEstimator
from tinydb import TinyDB
from tinydb.storages import MemoryStorage

from src.main import app

client = TestClient(app)


class MockModel(BaseEstimator):
    """A mock machine learning model for the following tests."""

    def fit(self, X=None, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        # suppose the model was fitted on the features "feat1" and "feat2"
        if set(X.columns) != {"feat1", "feat2"}:
            raise ValueError()
        else:
            return [10.0]


def test_health_check_endpoint():
    """Make sure the health check endpoint returns the status code 200."""
    resp = client.get("/health")
    assert resp.status_code == status.HTTP_200_OK


def test_load_model_invalid_file():
    """Simulate uploading an invalid pickle file and make sure the endpoint returns a 400 Bad Request status."""
    invalid_file = bytes("invalid_pickle_bytes", encoding="utf-8")
    resp = client.post("/model/load", files={"model_pkl": invalid_file})
    assert resp.status_code == status.HTTP_400_BAD_REQUEST


def test_load_model_valid_file(monkeypatch):
    """Upload a valid model file and make sure it's successfully loaded."""
    monkeypatch.setattr("src.main.app_data", {})
    model_pkl = pickle.dumps(MockModel())
    resp = client.post("/model/load", files={"model_pkl": model_pkl})
    assert resp.status_code == status.HTTP_200_OK


def test_prediction_model_not_loaded():
    """Send a prediction request without loading a model, making sure the status code 400 is returned."""
    resp = client.post("/model/predict", json={"features": {"feat1": 1, "feat2": 2}})
    assert resp.status_code == status.HTTP_400_BAD_REQUEST


def test_prediction_valid_execution(monkeypatch):
    """Send a valid prediction request and make sure the expected status code and prediction value are returned."""
    monkeypatch.setattr("src.main.app_data", {"model": MockModel(), "db": TinyDB(storage=MemoryStorage)})
    resp = client.post("/model/predict", json={"features": {"feat1": 1, "feat2": 2}})
    assert resp.status_code == status.HTTP_200_OK
    assert resp.json()["prediction"] == 10.0


def test_prediction_invalid_features(monkeypatch):
    """Send a prediction request with incorrect features and ensure the status code 500 is returned."""
    monkeypatch.setattr("src.main.app_data", {"model": MockModel()})
    resp = client.post("/model/predict", json={"features": {"feat1": 1, "feat3": 2}})
    assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


def test_history_endpoint(monkeypatch):
    """Insert some fake prediction history into the mock database and verify that is correctly returned by the history
    endpoint.
    """
    db = TinyDB(storage=MemoryStorage)
    history = [
        {"features": {"feat1": 1, "feat2": 2}, "prediction": 10.0},
        {"features": {"feat1": 1, "feat2": 3}, "prediction": 20.0},
    ]
    db.insert_multiple(history)
    monkeypatch.setattr("src.main.app_data", {"db": db})
    resp = client.get("/model/history")
    assert resp.status_code == status.HTTP_200_OK
    assert resp.json() == history


def test_db_lifespan():
    """Make sure the lifespan function is correctly initializing and closing the database."""
    with TestClient(app) as client:
        resp = client.get("/model/history")
        assert resp.status_code == status.HTTP_200_OK
    resp = client.get("/model/history")
    assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
