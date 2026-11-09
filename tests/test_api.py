from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_predict_without_model_returns_error():
    response = client.post("/predict", json={"players": []})
    assert response.status_code in (400, 503)
