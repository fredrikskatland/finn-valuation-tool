import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_endpoint():
    # Payload for prediction
    payload = {
        "features": {
            "chassisType": "hatchback",
            "color": "white",
            "driveWheels": "fwd",
            "engineVolume": 1.6,
            "fuelType": "diesel",
            "manufacturer": "audi",
            "mileage": 56600,
            "model": "a1",
            "modelYear": 2011,
            "power": 105,
        }
    }

    # Send POST request to the /predict/ endpoint
    response = client.post("/predict/", json=payload)

    # Assertions
    assert response.status_code == 200, "Prediction endpoint should return status code 200"
    data = response.json()
    assert "prediction" in data, "Response should contain 'prediction' field"
    assert isinstance(data["prediction"], list), "'prediction' field should be a list"
