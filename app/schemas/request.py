from pydantic import BaseModel

class PredictionRequest(BaseModel):
    features: dict

    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "chassisType": "hatchback",
                    "color": "white",
                    "driveWheels": "fwd",
                    "engineVolume": 1.6,
                    "fuelType": "diesel",
                    "manufacturer": "audi",
                    "mileage": 56600.0,
                    "model": "a1",
                    "modelYear": 2011,
                    "power": 105.0
                }
            }
        }
