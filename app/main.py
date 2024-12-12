from fastapi import FastAPI, HTTPException
from models.model import load_model, predict
from models.preprocessor import InferencePreprocessor
from schemas.request import PredictionRequest
from utils.logger import setup_logger
import numpy as np

# Initialize the logger
logger = setup_logger()

# Initialize the app
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application is starting up...")
    

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI application is shutting down...")

# Load the model and preprocessor
model = load_model("./my_model/dnn_model.keras")

categorical_features = ["fuelType", "driveWheels", "manufacturer", "chassisType", "model", "color"]

inference_preprocessor = InferencePreprocessor(
    categorical_features=categorical_features,
    column_order_path="./my_model/train_columns.csv"
)

@app.post("/predict/")
async def predict_endpoint(request: PredictionRequest):
    try:
        logger.info("Received a prediction request")
        processed_input = inference_preprocessor.preprocess(request.features)

        # Make prediction
        prediction = predict(model, processed_input)
        logger.info("Prediction successful")
        return {"prediction": prediction.tolist()}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))