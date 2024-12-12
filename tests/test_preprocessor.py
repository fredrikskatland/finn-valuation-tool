import pandas as pd
import numpy as np
from app.models.preprocessor import preprocess, InferencePreprocessor
import os

# Set the correct path 


def test_preprocess():
    # Sample training and test data
    train_data = {
        "chassisType": ["hatchback", "sedan"],
        "color": ["white", "black"],
        "driveWheels": ["fwd", "rwd"],
        "engineVolume": [1.6, 2.0],
        "fuelType": ["diesel", "petrol"],
        "manufacturer": ["audi", "bmw"],
        "mileage": [56600, 30000],
        "model": ["a1", "x3"],
        "modelYear": [2011, 2015],
        "asking_price": [115937.0, 250000.0],
        "power": [105, 150],
    }

    test_data = {
        "chassisType": ["sedan"],
        "color": ["black"],
        "driveWheels": ["rwd"],
        "engineVolume": [1.8],
        "fuelType": ["petrol"],
        "manufacturer": ["bmw"],
        "mileage": [40000],
        "model": ["x3"],
        "modelYear": [2016],
        "asking_price": [260000.0],
        "power": [140],
    }

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # Preprocess data
    categorical_features = ["chassisType", "color", "driveWheels", "fuelType", "manufacturer", "model"]
    train_features, train_labels, test_features, test_labels = preprocess(train_df, test_df, categorical_features)

    # Assertions
    assert train_features.shape[1] == test_features.shape[1], "Train and test features must have the same number of columns"
    assert not train_features.isna().any().any(), "No missing values should exist in train features"
    assert not test_features.isna().any().any(), "No missing values should exist in test features"
    assert train_labels.dtype == float, "Train labels should be float"
    assert test_labels.dtype == float, "Test labels should be float"

def test_inference_preprocessor():
    # Columns from training
    column_order = ["chassisType_hatchback", "chassisType_sedan", "color_white", "color_black", "driveWheels_fwd", "driveWheels_rwd",
                    "engineVolume", "fuelType_diesel", "fuelType_petrol", "manufacturer_audi", "manufacturer_bmw", "mileage",
                    "model_a1", "model_x3", "modelYear", "power"]

    # Save column order as CSV
    pd.Series(column_order).to_csv("train_columns.csv", index=False)

    # Initialize inference preprocessor
    preprocessor = InferencePreprocessor(categorical_features=["chassisType", "color", "driveWheels", "fuelType", "manufacturer", "model"],
                                         column_order_path="train_columns.csv")

    # Input for prediction
    features_dict = {
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

    # Preprocess input
    processed_input = preprocessor.preprocess(features_dict)

    # Assertions
    assert processed_input.shape[1] == len(column_order), "Processed input must match the training column order"
    assert processed_input.dtype == np.float32, "Processed input must be of type float32"
