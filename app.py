from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from typing import Dict
import pickle
import uvicorn

app = FastAPI()

# Load the trained model and scaler
with open("model/xgboost_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
    
with open("model/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Define a request body model using Pydantic

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "API is healthy!"}

# Prediction endpoint
@app.post("/predict")
async def predict(data: list[float]):

     # Convert input list to DataFrame and create a dict with X1, X2, ..., X23 keys
    input_data = pd.DataFrame([{
        f'X{i + 1}': data[i] for i in range(len(data))
    }])

    # Scale the input data using the scaler
    input_data_scaled = scaler.transform(input_data)

    # Make the prediction
    prediction = model.predict(input_data_scaled)

    # scaled_data = pd.DataFrame([{
    #     f'X{i + 1}': input_data_scaled[0][i] for i in range(len(input_data_scaled[0]))
    # }])

    # Append new data with prediction to a CSV (data collection)
    new_data = input_data.copy()
    new_data['Y'] = prediction

    # Define the file path
    file_path = './data/new_data.csv'

    # Check if the file exists, if not, create it with headers
    if not os.path.exists(file_path):
        headers = [f"X{i+1}" for i in range(23)]  + ["Y"]
        new_data.to_csv(file_path, mode='w', header=headers, index=False)
    else:
        new_data.to_csv(file_path, mode='a', header=False, index=False)

    return {"prediction": int(prediction[0])}

# To run the server, use: uvicorn app:app --reload
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
