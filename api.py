from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

# 1. Initialize FastAPI Application
app = FastAPI(title="Bengaluru House Price Prediction API", version="1.0")

# 2. Load the frozen ML artifacts once when the server starts
with open('bengaluru_house_production_bundle.pkl', 'rb') as f:
    artifacts = pickle.load(f)

model = artifacts['model']
encoder = artifacts['encoder']
scaler = artifacts['scaler']
categorical_features = artifacts['categorical_features']
cols_to_scale = artifacts['cols_to_scale']
all_encoded_cols = encoder.get_feature_names_out(categorical_features)

# 3. Define the structural contract for incoming data using Pydantic
class PropertyInputs(BaseModel):
    area_type: str
    availability: str
    location: str
    size: str
    society: str
    total_sqft: float
    bath: float
    balcony: float

@app.get("/")
def home():
    return {"message": "Bengaluru House Price Prediction API is live!"}

# 4. The Core Prediction Endpoint
@app.post("/predict")
def predict_price(data: PropertyInputs):
    # Convert incoming JSON data into a pandas DataFrame
    user_data = pd.DataFrame([data.model_dump()])
    
    # Process categories using the fitted encoder
    encoded_user_features = encoder.transform(user_data[categorical_features])
    encoded_user_df = pd.DataFrame(encoded_user_features, columns=all_encoded_cols)

    # Combine numerical variables and scale them
    numerical_user_df = user_data[['total_sqft', 'bath', 'balcony']]
    x_user = pd.concat([encoded_user_df, numerical_user_df], axis=1)
    x_user[cols_to_scale] = scaler.transform(x_user[cols_to_scale])

    # Enforce strict feature ordering matching training blueprints
    x_user = x_user[model.feature_names_in_]

    # Execute Random Forest Inference & reverse the target's natural log scale
    predicted_log_price = model.predict(x_user)[0]
    predicted_price_lakhs = float(np.expm1(predicted_log_price))

    return {"estimated_price_lakhs": round(predicted_price_lakhs, 2)}
