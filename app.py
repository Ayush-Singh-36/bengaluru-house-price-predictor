import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set up browser page configuration
st.set_page_config(page_title="Bengaluru House Price Predictor", layout="centered")

st.title("🏡 Bengaluru House Price Predictor")
st.write("Enter the property details below to estimate its real estate market value.")

# 1. Load the frozen production artifacts
@st.cache_resource
def load_artifacts():
    with open('bengaluru_house_production_bundle.pkl', 'rb') as f:
        return pickle.load(f)

artifacts = load_artifacts()
model = artifacts['model']
encoder = artifacts['encoder']
scaler = artifacts['scaler']
categorical_features = artifacts['categorical_features']
cols_to_scale = artifacts['cols_to_scale']

# --- SAFE EXTRACTION OF DROPDOWN VALUES ---
# Get all feature names out using the full categorical_features list
all_encoded_cols = encoder.get_feature_names_out(categorical_features)

# Extract unique values for our dropdown selection menus safely
area_types = [col.replace('area_type_', '') for col in all_encoded_cols if col.startswith('area_type_')]
availabilities = [col.replace('availability_', '') for col in all_encoded_cols if col.startswith('availability_')]
locations = [col.replace('location_', '') for col in all_encoded_cols if col.startswith('location_')]
sizes = [col.replace('size_', '') for col in all_encoded_cols if col.startswith('size_')]

# 2. Build the User Interface Layout
col1, col2 = st.columns(2)

with col1:
    user_area_type = st.selectbox("Area Type", sorted(area_types))
    user_location = st.selectbox("Location", sorted(locations))
    user_size = st.selectbox("Size / BHK", sorted(sizes))
    user_availability = st.selectbox("Availability Status", sorted(availabilities))

with col2:
    user_sqft = st.number_input("Total Square Footage", min_value=100, max_value=50000, value=1200)
    user_bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
    user_balcony = st.number_input("Number of Balconies", min_value=0, max_value=5, value=1)

# Default society name for simplicity
user_society = "other"

# 3. Prediction Execution Block
if st.button("Calculate Estimated Value", type="primary"):
    # Convert input to DataFrame with ALL columns present
    user_data = pd.DataFrame([{
        'area_type': user_area_type,
        'availability': user_availability,
        'location': user_location,
        'size': user_size,
        'society': user_society,
        'total_sqft': float(user_sqft),
        'bath': float(user_bath),
        'balcony': float(user_balcony)
    }])

    # Transform categories using the fitted encoder (Passing all features safely)
    encoded_user_features = encoder.transform(user_data[categorical_features])
    encoded_user_df = pd.DataFrame(encoded_user_features, columns=all_encoded_cols)

    # Combine numerical variables and scale them
    numerical_user_df = user_data[['total_sqft', 'bath', 'balcony']]
    x_user = pd.concat([encoded_user_df, numerical_user_df], axis=1)
    
    # Scale numerical inputs
    x_user[cols_to_scale] = scaler.transform(x_user[cols_to_scale])

    # Ensure column order matches the model training exactly
    # (Fixes any mismatch bugs between fit and predict shapes)
    x_user = x_user[model.feature_names_in_]

    # Predict
    predicted_price = model.predict(x_user)[0]

    # Show result to user
    st.success(f"### Estimated Price: ₹ {predicted_price:.2f} Lakhs")