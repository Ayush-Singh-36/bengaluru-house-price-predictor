import streamlit as st
import pickle
import requests  # Used to communicate with our FastAPI server

# Set up browser page configuration
st.set_page_config(page_title="Bengaluru House Price Predictor", layout="centered")

st.title("🏡 Bengaluru House Price Predictor")
st.write("Enter the property details below to estimate its real estate market value.")

# 1. Define the URL of your FastAPI endpoint
# Change this line in app.py:
API_URL = "http://backend-api:8000/predict"

# 2. Load just the encoder metadata to populate the dropdown menus safely
@st.cache_resource
def load_dropdown_options():
    with open('bengaluru_house_production_bundle.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    
    # We only extract column names here—no model math or scalers needed!
    encoder = artifacts['encoder']
    cat_features = artifacts['categorical_features']
    all_cols = encoder.get_feature_names_out(cat_features)
    
    return {
        'area_types': [col.replace('area_type_', '') for col in all_cols if col.startswith('area_type_')],
        'availabilities': [col.replace('availability_', '') for col in all_cols if col.startswith('availability_')],
        'locations': [col.replace('location_', '') for col in all_cols if col.startswith('location_')],
        'sizes': [col.replace('size_', '') for col in all_cols if col.startswith('size_')]
    }

options = load_dropdown_options()

# 3. Build the User Interface Layout
col1, col2 = st.columns(2)

with col1:
    user_area_type = st.selectbox("Area Type", sorted(options['area_types']))
    user_location = st.selectbox("Location", sorted(options['locations']))
    user_size = st.selectbox("Size / BHK", sorted(options['sizes']))
    user_availability = st.selectbox("Availability Status", sorted(options['availabilities']))

with col2:
    user_sqft = st.number_input("Total Square Footage", min_value=100, max_value=50000, value=1200)
    user_bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
    user_balcony = st.number_input("Number of Balconies", min_value=0, max_value=5, value=1)

# Default society name for simplicity
user_society = "other"

# 4. Prediction Execution Block via API
if st.button("Calculate Estimated Value", type="primary"):
    
    # Bundle inputs into a standard Python dictionary (JSON shape)
    payload = {
        "area_type": user_area_type,
        "availability": user_availability,
        "location": user_location,
        "size": user_size,
        "society": user_society,
        "total_sqft": float(user_sqft),
        "bath": float(user_bath),
        "balcony": float(user_balcony)
    }

    try:
        # Send an HTTP POST request to the FastAPI backend microservice
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            predicted_price = result["estimated_price_lakhs"]
            st.success(f"### Estimated Price: ₹ {predicted_price:.2f} Lakhs")
        else:
            st.error(f"Backend Server Error: Received status code {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the Backend API! Is your Uvicorn terminal running?")