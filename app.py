import streamlit as st
import pickle
import pandas as pd

# Set up browser page configuration
st.set_page_config(page_title="Bengaluru House Price Predictor", layout="centered")

st.title("🏡 Bengaluru House Price Predictor")
st.write("Enter the property details below to estimate its real estate market value.")

# 1. Load the full artifact bundle directly with loading indicator
@st.cache_resource
def load_artifacts():
    with st.spinner("Loading model and artifacts from disk..."):
        with open('bengaluru_house_production_bundle.pkl', 'rb') as f:
            data = pickle.load(f)
    return data

try:
    artifacts = load_artifacts()

    # Extract components from the pickle file
    model = artifacts['model']
    encoder = artifacts['encoder']
    scaler = artifacts.get('scaler')
    cat_features = artifacts['categorical_features']

    # Extract dropdown choices dynamically from encoder
    all_cols = encoder.get_feature_names_out(cat_features)
    options = {
        'area_types': [col.replace('area_type_', '') for col in all_cols if col.startswith('area_type_')],
        'availabilities': [col.replace('availability_', '') for col in all_cols if col.startswith('availability_')],
        'locations': [col.replace('location_', '') for col in all_cols if col.startswith('location_')],
        'sizes': [col.replace('size_', '') for col in all_cols if col.startswith('size_')]
    }

    # 2. Build the User Interface Layout
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

    user_society = "other"

    # 3. Direct In-Memory Prediction Execution Block
    if st.button("Calculate Estimated Value", type="primary"):
        try:
            # Create DataFrames for features
            cat_df = pd.DataFrame([{
                "area_type": user_area_type,
                "availability": user_availability,
                "location": user_location,
                "size": user_size,
                "society": user_society
            }])
            
            num_df = pd.DataFrame([{
                "total_sqft": float(user_sqft),
                "bath": float(user_bath),
                "balcony": float(user_balcony)
            }])

            # Transform categorical features
            cat_encoded = encoder.transform(cat_df)
            cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_features))

            # Combine numerical and categorical features
            X_input = pd.concat([num_df, cat_encoded_df], axis=1)

            # Scale features if scaler exists in pipeline
            if scaler:
                X_input = scaler.transform(X_input)

            # Generate direct prediction
            predicted_price = float(model.predict(X_input)[0])

            # Auto-convert >= 100 Lakhs to Crore
            if predicted_price >= 100:
                price_in_crore = predicted_price / 100.0
                st.success(f"### Estimated Price: ₹ {price_in_crore:.2f} Crore")
            else:
                st.success(f"### Estimated Price: ₹ {predicted_price:.2f} Lakhs")

        except Exception as e:
            st.error(f"Error calculating prediction: {str(e)}")

except Exception as e:
    st.error(f"Failed to load model file: {str(e)}")
