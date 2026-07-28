import streamlit as st
import pickle
import pandas as pd

# Set up browser page configuration
st.set_page_config(page_title="Bengaluru House Price Predictor", layout="centered")

st.title("🏡 Bengaluru House Price Predictor")
st.write("Enter the property details below to estimate its real estate market value.")

# 1. Load the full artifact bundle directly
@st.cache_resource
def load_artifacts():
    with open('bengaluru_house_production_bundle.pkl', 'rb') as f:
        return pickle.load(f)

try:
    artifacts = load_artifacts()

    # Extract components from the pickle file
    model = artifacts['model']
    encoder = artifacts['encoder']
    scaler = artifacts.get('scaler')
    cat_features = artifacts['categorical_features']

    # Extract dropdown choices dynamically from encoder categories
    categories = encoder.categories_
    options = {
        'area_types': list(categories[0]),
        'availabilities': list(categories[1]),
        'locations': list(categories[2]),
        'sizes': list(categories[3]),
        'societies': list(categories[4]) if len(categories) > 4 else ['other']
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

    user_society = options['societies'][0] if options['societies'] else "other"

    # 3. Direct In-Memory Prediction Execution Block
    if st.button("Calculate Estimated Value", type="primary"):
        try:
            # Construct complete input DataFrame matching raw feature names
            input_df = pd.DataFrame([{
                "area_type": user_area_type,
                "availability": user_availability,
                "location": user_location,
                "size": user_size,
                "society": user_society,
                "total_sqft": float(user_sqft),
                "bath": float(user_bath),
                "balcony": float(user_balcony)
            }])

            # Extract categorical and numerical parts
            cat_df = input_df[cat_features]
            num_df = input_df[["total_sqft", "bath", "balcony"]]

            # Encode categoricals using fitted OneHotEncoder
            cat_encoded = encoder.transform(cat_df)
            
            # Use get_feature_names_out on encoder directly
            encoded_feature_names = encoder.get_feature_names_out(cat_features)
            cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoded_feature_names)

            # Combine numerical and encoded categorical features
            X_input = pd.concat([num_df.reset_index(drop=True), cat_encoded_df.reset_index(drop=True)], axis=1)

            # Scale features if scaler exists
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
