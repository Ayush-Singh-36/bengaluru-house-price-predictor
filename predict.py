import pickle
import pandas as pd
import numpy as np

print("Loading Prediction Engine...")
# 1. Load the frozen production artifacts
with open('bengaluru_house_production_bundle.pkl', 'rb') as f:
    artifacts = pickle.load(f)

model = artifacts['model']
encoder = artifacts['encoder']
scaler = artifacts['scaler']
categorical_features = artifacts['categorical_features']
cols_to_scale = artifacts['cols_to_scale']

print("\n--- Bengaluru House Price Predictor ---")
print("Please enter the property details below:\n")

# 2. Collect inputs from the user via the terminal
user_area_type = input("Enter Area Type (e.g., Super built-up Area, Plot Area): ").strip()
user_availability = input("Enter Availability (e.g., Ready To Move, 19-Dec): ").strip()
user_location = input("Enter Location (e.g., Whitefield, Electronic City): ").strip()
user_size = input("Enter Size (e.g., 2 BHK, 3 BHK): ").strip()
user_society = input("Enter Society Name (or leave blank for 'other'): ").strip() or "other"

try:
    user_sqft = float(input("Enter Total Square Footage: "))
    user_bath = float(input("Enter Number of Bathrooms: "))
    user_balcony = float(input("Enter Number of Balconies: "))
except ValueError:
    print("Error: Square footage, bathrooms, and balconies must be numerical values!")
    exit()

# 3. Process the user input exactly like the training data
user_data = pd.DataFrame([{
    'area_type': user_area_type,
    'availability': user_availability,
    'location': user_location,
    'size': user_size,
    'society': user_society,
    'total_sqft': user_sqft,
    'bath': user_bath,
    'balcony': user_balcony
}])

# Encode the text categories
encoded_user_features = encoder.transform(user_data[categorical_features])
encoded_user_df = pd.DataFrame(encoded_user_features, columns=encoder.get_feature_names_out(categorical_features))

# Structure the full X matrix for the user
numerical_user_df = user_data[['total_sqft', 'bath', 'balcony']]
x_user = pd.concat([encoded_user_df, numerical_user_df], axis=1)

# Scale the numerical components using the training scaler constants
x_user[cols_to_scale] = scaler.transform(x_user[cols_to_scale])

# 4. Generate the prediction
predicted_price = model.predict(x_user)[0]

print("\n" + "="*40)
print(f" Estimated House Price: {predicted_price:.2f} Lakhs")
print("="*40)