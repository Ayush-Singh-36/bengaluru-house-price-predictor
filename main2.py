import os
import numpy as np
import pandas as pd
import pickle

# Scikit-learn preprocessing and splitting utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Model and Evaluation Metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

"""
# 1. This is the exact Kaggle dataset identifier for the Bengaluru House Data
dataset_slug = "amitabhajoy/bengaluru-house-price-data"

# 2. This creates a folder named 'data' inside your current VS Code workspace
download_path = "./data"

print("Downloading Bengaluru House Price dataset from Kaggle...")

# 3. This executes the automated download and extraction via Command Prompt
os.system(f"kaggle datasets download -d {dataset_slug} -p {download_path} --unzip")

print(f"Done! Your files have been saved to the '{download_path}' folder.")
"""
#Data Preprocessing
# Load the dataset
server_df = pd.read_csv("data/Bengaluru_House_Data.csv")

# Clean up location spaces and group rare locations
server_df['location'] = server_df['location'].astype(str).str.strip()
location_stats = server_df['location'].value_counts()
locations_less_than_10 = location_stats[location_stats <= 10]
server_df['location'] = server_df['location'].apply(lambda x: 'other' if x in locations_less_than_10 else x)

# Fill missing values
server2_df = server_df.fillna(method='ffill')


def func(total_sqft):
    try:
        total_sqft_str = str(total_sqft).strip()
        if '-' in total_sqft_str:
            tokens = total_sqft_str.split('-')
            return (float(tokens[0]) + float(tokens[1])) / 2
        else:
            return float(total_sqft)
    except:
        return None

func(server2_df['total_sqft'].iloc[0])


server2_df['total_sqft'] = server2_df['total_sqft'].apply(func)
server2_df['total_sqft'] = server2_df['total_sqft'].fillna(server2_df['total_sqft'].mean())

#Encoding the categorical data
categorical_features = ['area_type', 'availability', 'location', 'size', 'society']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(server2_df[categorical_features])

# Convert encoded features to a DataFrame with string column names
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

# Separate features and target (Reset indices to prevent alignment issues during concat)
numerical_df = server2_df[['total_sqft', 'bath', 'balcony']].reset_index(drop=True)
encoded_df = encoded_df.reset_index(drop=True)

x = pd.concat([encoded_df, numerical_df], axis=1)
#Taking the log of the price
y = np.log1p(server2_df['price']).reset_index(drop=True)

#Spliting data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
print("x_train size = ", x_train.shape, "\nx_test size = ", x_test.shape, "\ny_train size = ", y_train.shape, "\ny_test size = ", y_test.shape)

#Features Scaling (Standardization)
scaler = StandardScaler()
cols_to_scale = ['total_sqft', 'bath', 'balcony']
x_train_scaled_cols = scaler.fit_transform(x_train[cols_to_scale])
x_test_scaled_cols = scaler.transform(x_test[cols_to_scale])
x_scaled_df = pd.DataFrame(x_train_scaled_cols, columns=cols_to_scale)
x_train_final = x_train.copy()
x_test_final = x_test.copy()
x_train_final[cols_to_scale] = x_train_scaled_cols
x_test_final[cols_to_scale] = x_test_scaled_cols

#Training the model
print("Training Random Forest Regressor model ...")
model = RandomForestRegressor(n_estimators = 100, random_state = 42)
model.fit(x_train_final, y_train)
print("Model training is complete")

#Error metrices to see real impact
#making predictions on the test set
y_pred_log = model.predict(x_test_final)

#converting the logged predictions and logged true values back to their original states
y_test_original = np.expm1(y_test)
y_pred_original = np.expm1(y_pred_log)

#calculate explicit error metrics
mae = mean_absolute_error(y_test_original, y_pred_original)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
r2 = r2_score(y_test_original, y_pred_original)
print(f"Mean Absolute Error: {mae:.2f} lakhs, \nRoot Mean Squared Error: {rmse:.2f} Lakhs")
print(f"Model R2 Score on test set: {r2*100:.3f}%")

#Exporting the encoder & scaler

# Bundle everything the user interface will need into a single dictionary
model_artifacts = {
    'model': model,
    'encoder': encoder,
    'scaler': scaler,
    'categorical_features': categorical_features,
    'cols_to_scale': cols_to_scale
}

# Save the bundle to a file
with open('bengaluru_house_production_bundle.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)

print("All production artifacts saved successfully!")