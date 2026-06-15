import os
import zipfile
import kaggle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
server_df.info()
server_df.describe()
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
y = server2_df['price'].reset_index(drop=True)

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
model = LinearRegression()
model.fit(x_scaled_df, y_train)
print("Model Training is Complete")
#print("Model R2 Score on test set: ", model.score(x_test_final, y_test))
