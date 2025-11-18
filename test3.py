import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

DATA_PATH = "Indoor_Plant_Health_and_Growth_Factors.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {os.path.abspath(DATA_PATH)}")

# Load Data
df = pd.read_csv(DATA_PATH)
print("Original Shape:", df.shape)
print(df.head(), "\n")

# Handle Missing Values
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# Convert Plant_ID to numeric
plant_id_unique = df['Plant_ID'].unique()
plant_id_mapping = {name: i+1 for i, name in enumerate(plant_id_unique)}
df['Plant_ID'] = df['Plant_ID'].map(plant_id_mapping)

TARGET = "Health_Score"
X = df.drop(TARGET, axis=1)
y = df[TARGET]

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print("\nNumeric features:", numeric_features)
print("Categorical features :", categorical_features)

# One-Hot Encoding for categorical columns
encoder = OneHotEncoder(sparse_output=False, drop='first')
X_cat_encoded = encoder.fit_transform(X[categorical_features])
encoded_cat_cols = encoder.get_feature_names_out(categorical_features)
df_cat_encoded = pd.DataFrame(X_cat_encoded, columns=encoded_cat_cols, index=X.index)
X = pd.concat([X[numeric_features], df_cat_encoded], axis=1)

# Scale numeric columns
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Initialize and fit model
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Predictions and evaluation
train_pred = rf.predict(X_train)
r2_train = metrics.r2_score(y_train, train_pred)
print("R-squared value on Training set: ", round(r2_train, 5))

# Save Processed Data
df_processed = X_train.copy()
df_processed['target'] = y_train.values
output_file = "Processed_Indoor_Plant_Data.csv"
df_processed.to_csv(output_file, index=False)
print(f"\nProcessed data saved successfully as: {output_file}")

df_test = X_test.copy()
df_test['target'] = y_test.values
df_test.to_csv("Processed_Indoor_Plant_Data_test.csv", index=False)

# Check processed data
print(df_processed.isna().sum())
print(df_processed.dtypes)
print(df_processed.describe())