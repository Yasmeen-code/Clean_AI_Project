import pandas as pd
import numpy as np
import pickle

# Load saved artifacts
with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("final_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("final_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Load top 10 features from training (assuming we know them or load from file)
# For simplicity, hardcode based on previous output
top_10_features = ['Soil_Moisture_%', 'Height_cm', 'Watering_Amount_ml', 'Room_Temperature_C', 'Humidity_%', 'Fertilizer_Amount_ml', 'Leaf_Count', 'New_Growth_Count', 'Watering_Frequency_days', 'Fertilizer_Type_Liquid feed']

# Sample new data (simulate unseen data)
sample_data = {
    'Height_cm': [15.0],
    'Leaf_Count': [25],
    'New_Growth_Count': [5],
    'Room_Temperature_C': [22.0],
    'Humidity_%': [60.0],
    'Soil_Moisture_%': [45.0],
    'Watering_Amount_ml': [200.0],
    'Fertilizer_Amount_ml': [10.0],
    'Watering_Frequency_days': [7],
    'Plant_Type': ['Monstera deliciosa'],
    'Soil_Type': ['Loamy'],
    'Pest_Presence': ['None'],
    'Pest_Severity': ['None'],
    'Fertilizer_Type': ['Liquid feed']
}

df_sample = pd.DataFrame(sample_data)

# Separate numeric and categorical
numeric_features = ['Height_cm', 'Leaf_Count', 'New_Growth_Count', 'Room_Temperature_C', 'Humidity_%', 'Soil_Moisture_%', 'Watering_Amount_ml', 'Fertilizer_Amount_ml', 'Watering_Frequency_days']
categorical_features = ['Plant_Type', 'Soil_Type', 'Pest_Presence', 'Pest_Severity', 'Fertilizer_Type']

# Scale numeric
df_sample[numeric_features] = scaler.transform(df_sample[numeric_features])

# Encode categorical
X_cat = encoder.transform(df_sample[categorical_features])
cat_cols = encoder.get_feature_names_out(categorical_features)
df_cat = pd.DataFrame(X_cat, columns=cat_cols, index=df_sample.index)

# Combine
X_full = pd.concat([df_sample[numeric_features], df_cat], axis=1)

# Select top 10 features
X_pred = X_full[top_10_features]

# Predict
prediction = model.predict(X_pred)
print("Prediction for sample data:", prediction[0])

# Edge case: Missing values
print("\nTesting edge case: Missing values")
sample_missing = sample_data.copy()
sample_missing['Humidity_%'] = [np.nan]  # Simulate missing
df_missing = pd.DataFrame(sample_missing)

# Handle missing (using median/mode as in training)
for col in df_missing.columns:
    if pd.api.types.is_numeric_dtype(df_missing[col]):
        df_missing[col] = df_missing[col].fillna(df_missing[col].median())
    else:
        df_missing[col] = df_missing[col].fillna(df_missing[col].mode()[0])

# Repeat preprocessing
df_missing[numeric_features] = scaler.transform(df_missing[numeric_features])
X_cat_miss = encoder.transform(df_missing[categorical_features])
df_cat_miss = pd.DataFrame(X_cat_miss, columns=cat_cols, index=df_missing.index)
X_full_miss = pd.concat([df_missing[numeric_features], df_cat_miss], axis=1)
X_pred_miss = X_full_miss[top_10_features]
pred_miss = model.predict(X_pred_miss)
print("Prediction with missing value handled:", pred_miss[0])

# Edge case: Out-of-range values
print("\nTesting edge case: Out-of-range values")
sample_range = sample_data.copy()
sample_range['Soil_Moisture_%'] = [200.0]  # Unrealistic high
df_range = pd.DataFrame(sample_range)
df_range[numeric_features] = scaler.transform(df_range[numeric_features])
X_cat_range = encoder.transform(df_range[categorical_features])
df_cat_range = pd.DataFrame(X_cat_range, columns=cat_cols, index=df_range.index)
X_full_range = pd.concat([df_range[numeric_features], df_cat_range], axis=1)
X_pred_range = X_full_range[top_10_features]
pred_range = model.predict(X_pred_range)
print("Prediction with out-of-range value:", pred_range[0])
