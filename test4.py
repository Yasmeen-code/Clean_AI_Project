import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# --- Load Data ---
DATA_PATH = "Indoor_Plant_Health_and_Growth_Factors.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {os.path.abspath(DATA_PATH)}")

df = pd.read_csv(DATA_PATH)
print("Original Shape:", df.shape)
print(df.head(), "\n")

# --- Handle Missing Values ---
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# --- Define Target and Features ---
TARGET = "Health_Score"
X = df.drop(TARGET, axis=1)
y = df[TARGET]

# --- Identify Feature Types ---
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# --- One-Hot Encoding for Categorical Data ---
encoder = OneHotEncoder(sparse_output=False, drop='first')
X_cat = encoder.fit_transform(X[categorical_features])
cat_cols = encoder.get_feature_names_out(categorical_features)
df_cat = pd.DataFrame(X_cat, columns=cat_cols, index=X.index)

# Combine numerical + encoded categorical
X_full = pd.concat([X[numeric_features], df_cat], axis=1)

# --- Scale Numeric Features ---
scaler = StandardScaler()
X_full[numeric_features] = scaler.fit_transform(X_full[numeric_features])

# --- Temporary Train/Test split to get top features ---
X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

rf_tmp = RandomForestRegressor(n_estimators=200, random_state=42)
rf_tmp.fit(X_train_tmp, y_train_tmp)

importances = rf_tmp.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_full.columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)

top_10_features = feature_importance_df.head(10)['feature'].tolist()
print("\nSelected Top 10 Features:", top_10_features)

# --- Final Train/Test Split Using Top 10 Selected Features ---
X = X_full[top_10_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

print("\nFinal Training set shape:", X_train.shape)
print("Final Test set shape:", X_test.shape)

# --- Final Model Training ---
rf = RandomForestRegressor(n_estimators=250, max_depth=12, random_state=42)
rf.fit(X_train, y_train)

# --- Evaluation ---
train_pred = rf.predict(X_train)
test_pred = rf.predict(X_test)

print("\nR² Train:", round(metrics.r2_score(y_train, train_pred), 4))
print("R² Test:", round(metrics.r2_score(y_test, test_pred), 4))
print("MAE:", round(metrics.mean_absolute_error(y_test, test_pred), 4))
print("RMSE:", round(np.sqrt(metrics.mean_squared_error(y_test, test_pred)), 4))

# --- Save Processed Datasets ---
df_train = X_train.copy()
df_train["target"] = y_train
df_train.to_csv("Processed_Indoor_Plant_Data_top10.csv", index=False)

df_test = X_test.copy()
df_test["target"] = y_test
df_test.to_csv("Processed_Indoor_Plant_Data_test_top10.csv", index=False)

print("\nProcessed files saved with top 10 features only.")
