import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load Data ---
DATA_PATH = "Indoor_Plant_Health_and_Growth_Factors.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {os.path.abspath(DATA_PATH)}")

df = pd.read_csv(DATA_PATH)
print("Original Shape:", df.shape)

# --- Exploratory Data Analysis: Correlation Heatmap ---
numeric_df = df.select_dtypes(include=['float64', 'int64'])
if numeric_df.shape[1] > 1:
    corr = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, linewidths=0.5, fmt=".2f", cmap="viridis")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.show()
    print("Correlation heatmap saved as correlation_heatmap.png")

# --- Drop Unnecessary Columns ---
columns_to_drop = ['Plant_ID', 'Health_Notes']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
print(f"Dropped columns: {columns_to_drop}")
print("Shape after dropping:", df.shape)

# --- Handle Missing Values ---
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])
print("Missing values handled.")

# --- Remove Outliers using IQR ---
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
def remove_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    before = data.shape[0]
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    after = data.shape[0]
    print(f"Removed {before - after} outliers in {col}")
    return data

for col in numeric_cols:
    df = remove_outliers_iqr(df, col)
print("Shape after outlier removal:", df.shape)

# --- Define Target and Features ---
TARGET = "Health_Score"
X = df.drop(TARGET, axis=1)
y = df[TARGET]

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# --- One-Hot Encoding ---
encoder = OneHotEncoder(sparse_output=False, drop='first')
X_cat = encoder.fit_transform(X[categorical_features])
cat_cols = encoder.get_feature_names_out(categorical_features)
df_cat = pd.DataFrame(X_cat, columns=cat_cols, index=X.index)
X_full = pd.concat([X[numeric_features], df_cat], axis=1)

# --- Scale Numeric Features ---
scaler = StandardScaler()
X_full[numeric_features] = scaler.fit_transform(X_full[numeric_features])

# --- Feature Selection: Top 10 using RF ---
X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)
rf_tmp = RandomForestRegressor(n_estimators=200, random_state=42)
rf_tmp.fit(X_train_tmp, y_train_tmp)
importances = rf_tmp.feature_importances_
feat_df = pd.DataFrame({'feature': X_full.columns, 'importance': importances}).sort_values(by='importance', ascending=False)
top_10_features = feat_df.head(10)['feature'].tolist()
print("Top 10 Features:", top_10_features)

X = X_full[top_10_features]

# --- Final Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# --- Hyperparameter Tuning with GridSearchCV ---
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# --- Evaluate ---
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)
r2_train = metrics.r2_score(y_train, pred_train)
r2_test = metrics.r2_score(y_test, pred_test)
mae = metrics.mean_absolute_error(y_test, pred_test)
rmse = np.sqrt(metrics.mean_squared_error(y_test, pred_test))
print("R² Train:", round(r2_train, 4))
print("R² Test:", round(r2_test, 4))
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))

# --- Feature Importance ---
feat_imp = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_}).sort_values(by='importance', ascending=False)
print("Feature Importances:\n", feat_imp)

# --- Save Model, Scaler, Encoder ---
with open("final_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("final_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("final_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
print("Model, Scaler, and Encoder saved.")

# --- Save Processed Data ---
df_train = X_train.copy()
df_train['target'] = y_train
df_train.to_csv("final_processed_train.csv", index=False)

df_test = X_test.copy()
df_test['target'] = y_test
df_test.to_csv("final_processed_test.csv", index=False)
print("Processed datasets saved.")
