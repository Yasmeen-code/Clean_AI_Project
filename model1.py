import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import pickle

# -----------------------------
# Load Processed Data (Top 10 Features)
# -----------------------------
df_train = pd.read_csv("Processed_Indoor_Plant_Data_top10.csv")
df_test = pd.read_csv("Processed_Indoor_Plant_Data_test_top10.csv")

X_train = df_train.drop("target", axis=1)
y_train = df_train["target"]

X_test = df_test.drop("target", axis=1)
y_test = df_test["target"]

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -----------------------------
# Build Improved Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=350,
    max_depth=14,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

r2_train = metrics.r2_score(y_train, pred_train)
r2_test = metrics.r2_score(y_test, pred_test)
mae = metrics.mean_absolute_error(y_test, pred_test)
rmse = np.sqrt(metrics.mean_squared_error(y_test, pred_test))

print("\n----- MODEL PERFORMANCE -----")
print("R² (Train):", round(r2_train, 4))
print("R² (Test): ", round(r2_test, 4))
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))

# -----------------------------
# Feature Importance
# -----------------------------
importances = model.feature_importances_
feat_imp_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nTop 10 Features Used in Model:")
print(feat_imp_df)

# -----------------------------
# Save Model
# -----------------------------
with open("plant_health_model_top10.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved as: plant_health_model_top10.pkl")
