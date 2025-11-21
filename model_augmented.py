import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import pickle

# Load augmented data
df = pd.read_csv("augmented_data.csv")
print("Augmented Shape:", df.shape)

TARGET = "Health_Score"
X = df.drop(TARGET, axis=1)
y = df[TARGET]

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False, drop='first')
X_cat = encoder.fit_transform(X[categorical_features])
cat_cols = encoder.get_feature_names_out(categorical_features)
df_cat = pd.DataFrame(X_cat, columns=cat_cols, index=X.index)
X_full = pd.concat([X[numeric_features], df_cat], axis=1)

# Scale Numeric
scaler = StandardScaler()
X_full[numeric_features] = scaler.fit_transform(X_full[numeric_features])

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [15, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Evaluate
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

# Feature Importance
feat_imp = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_}).sort_values(by='importance', ascending=False).head(10)
print("Top 10 Feature Importances:\n", feat_imp)

# Save
with open("augmented_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("augmented_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("augmented_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
print("Model, Scaler, and Encoder saved.")
