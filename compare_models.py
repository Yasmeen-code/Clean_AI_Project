import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import xgboost as xgb
import pickle

# Load data (reuse preprocessing from final_model.py)
DATA_PATH = "Indoor_Plant_Health_and_Growth_Factors.csv"
df = pd.read_csv(DATA_PATH)

# Quick preprocessing (same as final_model.py)
columns_to_drop = ['Plant_ID', 'Health_Notes']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# Outlier removal (simplified, no print)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

TARGET = "Health_Score"
X = df.drop(TARGET, axis=1)
y = df[TARGET]

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

encoder = OneHotEncoder(sparse_output=False, drop='first')
X_cat = encoder.fit_transform(X[categorical_features])
cat_cols = encoder.get_feature_names_out(categorical_features)
df_cat = pd.DataFrame(X_cat, columns=cat_cols, index=X.index)
X_full = pd.concat([X[numeric_features], df_cat], axis=1)

scaler = StandardScaler()
X_full[numeric_features] = scaler.fit_transform(X_full[numeric_features])

# Feature selection (top 10)
X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(X_full, y, test_size=0.2, random_state=42)
rf_tmp = RandomForestRegressor(n_estimators=200, random_state=42)
rf_tmp.fit(X_train_tmp, y_train_tmp)
importances = rf_tmp.feature_importances_
feat_df = pd.DataFrame({'feature': X_full.columns, 'importance': importances}).sort_values(by='importance', ascending=False)
top_10_features = feat_df.head(10)['feature'].tolist()

X = X_full[top_10_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Model 1: Tuned RandomForest
rf = RandomForestRegressor(random_state=42)
param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [10, 15]}
grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, scoring='r2')
grid_rf.fit(X_train, y_train)
rf_model = grid_rf.best_estimator_
rf_pred = rf_model.predict(X_test)
rf_r2 = metrics.r2_score(y_test, rf_pred)
rf_mae = metrics.mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(metrics.mean_squared_error(y_test, rf_pred))

print("RandomForest - R²:", round(rf_r2, 4), "MAE:", round(rf_mae, 4), "RMSE:", round(rf_rmse, 4))

# Model 2: XGBoost
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
param_grid_xgb = {'n_estimators': [100, 200], 'max_depth': [3, 6], 'learning_rate': [0.1, 0.2]}
grid_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=3, scoring='r2')
grid_xgb.fit(X_train, y_train)
xgb_best = grid_xgb.best_estimator_
xgb_pred = xgb_best.predict(X_test)
xgb_r2 = metrics.r2_score(y_test, xgb_pred)
xgb_mae = metrics.mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(metrics.mean_squared_error(y_test, xgb_pred))

print("XGBoost - R²:", round(xgb_r2, 4), "MAE:", round(xgb_mae, 4), "RMSE:", round(xgb_rmse, 4))

# Compare and save best
if xgb_r2 > rf_r2:
    best_model = xgb_best
    print("XGBoost performs better.")
else:
    best_model = rf_model
    print("RandomForest performs better.")

with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("Best model saved as best_model.pkl")
