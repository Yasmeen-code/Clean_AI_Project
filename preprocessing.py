import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_PATH = "Indoor_Plant_Health_and_Growth_Factors.csv" 

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print("\nColumns and types:\n")
print(df.dtypes)
print("\nFirst 5 rows:\n")
print(df.head())

print("\nMissing values per column:\n")
print(df.isna().sum())

# --------- Drop unnecessary columns ----------
columns_to_drop = ['Plant_ID', 'Health_Notes']

df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

print(f"\nDropped columns: {columns_to_drop}")
print("Shape after dropping columns:", df.shape)

# --------- Handling Missing Values ----------

for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        mean_value = df[col].mean()
        df[col] = df[col].fillna(mean_value)
        print(f"Filled numeric column '{col}' with mean: {mean_value}")

    elif pd.api.types.is_object_dtype(df[col]):
        mode_value = df[col].mode()[0]  
        df[col] = df[col].fillna(mode_value)
        print(f"Filled object column '{col}' with mode: {mode_value}")

print("\nAfter filling missing values:\n")
print(df.isna().sum())

# --------- Removing Outliers using IQR ----------

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

def remove_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    before_count = data.shape[0]
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    after_count = data.shape[0]

    print(f"Column '{col}': removed {before_count - after_count} outliers")
    
    return data

# Apply outlier removal on each numeric column
for col in numeric_cols:
    df = remove_outliers_iqr(df, col)

print("\nShape after removing outliers:", df.shape)

# --------- Encoding Categorical Columns (One-Hot Encoding) ----------

object_columns = df.select_dtypes(include=['object']).columns

print("\nObject columns to encode:", list(object_columns))

df = pd.get_dummies(df, columns=object_columns, drop_first=True)

print("\nShape after encoding:", df.shape)
# 1) تحديد الأعمدة الرقمية بعد الـ Encoding
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# 2) تطبيق StandardScaler
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\nScaling completed. Numeric columns are standardized.")
# --------- Save cleaned dataset to a new file ----------
OUTPUT_PATH = "Indoor_Plant_Health_and_Growth_Factors_CLEANED.csv"

df.to_csv(OUTPUT_PATH, index=False)

print(f"\nCleaned dataset saved successfully as: {OUTPUT_PATH}")
