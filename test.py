import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("Indoor_Plant_Health_and_Growth_Factors.csv")
print(f" Shape before cleaning: {df.shape}")
print("\n Columns:\n", df.columns)

#  Handle missing values
print("\n Checking for missing values...")
missing = df.isnull().sum()
print(missing[missing > 0])

for col in ['Room_Temperature_C', 'Humidity_%', 'Soil_Moisture_%', 'Watering_Amount_ml', 'Health_Score']:
      df[col] = df[col].fillna(df[col].mean())
print(" Missing values handled successfully!")

#  Drop irrelevant columns
if "Health_Notes" in df.columns:
    df.drop(columns=["Health_Notes"], inplace=True)
if "Plant_ID" in df.columns:
    df.drop(columns=["Plant_ID"], inplace=True)
print("\n \n Columns After Drop irrelevant columns:\n", df.columns.tolist(),"\n")

#  Encode categorical columns 
cat_cols = df.select_dtypes(include="object").columns
encoder = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])
print(f" Encoded categorical columns: {list(cat_cols)}")

# Standardize numerical columns 
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
print(" Applied Standardization to numerical columns.")

#  Save cleaned dataset
output_path = "Cleaned_Indoor_Plant_Data.csv"
df.to_csv(output_path, index=False)
print(f"\n Cleaned dataset saved as: {output_path}")

print(f"\n Shape after cleaning: {df.shape}\n")