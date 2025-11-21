import pandas as pd
import numpy as np
from sklearn.utils import resample

# Load original data
DATA_PATH = "Indoor_Plant_Health_and_Growth_Factors.csv"
df = pd.read_csv(DATA_PATH)

print("Original Shape:", df.shape)

# Drop unnecessary columns
columns_to_drop = ['Plant_ID', 'Health_Notes']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Handle missing values
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# Augment data: Add more synthetic samples
n_aug = 2000  # Add 2000 more rows for total 3000
df_aug = resample(df, replace=True, n_samples=n_aug, random_state=42)

# Add small noise to numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    noise = np.random.normal(0, df[col].std() * 0.05, size=df_aug.shape[0])  # Reduced noise to 5%
    df_aug[col] += noise
    if col in ['Height_cm', 'Leaf_Count', 'New_Growth_Count']:
        df_aug[col] = df_aug[col].clip(lower=0)
    elif col.endswith('%') or col.endswith('_ml'):
        df_aug[col] = df_aug[col].clip(lower=0)

# Combine
df_combined = pd.concat([df, df_aug], ignore_index=True)
print("Augmented Shape:", df_combined.shape)

# Save
df_combined.to_csv("augmented_more.csv", index=False)
print("Augmented data saved as augmented_more.csv")
