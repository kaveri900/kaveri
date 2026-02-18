import pandas as pd

data = {
    "Age": [22, 25, 30, 35, 40],
    "Salary": [30000, 50000, 70000, 90000, 110000]
}

df = pd.DataFrame(data)
print("Original Data:")
print(df)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_standardized = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns
)

print("\nStandardized Data (Z-score):")
print(df_standardized)
from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler()
df_normalized = pd.DataFrame(
    mm.fit_transform(df),
    columns=df.columns
)

print("\nNormalized Data (0 to 1):")
print(df_normalized)
