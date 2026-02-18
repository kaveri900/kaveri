import pandas as pd

# Sample dataset
data = {
    "Transmission": ["Automatic", "Manual", "Automatic", "Manual", "Automatic"],
    "Color": ["Red", "Blue", "Green", "Red", "Blue"]
}

df = pd.DataFrame(data)
print("Original Data:")
print(df)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["Transmission_encoded"] = le.fit_transform(df["Transmission"])

print("\nAfter Label Encoding Transmission:")
print(df)
df_encoded = pd.get_dummies(df, columns=["Color"], drop_first=True)

print("\nAfter One-Hot Encoding Color:")
print(df_encoded)
