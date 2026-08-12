import pandas as pd
import numpy as np

print("Libraries imported successfully!")
# Load the dataset
file_path = "01_Dataset/Dataset.xlsx"

df = pd.read_excel(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)# Display column names
print("\nColumn names:")
print(df.columns.tolist())# Check data types and missing values
print("\nDataset information:")
print(df.info())
# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Examine important categorical values

print("\nOrder Status values:")
print(df["OrderStatus"].value_counts())

print("\nPayment Method values:")
print(df["PaymentMethod"].value_counts())

print("\nReferral Source values:")
print(df["ReferralSource"].value_counts())

print("\nProduct values:")
print(df["Product"].value_counts())