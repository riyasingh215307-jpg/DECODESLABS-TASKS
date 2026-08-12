import pandas as pd
import numpy as np

# Load the dataset
file_path = "01_Dataset/Dataset.xlsx"
df = pd.read_excel(file_path)

print("Dataset loaded successfully!")
print("Original shape:", df.shape)
# Create a derived sentiment proxy from OrderStatus
df["Sentiment"] = df["OrderStatus"].map({
    "Delivered": "Positive",
    "Shipped": "Positive",
    "Cancelled": "Negative",
    "Returned": "Negative"
})

# Remove Pending orders because they are neither clearly positive nor negative
df = df.dropna(subset=["Sentiment"]).copy()

print("\nSentiment distribution:")
print(df["Sentiment"].value_counts())# Save the cleaned dataset
output_path = "02_Data_Preprocessing/cleaned_dataset.csv"

df.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully!")
print("Saved file:", output_path)
print("Final shape:", df.shape)
