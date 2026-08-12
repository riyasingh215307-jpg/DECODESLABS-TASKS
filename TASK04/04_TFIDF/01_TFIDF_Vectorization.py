import pandas as pd
import numpy as np

# Load the processed text dataset
file_path = "03_Text_Processing/processed_text.csv"
df = pd.read_csv(file_path)

print("Processed text dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nAvailable columns:")
print(df.columns.tolist())
from sklearn.feature_extraction.text import TfidfVectorizer

# Apply TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=1000)

tfidf_matrix = vectorizer.fit_transform(df["Cleaned_Text"])

print("\nTF-IDF vectorization completed successfully!")
print("TF-IDF matrix shape:", tfidf_matrix.shape)
print("Number of features:", len(vectorizer.get_feature_names_out()))
# Save TF-IDF matrix
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=vectorizer.get_feature_names_out()
)

tfidf_df.to_csv("04_TFIDF/tfidf_features.csv", index=False)

print("TF-IDF features saved successfully!")