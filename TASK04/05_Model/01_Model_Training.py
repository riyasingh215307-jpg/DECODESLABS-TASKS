import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the processed dataset
file_path = "03_Text_Processing/processed_text.csv"
df = pd.read_csv(file_path)

print("Processed dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Separate text and target
X = df["Cleaned_Text"]
y = df["Sentiment"]

print("\nTarget distribution:")
print(y.value_counts())
 # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain-test split completed!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2)
)

# Fit only on training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the test data using the same vectorizer
X_test_tfidf = vectorizer.transform(X_test)

print("\nModel TF-IDF transformation completed!")
print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)
from sklearn.naive_bayes import MultinomialNB

# Create and train the Naive Bayes classifier
model = MultinomialNB(alpha=1.0)

model.fit(X_train_tfidf, y_train)

print("\nMultinomial Naive Bayes model trained successfully!")