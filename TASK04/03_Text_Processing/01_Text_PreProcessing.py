import pandas as pd
import numpy as np
import nltk
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Download required NLTK resources
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger_eng")

print("NLTK resources are ready!")
# Load the cleaned dataset
file_path = "02_Data_PreProcessing/cleaned_dataset.csv"
df = pd.read_csv(file_path)

print("Cleaned dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
# Create a combined text field from available categorical information
text_columns = [
    "Product",
    "PaymentMethod",
    "ReferralSource",
    "CouponCode"
]

# Convert values to text and handle missing values
for column in text_columns:
    df[column] = df[column].fillna("unknown").astype(str)

# Combine the selected columns into one text field
df["Text"] = df[text_columns].agg(" ".join, axis=1)

print("\nText column created successfully!")
print("\nSample text:")
print(df["Text"].head())
# Tokenize the text
df["Tokens"] = df["Text"].apply(word_tokenize)

print("\nTokenization completed!")
print("\nSample tokens:")
print(df["Tokens"].head())
# Remove stop words
stop_words = set(stopwords.words("english"))
negation_words = {"not", "no", "never", "neither", "nor"}
stop_words = stop_words - negation_words


df["Filtered_Tokens"] = df["Tokens"].apply(
    lambda tokens: [word for word in tokens if word.lower() not in stop_words]
)

print("\nStop-word removal completed!")

print("\nSample filtered tokens:")
print(df["Filtered_Tokens"].head())
# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Lemmatize the filtered tokens
def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()

    if tag == "J":
        return wordnet.ADJ
    elif tag == "V":
        return wordnet.VERB
    elif tag == "N":
        return wordnet.NOUN
    elif tag == "R":
        return wordnet.ADV
    else:
        return wordnet.NOUN


df["Lemmatized_Tokens"] = df["Filtered_Tokens"].apply(
    lambda tokens: [
        lemmatizer.lemmatize(word.lower(), get_wordnet_pos(word))
        for word in tokens
    ]
)

print("\nLemmatization completed!")
print("\nSample lemmatized tokens:")
print(df["Lemmatized_Tokens"].head())
# Convert lemmatized tokens back into text
df["Cleaned_Text"] = df["Lemmatized_Tokens"].apply(
    lambda tokens: " ".join(tokens)
)

print("\nCleaned text created successfully!")
print("\nSample cleaned text:")
print(df["Cleaned_Text"].head())
# Save the processed text dataset
output_path = "03_Text_Processing/processed_text.csv"

df.to_csv(output_path, index=False)

print("\nProcessed text dataset saved successfully!")
print("Saved file:", output_path)
print("Final shape:", df.shape)


