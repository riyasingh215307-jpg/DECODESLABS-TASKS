# NLP Sentiment Analysis Project

## 1. Introduction

This project focuses on sentiment analysis using Natural Language Processing (NLP) techniques.

## 2. Dataset

The project uses the provided dataset containing customer/order-related information.

## 3. Data Preprocessing

The dataset was loaded and examined for:
- Missing values
- Column information
- Categorical values
- Data types

Text-related columns were combined into a single `Text` column for further NLP processing.

## 4. Text Preprocessing

The text data was processed using NLP techniques including:
- Tokenization
- Stop-word handling
- Lemmatization
- Text cleaning

## 5. TF-IDF Vectorization

TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert the processed text into numerical features suitable for machine learning.

## 6. Model Training

A machine learning classification model was trained using the TF-IDF features.

## 7. Model Evaluation


The model achieved an accuracy of 49%.

Classification results:
- Negative: Precision = 0.50, Recall = 0.56, F1-score = 0.53
- Positive: Precision = 0.46, Recall = 0.41, F1-score = 0.43

Confusion Matrix:
[[56, 44],
 [55, 38]]
 
## 8. Conclusion

The project demonstrates a complete NLP workflow, from data loading and text preprocessing to TF-IDF feature extraction, model training, and evaluation.