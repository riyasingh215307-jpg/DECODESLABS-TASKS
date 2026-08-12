# ==========================================
# Decode Labs - Project 3
# Customer Segmentation using PCA & K-Means
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


print("All libraries imported successfully!")
# Load Dataset
df = pd.read_excel("dataset/Dataset for Data Analytics.xlsx")

# Display first 5 rows
print(df.head())
# ==========================================
# Save Dataset Preview
# ==========================================

# Save first 5 rows
df.head().to_csv("outputs/01_dataset/dataset_preview.csv", index=False)

# Save dataset shape
with open("outputs/01_dataset/dataset_shape.txt", "w") as f:
    f.write(f"Rows: {df.shape[0]}\n")
    f.write(f"Columns: {df.shape[1]}")

# Save data types
df.dtypes.to_csv("outputs/01_dataset/data_types.csv")

# Save summary statistics
df.describe(include="all").to_csv("outputs/01_dataset/summary_statistics.csv")

print("Dataset files saved successfully!")
# ==========================================
# Data Inspection
# ==========================================

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

# Save missing values
df.isnull().sum().to_csv("outputs/02_cleaning/missing_values.csv")

# Save duplicate count
with open("outputs/02_cleaning/duplicate_records.txt", "w") as f:
    f.write(f"Duplicate Records: {df.duplicated().sum()}")

print("Cleaning files saved successfully!")
# ==========================================
# Data Cleaning
# ==========================================

# Remove duplicate records
df = df.drop_duplicates()

# Fill missing numeric values with the median
numeric_columns = df.select_dtypes(include=['number']).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

# Save cleaned dataset
df.to_csv("outputs/02_cleaning/cleaned_dataset.csv", index=False)

print("Cleaned dataset saved successfully!")
# ==========================================
# Histogram
# ==========================================

import os

# Create folder if it doesn't exist
os.makedirs("outputs/03_eda", exist_ok=True)

# Plot histograms
df.hist(figsize=(12, 10))

plt.tight_layout()

plt.savefig("outputs/03_eda/histograms.png")

plt.show()

print("Histogram saved successfully!")
# ==========================================
# Boxplots
# ==========================================

numeric_columns = df.select_dtypes(include=['number']).columns

for column in numeric_columns:
    plt.figure(figsize=(8, 4))
    plt.boxplot(df[column].dropna(), vert=False)
    plt.title(f"Boxplot - {column}")
    plt.xlabel(column)

    plt.savefig(f"outputs/03_eda/boxplot_{column}.png")
    plt.close()

print("All boxplots saved successfully!")
# ==========================================
# Correlation Heatmap
# ==========================================

correlation_matrix = df.select_dtypes(include=['number']).corr()

plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap="coolwarm", aspect="auto")
plt.colorbar()

plt.xticks(range(len(correlation_matrix.columns)),
           correlation_matrix.columns,
           rotation=90)

plt.yticks(range(len(correlation_matrix.columns)),
           correlation_matrix.columns)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("outputs/03_eda/correlation_heatmap.png")

plt.show()

print("Correlation Heatmap saved successfully!")
# ==========================================
# Feature Scaling
# ==========================================

# Select only numeric columns
numeric_data = df.select_dtypes(include=['number'])

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Save scaled dataset
scaled_df = pd.DataFrame(scaled_data, columns=numeric_data.columns)
scaled_df.to_csv("outputs/04_scaling/scaled_dataset.csv", index=False)

print("Feature Scaling completed successfully!")
# ==========================================
# Principal Component Analysis (PCA)
# ==========================================

# Apply PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Create DataFrame
pca_df = pd.DataFrame(pca_data, columns=["PC1", "PC2"])

# Save PCA dataset
pca_df.to_csv("outputs/05_pca/pca_dataset.csv", index=False)

print("PCA completed successfully!")
# ==========================================
# Explained Variance Plot
# ==========================================

explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(8,5))
plt.bar(
    range(1, len(explained_variance)+1),
    explained_variance
)

plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance by Principal Components")

plt.savefig("outputs/05_pca/explained_variance.png")

plt.show()

print("Explained Variance Plot saved successfully!")# ==========================================
# Cumulative Variance Plot
# ==========================================

cumulative_variance = pca.explained_variance_ratio_.cumsum()

plt.figure(figsize=(8,5))

plt.plot(
    range(1, len(cumulative_variance)+1),
    cumulative_variance,
    marker='o'
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance")

plt.grid(True)

plt.savefig("outputs/05_pca/cumulative_variance.png")

plt.show()

print("Cumulative Variance Plot saved successfully!")
# ==========================================
# PART 1 - PCA VISUALIZATION & ELBOW METHOD
# ==========================================

import os

# Create folders
os.makedirs("outputs/05_pca", exist_ok=True)
os.makedirs("outputs/06_clustering", exist_ok=True)

# -----------------------------
# PCA Scatter Plot
# -----------------------------
plt.figure(figsize=(8,6))
plt.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Scatter Plot")
plt.grid(True)

plt.savefig("outputs/05_pca/pca_scatter_plot.png")
plt.show()

# Save PCA summary
with open("outputs/05_pca/pca_summary.txt","w") as f:
    f.write("Explained Variance Ratio:\n")
    for i,v in enumerate(pca.explained_variance_ratio_):
        f.write(f"PC{i+1}: {v:.4f}\n")

# -----------------------------
# Elbow Method
# -----------------------------
inertia = []

for k in range(1,11):
    model = KMeans(n_clusters=k,
                   random_state=42,
                   n_init=10)
    model.fit(scaled_data)
    inertia.append(model.inertia_)

# Save inertia values
pd.DataFrame({
    "Clusters":range(1,11),
    "Inertia":inertia
}).to_csv(
    "outputs/06_clustering/inertia_values.csv",
    index=False
)

# Elbow Plot
plt.figure(figsize=(8,5))
plt.plot(range(1,11),
         inertia,
         marker="o")

plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(True)

plt.savefig("outputs/06_clustering/elbow_method.png")
plt.show()

print("PART 1 COMPLETED SUCCESSFULLY!")# ==========================================
# PART 2 - SILHOUETTE SCORE & K-MEANS
# ==========================================

from sklearn.metrics import silhouette_score

# -----------------------------
# Silhouette Scores
# -----------------------------
silhouette_scores = []


for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_data)
    score = silhouette_score(scaled_data, labels)
    silhouette_scores.append(score)

import os
print(os.getcwd())

# Save silhouette scores
import os

os.makedirs("outputs/07_evaluation", exist_ok=True)

pd.DataFrame({
    "Clusters": range(2,11),
    "Silhouette Score": silhouette_scores
}).to_csv(
    "outputs/07_evaluation/silhouette_scores.csv",
    index=False
)

# Plot Silhouette Scores
plt.figure(figsize=(8,5))
plt.plot(range(2,11), silhouette_scores, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score")
plt.grid(True)

plt.savefig("outputs/07_evaluation/silhouette_score_plot.png")
plt.show()

# -----------------------------
# Best K (Highest Silhouette Score)
# -----------------------------
best_k = range(2,11)[np.argmax(silhouette_scores)]

print(f"Best Number of Clusters = {best_k}")

# -----------------------------
# Final KMeans Model
# -----------------------------
kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels
df["Cluster"] = clusters

# Save clustered dataset
df.to_csv(
    "outputs/06_clustering/clustered_dataset.csv",
    index=False
)
print(BASE_DIR)
print(BASE_DIR / "outputs")
print((BASE_DIR / "outputs" / "07_evaluation").exists())

# Save cluster centers
pd.DataFrame(
    kmeans.cluster_centers_,
    columns=numeric_data.columns
).to_csv(
    "outputs/06_clustering/cluster_centers.csv",
    index=False
)

# Cluster Scatter Plot
plt.figure(figsize=(8,6))

plt.scatter(
    pca_df["PC1"],
    pca_df["PC2"],
    c=clusters,
    cmap="viridis"
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Customer Clusters")

plt.colorbar(label="Cluster")

plt.savefig(
    "outputs/06_clustering/customer_clusters.png"
)

plt.show()

# Cluster Distribution
cluster_counts = df["Cluster"].value_counts().sort_index()

plt.figure(figsize=(7,5))
cluster_counts.plot(kind="bar")

plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.title("Cluster Distribution")

plt.savefig(
    "outputs/06_clustering/cluster_distribution.png"
)

plt.show()

print("PART 2 COMPLETED SUCCESSFULLY!")
# ==========================================
# PART 3 - BUSINESS INSIGHTS & REPORT
# ==========================================

import os

os.makedirs("outputs/08_business_insights", exist_ok=True)
os.makedirs("outputs/09_report", exist_ok=True)

# -----------------------------
# Cluster Summary
# -----------------------------
cluster_summary = df.groupby("Cluster").mean(numeric_only=True)

cluster_summary.to_csv(
    "outputs/08_business_insights/cluster_summary.csv"
)

# -----------------------------
# Customer Personas
# -----------------------------
personas = []

for cluster in cluster_summary.index:
    personas.append({
        "Cluster": cluster,
        "Persona": f"Customer Group {cluster}",
        "Recommendation":
        "Design personalized marketing campaigns for this customer segment."
    })

persona_df = pd.DataFrame(personas)

persona_df.to_csv(
    "outputs/08_business_insights/customer_personas.csv",
    index=False
)

# -----------------------------
# Business Insights
# -----------------------------
with open(
    "outputs/08_business_insights/business_insights.txt",
    "w"
) as f:

    f.write("BUSINESS INSIGHTS\n")
    f.write("=============================\n\n")

    for cluster in cluster_summary.index:
        f.write(f"Cluster {cluster}\n")
        f.write(
            "Recommendation: Personalized offers and targeted marketing.\n\n"
        )

# -----------------------------
# README
# -----------------------------
with open(
    "outputs/09_report/README.txt",
    "w"
) as f:

    f.write("Decode Labs Project 3\n")
    f.write("=============================\n\n")
    f.write("Workflow:\n")
    f.write("- Data Cleaning\n")
    f.write("- Feature Scaling\n")
    f.write("- PCA\n")
    f.write("- Elbow Method\n")
    f.write("- Silhouette Score\n")
    f.write("- K-Means Clustering\n")
    f.write("- Customer Personas\n")
    f.write("- Business Insights\n")

# -----------------------------
# Final Report
# -----------------------------
with open(
    "outputs/09_report/final_report.txt",
    "w"
) as f:

    f.write("PROJECT 3 REPORT\n\n")

    f.write("Objective:\n")
    f.write(
        "Customer Segmentation using PCA and K-Means Clustering.\n\n"
    )

    f.write(f"Optimal Number of Clusters: {best_k}\n\n")

    f.write("Deliverables:\n")
    f.write("- Cleaned Dataset\n")
    f.write("- PCA\n")
    f.write("- Elbow Method\n")
    f.write("- Silhouette Score\n")
    f.write("- Cluster Visualization\n")
    f.write("- Customer Personas\n")
    f.write("- Business Insights\n")

print("PROJECT 3 COMPLETED SUCCESSFULLY!")