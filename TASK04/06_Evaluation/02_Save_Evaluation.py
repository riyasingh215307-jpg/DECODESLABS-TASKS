import pandas as pd

# Save evaluation results
results = {
    "Metric": [
        "Accuracy",
        "Negative Precision",
        "Negative Recall",
        "Negative F1-Score",
        "Positive Precision",
        "Positive Recall",
        "Positive F1-Score"
    ],
    "Value": [
        0.49,
        0.50,
        0.56,
        0.53,
        0.46,
        0.41,
        0.43
    ]
}

results_df = pd.DataFrame(results)

results_df.to_csv(
    "06_Evaluation/evaluation_results.csv",
    index=False
)

print("Evaluation results saved successfully!")