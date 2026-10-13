
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
    ConfusionMatrixDisplay
)

# -----------------------------------
# 1. Create folders
# -----------------------------------

Path("charts").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

# -----------------------------------
# 2. Load dataset
# -----------------------------------

df = pd.read_parquet(
    "data/segmented_doctors.parquet"
)

print("Dataset shape:", df.shape)

# -----------------------------------
# 3. Define target variable
# -----------------------------------

# Top 20% by observed drug cost
threshold = df["total_drug_cost"].quantile(0.80)

df["high_potential"] = (
    df["total_drug_cost"] >= threshold
).astype(int)

print("\nCost threshold:", round(threshold, 2))

print("\nTarget distribution:")
print(df["high_potential"].value_counts())

# -----------------------------------
# 4. Select features
# -----------------------------------

# Avoid label leakage:
# Exclude total_drug_cost, cost_per_claim,
# and segment (created using drug cost).

features = [
    "total_claims",
    "total_fills",
    "total_days_supply",
    "unique_drugs",
    "drug_records",
    "days_per_claim",
    "fills_per_claim"
]

X = df[features].copy()
y = df["high_potential"]

# -----------------------------------
# 5. Train-test split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# -----------------------------------
# 6. Define models
# -----------------------------------

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),

    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000)
    )
}

# -----------------------------------
# 7. Train and evaluate
# -----------------------------------

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": auc
    })

    print("\nMODEL:", name)
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

# Save metrics for README/dashboard
results_df = pd.DataFrame(results)
results_df.to_csv(
    "models/model_results.csv",
    index=False
)

# -----------------------------------
# 8. Select Random Forest
# -----------------------------------

rf = models["Random Forest"]

# Already trained in the loop above

# -----------------------------------
# 9. ROC curve
# -----------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

RocCurveDisplay.from_estimator(
    rf,
    X_test,
    y_test,
    ax=ax,
    name="Random Forest"
)

plt.title("ROC Curve - High-Spending Prescribers")
plt.tight_layout()

plt.savefig(
    "charts/roc_curve.png",
    dpi=150
)

plt.close()

# -----------------------------------
# 10. Confusion matrix
# -----------------------------------

ConfusionMatrixDisplay.from_estimator(
    rf,
    X_test,
    y_test,
    cmap="Blues",
    values_format="d"
)

plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "charts/confusion_matrix.png",
    dpi=150
)

plt.close()

# -----------------------------------
# 11. Feature importance
# -----------------------------------

importance = pd.Series(
    rf.feature_importances_,
    index=features
).sort_values()

plt.figure(figsize=(8, 5))

importance.plot(kind="barh")

plt.xlabel("Feature Importance")
plt.title("Random Forest - Feature Importance")

plt.tight_layout()

plt.savefig(
    "charts/feature_importance.png",
    dpi=150
)

plt.close()

print("\nFEATURE IMPORTANCE:")
print(
    importance.sort_values(
        ascending=False
    ).round(4)
)

# -----------------------------------
# 12. Save trained model
# -----------------------------------

joblib.dump(
    {
        "model": rf,
        "features": features,
        "cost_threshold": float(threshold)
    },
    "models/random_forest.joblib"
)

print("\nSUCCESS!")
print("Saved model: models/random_forest.joblib")
print("Saved metrics: models/model_results.csv")
print("Saved charts:")
print(" - charts/roc_curve.png")
print(" - charts/confusion_matrix.png")
print(" - charts/feature_importance.png")
