
import pandas as pd
import numpy as np
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
    roc_auc_score
)

Path("models").mkdir(exist_ok=True)

# 1. Load matched 2022-2023 dataset
df = pd.read_parquet(
    "data/doctor_growth_2022_2023.parquet"
)

# 2. Use ONLY 2022 features
features = [
    "total_claims_2022",
    "total_cost_2022",
    "total_fills_2022",
    "total_days_supply_2022",
    "unique_drugs_2022",
    "drug_records_2022",
    "cost_per_claim_2022",
    "days_per_claim_2022",
    "fills_per_claim_2022"
]

X = df[features].copy()
y = df["rising_prescriber"]

# Log transform skewed volume/cost features
log_features = [
    "total_claims_2022",
    "total_cost_2022",
    "total_fills_2022",
    "total_days_supply_2022",
    "unique_drugs_2022",
    "drug_records_2022",
    "cost_per_claim_2022"
]

X[log_features] = np.log1p(
    X[log_features].clip(lower=0)
)

# 3. Holdout split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
}

def ranking_metrics(y_true, scores):
    ranking = pd.DataFrame({
        "actual": np.asarray(y_true),
        "score": scores
    }).sort_values("score", ascending=False)

    top_n = max(1, int(np.ceil(len(ranking) * 0.10)))
    top = ranking.head(top_n)

    precision_at_10 = top["actual"].mean()
    recall_at_10 = (
        top["actual"].sum() / ranking["actual"].sum()
    )

    baseline = ranking["actual"].mean()
    lift_at_10 = precision_at_10 / baseline

    return precision_at_10, recall_at_10, lift_at_10


results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    p10, r10, lift10 = ranking_metrics(
        y_test, probabilities
    )

    row = {
        "Model": name,
        "Precision": precision_score(
            y_test, predictions
        ),
        "Recall": recall_score(
            y_test, predictions
        ),
        "F1": f1_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(
            y_test, probabilities
        ),
        "Precision@10%": p10,
        "Recall@10%": r10,
        "Lift@10%": lift10
    }

    results.append(row)

    print("\nMODEL:", name)
    for metric, value in row.items():
        if metric != "Model":
            print(f"{metric}: {value:.4f}")

# Save results
results_df = pd.DataFrame(results)

results_df.to_csv(
    "models/growth_model_results.csv",
    index=False
)

# Save both fitted models and transformation details
for name, filename in [
    ("Random Forest", "growth_random_forest.joblib"),
    ("Logistic Regression", "growth_logistic_regression.joblib")
]:
    joblib.dump(
        {
            "model": models[name],
            "features": features,
            "log_features": log_features,
            "growth_threshold": 0.20
        },
        f"models/{filename}"
    )

print("\nSaved growth models and metrics.")
