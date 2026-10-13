
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split

Path("charts").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

# Load matched dataset and saved Random Forest
df = pd.read_parquet(
    "data/doctor_growth_2022_2023.parquet"
)

bundle = joblib.load(
    "models/growth_random_forest.joblib"
)

model = bundle["model"]
features = bundle["features"]
log_features = bundle["log_features"]

X = df[features].copy()
y = df["rising_prescriber"]

X[log_features] = np.log1p(
    X[log_features].clip(lower=0)
)

# Reproduce exact holdout split from growth_model.py
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scores = model.predict_proba(X_test)[:, 1]

ranking = pd.DataFrame({
    "actual": y_test.to_numpy(),
    "score": scores
}).sort_values(
    "score", ascending=False
).reset_index(drop=True)

# Decile 1 = highest-ranked 10%
ranking["decile"] = (
    np.floor(
        np.arange(len(ranking)) * 10 / len(ranking)
    ).astype(int) + 1
)

deciles = ranking.groupby(
    "decile"
).agg(
    doctors=("actual", "size"),
    rising_doctors=("actual", "sum"),
    precision=("actual", "mean")
).reset_index()

baseline = ranking["actual"].mean()

deciles["lift"] = (
    deciles["precision"] / baseline
)

deciles.to_csv(
    "models/growth_decile_results.csv",
    index=False
)

# Plot lift by decile
plt.figure(figsize=(9, 5))

plt.bar(
    deciles["decile"].astype(str),
    deciles["lift"]
)

plt.axhline(
    1.0,
    color="red",
    linestyle="--",
    label="Random Selection Baseline"
)

plt.xlabel("Decile (1 = Highest Predicted Growth)")
plt.ylabel("Lift")
plt.title("Rising Prescriber Prediction - Lift by Decile")
plt.legend()
plt.tight_layout()

plt.savefig(
    "charts/growth_lift_deciles.png",
    dpi=150
)

plt.close()

print("\nDECILE RESULTS:")
print(deciles.round(4).to_string(index=False))

print("\nBaseline rising rate:", round(baseline, 4))

print("\nSaved:")
print("charts/growth_lift_deciles.png")
print("models/growth_decile_results.csv")
