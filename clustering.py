
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path

Path("charts").mkdir(exist_ok=True)

df = pd.read_parquet("data/doctor_features.parquet")

# Select non-redundant behavioral features
features = [
    "total_claims",
    "total_drug_cost",
    "unique_drugs",
    "cost_per_claim",
    "days_per_claim",
    "fills_per_claim"
]

# Reduce the effect of extreme values
X = df[features].copy()

for col in ["total_claims", "total_drug_cost",
            "unique_drugs", "cost_per_claim"]:
    X[col] = np.log1p(X[col])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compare cluster counts
inertias = []
silhouettes = []

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    inertias.append(model.inertia_)
    score = silhouette_score(
        X_scaled, labels,
        sample_size=min(3000, len(df)),
        random_state=42
    )
    silhouettes.append(score)

    print(f"K={k}: Inertia={model.inertia_:.2f}, "
          f"Silhouette={score:.4f}")

# Elbow plot
plt.figure(figsize=(8, 5))
plt.plot(range(2, 9), inertias, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method — Prescriber Segmentation")
plt.xticks(range(2, 9))
plt.tight_layout()
plt.savefig("charts/elbow_plot.png", dpi=150)
plt.close()

print("\nBest silhouette K:",
      range(2, 9)[np.argmax(silhouettes)])

print("Saved: charts/elbow_plot.png")
