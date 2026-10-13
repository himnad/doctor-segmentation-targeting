
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

df = pd.read_parquet("data/doctor_features.parquet")

features = [
    "total_claims",
    "total_drug_cost",
    "unique_drugs",
    "cost_per_claim",
    "days_per_claim",
    "fills_per_claim"
]

X = df[features].copy()

for col in [
    "total_claims",
    "total_drug_cost",
    "unique_drugs",
    "cost_per_claim"
]:
    X[col] = np.log1p(X[col])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["segment"] = model.fit_predict(X_scaled)

# Segment statistics
profile = df.groupby("segment").agg(
    doctors=("Prescriber NPI", "count"),
    avg_claims=("total_claims", "mean"),
    avg_cost=("total_drug_cost", "mean"),
    avg_drugs=("unique_drugs", "mean"),
    avg_cost_per_claim=("cost_per_claim", "mean")
).round(2)

print("SEGMENT PROFILE:")
print(profile.to_string())

print("\nSEGMENT COUNTS:")
print(df["segment"].value_counts().sort_index())

# Segment profile chart
profile[[
    "avg_claims",
    "avg_drugs",
    "avg_cost_per_claim"
]].plot(
    kind="bar",
    subplots=True,
    figsize=(10, 9),
    layout=(3, 1),
    legend=False,
    sharex=True
)

plt.suptitle("Prescriber Segment Profiles")
plt.tight_layout()
plt.savefig("charts/segment_profiles.png", dpi=150)
plt.close()

# PCA: project 6 features into 2 dimensions
pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)

plt.figure(figsize=(9, 6))

scatter = plt.scatter(
    coords[:, 0],
    coords[:, 1],
    c=df["segment"],
    cmap="tab10",
    s=8,
    alpha=0.5
)

plt.colorbar(scatter, label="Segment")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Prescriber Segments — PCA Visualization")
plt.tight_layout()
plt.savefig("charts/pca_segments.png", dpi=150)
plt.close()

print("\nPCA variance explained:",
      round(pca.explained_variance_ratio_.sum(), 4))

df.to_parquet(
    "data/segmented_doctors.parquet",
    index=False
)

print("Saved segmentation dataset and charts.")
