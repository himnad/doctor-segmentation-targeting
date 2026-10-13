
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Path("charts").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

df = pd.read_parquet("data/prescribers_2023_tx.parquet")

# Keep Internal Medicine only
df = df[df["Specialty"] == "Internal Medicine"].copy()

# Remove invalid records
df = df.drop_duplicates()
df = df.dropna(subset=["Prescriber NPI"])

df = df[
    (df["Total Claims"] > 0) &
    (df["Total Drug Cost"] >= 0)
].copy()

# Preserve missing beneficiary counts
# Do not replace them with zero

df.to_parquet(
    "data/internal_medicine_clean.parquet",
    index=False
)

print("Cleaned rows:", len(df))
print("Unique prescribers:", df["Prescriber NPI"].nunique())
print("Missing beneficiaries:", df["Total Beneficiaries"].isna().sum())

# Plot drug-record-level claim distribution
plt.figure(figsize=(9, 5))

plt.hist(
    np.log1p(df["Total Claims"]),
    bins=50,
    edgecolor="white"
)

plt.title("Prescription Claims Distribution — Internal Medicine")
plt.xlabel("log(1 + Total Claims per Drug Record)")
plt.ylabel("Number of Drug Records")

plt.tight_layout()
plt.savefig("charts/claims_distribution.png", dpi=150)
plt.close()

print("Chart saved: charts/claims_distribution.png")
