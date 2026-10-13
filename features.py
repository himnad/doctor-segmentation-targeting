
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_parquet("data/internal_medicine_clean.parquet")

# One row per prescriber
doctors = df.groupby("Prescriber NPI").agg(
    total_claims=("Total Claims", "sum"),
    total_drug_cost=("Total Drug Cost", "sum"),
    total_fills=("Total 30-Day Fills", "sum"),
    total_days_supply=("Total Days Supply", "sum"),
    unique_drugs=("Generic Name", "nunique"),
    drug_records=("Generic Name", "size")
).reset_index()

# Derived prescribing features
doctors["cost_per_claim"] = (
    doctors["total_drug_cost"] / doctors["total_claims"]
)

doctors["days_per_claim"] = (
    doctors["total_days_supply"] / doctors["total_claims"]
)

doctors["fills_per_claim"] = (
    doctors["total_fills"] / doctors["total_claims"]
)

# Check for invalid values
print("SHAPE:", doctors.shape)
print("\nMISSING VALUES:")
print(doctors.isnull().sum().to_string())

print("\nFIRST 5 ROWS:")
print(doctors.head().to_string())

print("\nDUPLICATE PRESCRIBERS:",
      doctors["Prescriber NPI"].duplicated().sum())

doctors.to_parquet(
    "data/doctor_features.parquet",
    index=False
)

print("\nSaved: data/doctor_features.parquet")
