
import pandas as pd

df = pd.read_parquet("data/prescribers_2023_tx.parquet")

print("TOP 15 SPECIALTIES:")
print(df["Specialty"].value_counts().head(15))

print("\nINTERNAL MEDICINE:")
im = df[df["Specialty"] == "Internal Medicine"]

print("Rows:", len(im))
print("Unique prescribers:", im["Prescriber NPI"].nunique())
print("Missing beneficiary %:",
      round(im["Total Beneficiaries"].isna().mean() * 100, 2))
