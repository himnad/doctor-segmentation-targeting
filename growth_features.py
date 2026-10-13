
import pandas as pd
from pathlib import Path

Path("data").mkdir(exist_ok=True)

def prepare_year(file_path, year):
    df = pd.read_parquet(file_path)

    # Internal Medicine only
    df = df[
        df["Specialty"] == "Internal Medicine"
    ].copy()

    # Clean invalid records
    df = df.dropna(subset=[
        "Prescriber NPI",
        "Total Claims",
        "Total Drug Cost"
    ])

    df = df.drop_duplicates()

    df = df[
        (df["Total Claims"] >= 0) &
        (df["Total Drug Cost"] >= 0)
    ]

    # One row per prescriber
    doctors = df.groupby(
        "Prescriber NPI"
    ).agg(
        total_claims=("Total Claims", "sum"),
        total_cost=("Total Drug Cost", "sum"),
        total_fills=("Total 30-Day Fills", "sum"),
        total_days_supply=("Total Days Supply", "sum"),
        unique_drugs=("Generic Name", "nunique"),
        drug_records=("Generic Name", "size")
    ).reset_index()

    doctors["cost_per_claim"] = (
        doctors["total_cost"] /
        doctors["total_claims"].clip(lower=1)
    )

    doctors["days_per_claim"] = (
        doctors["total_days_supply"] /
        doctors["total_claims"].clip(lower=1)
    )

    doctors["fills_per_claim"] = (
        doctors["total_fills"] /
        doctors["total_claims"].clip(lower=1)
    )

    print(
        f"{year} Internal Medicine doctors:",
        len(doctors)
    )

    return doctors


# 2022 = input features
df_2022 = prepare_year(
    "data/prescribers_2022_tx.parquet",
    2022
)

# 2023 = future outcome
df_2023 = prepare_year(
    "data/prescribers_2023_tx.parquet",
    2023
)

# Match same doctors across years
growth = df_2022.merge(
    df_2023[[
        "Prescriber NPI",
        "total_claims"
    ]],
    on="Prescriber NPI",
    how="inner",
    suffixes=("_2022", "_2023"),
    validate="one_to_one"
)

# Exclude zero baseline to avoid undefined growth
growth = growth[
    growth["total_claims_2022"] > 0
].copy()

# Calculate next-year claims growth
growth["growth_rate"] = (
    growth["total_claims_2023"] -
    growth["total_claims_2022"]
) / growth["total_claims_2022"]

# Rule: 20% or greater increase
growth["rising_prescriber"] = (
    growth["growth_rate"] >= 0.20
).astype(int)

# Rename other 2022 features for clarity
rename_columns = {
    "total_cost": "total_cost_2022",
    "total_fills": "total_fills_2022",
    "total_days_supply": "total_days_supply_2022",
    "unique_drugs": "unique_drugs_2022",
    "drug_records": "drug_records_2022",
    "cost_per_claim": "cost_per_claim_2022",
    "days_per_claim": "days_per_claim_2022",
    "fills_per_claim": "fills_per_claim_2022"
}

growth = growth.rename(columns=rename_columns)

# Save
growth.to_parquet(
    "data/doctor_growth_2022_2023.parquet",
    index=False
)

print("\nMatched doctors:", len(growth))

print("\nRising prescriber distribution:")
print(growth["rising_prescriber"].value_counts())

print("\nGrowth statistics:")
print(
    growth["growth_rate"]
    .describe()
    .round(3)
)

print("\nMissing values:")
print(growth.isnull().sum())

print("\nSaved: data/doctor_growth_2022_2023.parquet")
