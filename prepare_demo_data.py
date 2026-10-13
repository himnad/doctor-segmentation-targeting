
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

DATA_DIR = Path("data")
DEMO_DIR = DATA_DIR / "demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

N = 1500

# Artificial IDs, not real NPIs
ids = [f"DEMO-{i:05d}" for i in range(1, N + 1)]

# -----------------------------------------
# Synthetic 2023 segmentation data
# -----------------------------------------

segment = rng.choice(
    [0, 1, 2, 3],
    size=N,
    p=[0.25, 0.12, 0.45, 0.18]
)

claim_means = {
    0: 1200,
    1: 160,
    2: 6200,
    3: 76
}

cost_per_claim_means = {
    0: 187,
    1: 45,
    2: 107,
    3: 11
}

drug_means = {
    0: 25,
    1: 8,
    2: 97,
    3: 4
}

rows = []

for i in range(N):

    s = int(segment[i])

    claims = max(
        1,
        int(rng.lognormal(
            np.log(claim_means[s]),
            0.5
        ))
    )

    unique_drugs = max(
        1,
        int(rng.normal(
            drug_means[s],
            max(2, drug_means[s] * 0.20)
        ))
    )

    drug_records = unique_drugs + int(
        rng.integers(0, 15)
    )

    fills_per_claim = rng.uniform(1.0, 1.5)
    days_per_claim = rng.uniform(25, 42)

    fills = claims * fills_per_claim
    days = claims * days_per_claim

    cost_per_claim = max(
        1,
        rng.normal(
            cost_per_claim_means[s],
            cost_per_claim_means[s] * 0.25
        )
    )

    rows.append({
        "Prescriber NPI": ids[i],
        "total_claims": claims,
        "total_drug_cost": claims * cost_per_claim,
        "total_fills": fills,
        "total_days_supply": days,
        "unique_drugs": unique_drugs,
        "drug_records": drug_records,
        "cost_per_claim": cost_per_claim,
        "days_per_claim": days_per_claim,
        "fills_per_claim": fills_per_claim,
        "segment": s
    })

segment_df = pd.DataFrame(rows)

segment_df.to_parquet(
    DEMO_DIR / "segmented_doctors.parquet",
    index=False
)

# -----------------------------------------
# Synthetic 2022-2023 growth data
# -----------------------------------------

growth_rows = []

for i in range(N):

    claims_2022 = max(
        1,
        int(rng.lognormal(
            np.log(1200),
            1.0
        ))
    )

    growth_rate = float(
        rng.normal(0.08, 0.35)
    )

    claims_2023 = max(
        1,
        int(claims_2022 * (1 + growth_rate))
    )

    actual_growth = (
        claims_2023 - claims_2022
    ) / claims_2022

    cost_2022 = (
        claims_2022 * rng.uniform(30, 250)
    )

    fills_2022 = (
        claims_2022 * rng.uniform(1.0, 1.5)
    )

    days_2022 = (
        claims_2022 * rng.uniform(25, 42)
    )

    unique_drugs = int(
        rng.integers(5, 120)
    )

    drug_records = (
        unique_drugs + int(rng.integers(0, 20))
    )

    growth_rows.append({
        "Prescriber NPI": ids[i],
        "total_claims_2022": claims_2022,
        "total_cost_2022": cost_2022,
        "total_fills_2022": fills_2022,
        "total_days_supply_2022": days_2022,
        "unique_drugs_2022": unique_drugs,
        "drug_records_2022": drug_records,
        "cost_per_claim_2022": (
            cost_2022 / claims_2022
        ),
        "days_per_claim_2022": (
            days_2022 / claims_2022
        ),
        "fills_per_claim_2022": (
            fills_2022 / claims_2022
        ),
        "total_claims_2023": claims_2023,
        "growth_rate": actual_growth,
        "rising_prescriber": int(
            actual_growth >= 0.20
        )
    })

growth_df = pd.DataFrame(growth_rows)

growth_df.to_parquet(
    DEMO_DIR / "doctor_growth_2022_2023.parquet",
    index=False
)

print("Synthetic demo datasets generated.")
print("Segmentation:", segment_df.shape)
print("Growth:", growth_df.shape)
print("Output directory:", DEMO_DIR)
