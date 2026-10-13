# Doctor Segmentation & Targeting Using Machine Learning

An end-to-end healthcare analytics project that segments Medicare prescribers by prescribing behavior, classifies high-spending prescribers, and predicts which prescribers are likely to grow their prescribing activity the following year. Results are delivered through an interactive Streamlit dashboard.

**[Live Demo: Prescriber Intelligence Dashboard](https://doctor-segmentation-targeting-mjt3y7jxofqevubrw6kpzy.streamlit.app/)**

> **Demo notice:** The deployed dashboard uses synthetic prescriber records. Model evaluation metrics and charts are based on historical CMS Medicare Part D data.

<!-- Add 2-3 screenshots here, e.g.:
![Segmentation overview](docs/segmentation_overview.png)
![Lift chart](docs/lift_chart.png)
-->

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Dashboard](#dashboard)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Limitations](#limitations)
- [Data Attribution](#data-attribution)

---

## Overview

Pharmaceutical commercial teams need to decide which doctors to prioritize. This project explores that question on public Medicare data through three analyses:

1. **Segmentation:** group prescribers into behavioral segments with K-Means clustering.
2. **High-spending classification:** identify prescribers in the top 20% of 2023 drug spending.
3. **Growth prediction:** predict whether a prescriber's total claims will rise by at least 20% in the following year, and evaluate targeting quality with decile lift.

## Dataset

**Source:** [CMS Medicare Part D Prescriber data](https://data.cms.gov/), accessed through a public Hugging Face dataset mirror.
**Scope:** Texas, Internal Medicine, 2022 and 2023.

| Item | Count |
|---|---:|
| 2023 Internal Medicine prescribers | 6,277 |
| 2022 Internal Medicine prescribers | 6,121 |
| Prescribers matched across both years (by NPI) | 5,284 |
| Observed rising prescribers (claims up 20% or more) | 1,539 (29.1%) |

Fields used include prescribing claims, drug costs, days supplied, prescription fills, drug diversity, and prescriber identifiers.

## Methodology

### 1. Prescriber Segmentation

- Aggregated provider-drug level records into one row per prescriber (total claims, total drug cost, unique drugs, fills, and normalized prescribing metrics).
- Applied feature transformations and standardization, then K-Means clustering (4 segments). PCA was used to visualize cluster separation.

| Segment | Label |
|---:|---|
| 0 | Cost-Intensive Prescribers |
| 1 | Low-Volume Prescribers |
| 2 | High-Volume Prescribers |
| 3 | Minimal-Activity Prescribers |

### 2. High-Spending Prescriber Classification

- **Label:** prescribers in the top 20% of observed 2023 total drug cost.
- **Features:** prescribing-activity features, excluding direct drug-cost measures.
- **Models:** Random Forest and Logistic Regression.
- **Scope:** this classifies same-year spending behavior. It does not forecast future spending.

### 3. Rising Prescriber Prediction

- Matched prescribers across 2022 and 2023 using NPI.
- **Label:** total claims increased by at least 20% from 2022 to 2023.
- **Features:** 2022 prescribing features only. 2023 claims are used solely to construct the label.
- **Models:** Random Forest and Logistic Regression, compared on classification metrics and top-decile targeting performance.

### 4. Priority Tiers

Prescribers are ranked by predicted growth score and grouped into illustrative tiers: **Tier 1** (top 10%), **Tier 2** (next 20%), **Tier 3** (remaining 70%).

## Results

### High-Spending Classification

| Model | F1 | ROC-AUC |
|---|---:|---:|
| Random Forest | 0.8025 | 0.9617 |
| Logistic Regression | 0.7832 | 0.9582 |

### Rising Prescriber Prediction

| Metric | Random Forest | Logistic Regression |
|---|---:|---:|
| Precision | 0.4453 | 0.4899 |
| Recall | 0.7922 | 0.7110 |
| F1 | 0.5701 | 0.5801 |
| ROC-AUC | 0.7549 | 0.7753 |
| Precision@10% | 0.7170 | 0.6604 |
| Lift@10% | 2.4605 | 2.2663 |

**Interpretation:** About 29% of matched prescribers are rising prescribers. Among the 10% of prescribers the Random Forest ranks highest, about 72% are truly rising (Lift@10% of 2.46, i.e. roughly 2.5x better than random selection). Logistic Regression scores slightly higher on F1 and ROC-AUC, but Random Forest was selected for the targeting demo because it gives better top-decile precision and lift.

## Dashboard

| Section | Contents |
|---|---|
| **Overview & Segmentation** | KPIs, behavioral segments, segment profiles, PCA plot, downloadable results |
| **High-Spending Targeting** | Model comparison, ROC curve, confusion matrix, feature importance, filters, live prediction |
| **Rising Prescriber Prediction** | Model comparison, decile lift chart, priority tiers, ranked prescriber list, CSV export, live prediction |

## Tech Stack

Python, Pandas, NumPy, scikit-learn, Matplotlib, Streamlit, Joblib, PyArrow

## Getting Started

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
streamlit run app.py
```

Then open the local address Streamlit prints, usually `http://localhost:8501`.

The app expects the processed Parquet datasets, trained models, and chart files referenced in `app.py` to be present in the repository.

## Limitations

- Scope is limited to Texas Internal Medicine prescribers.
- Growth prediction uses a single 2022-to-2023 transition and a random hold-out split. Generalization to later years is not established.
- Only prescribers present in both years are included in the growth analysis.
- High-spending classification uses features such as fills and days supplied, which are strongly related to cost. High scores partly reflect this relationship.
- Dashboard rankings over the full dataset include training-set predictions and must not be read as independent test performance.
- Models predict prescribing activity patterns, not clinical quality, treatment appropriateness, or sales outcomes. Priority tiers are illustrative.
- Model scores have not been calibrated.
- The deployed dashboard uses synthetic records for demonstration.

## Data Attribution

The underlying data comes from the CMS Medicare Part D public-use data program and was obtained through a third-party public dataset mirror. This is an educational portfolio project and is not affiliated with or endorsed by CMS.
