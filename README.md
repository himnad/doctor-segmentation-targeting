## Live Demo

**[Launch Prescriber Intelligence Dashboard](https://doctor-segmentation-targeting-mjt3y7jxofqevubrw6kpzy.streamlit.app/)**

Explore doctor segmentation, high-spending classification,
and year-over-year prescriber growth prediction.

**Demo notice:** The deployed dashboard uses synthetic
prescriber records. Model evaluation metrics and charts
are based on historical CMS Medicare Part D data.

\# Doctor Segmentation \& Targeting Using Machine Learning



An end-to-end healthcare analytics and machine learning project that segments Medicare prescribers, identifies high-spending prescribing patterns, and predicts year-over-year growth in prescribing activity.



The project combines unsupervised learning, supervised classification, model evaluation, and an interactive Streamlit dashboard to demonstrate data-driven prescriber targeting.



\## Project Objectives



\- Segment doctors based on prescribing behavior using K-Means clustering.

\- Identify prescribers belonging to the top 20% of observed drug spending.

\- Predict whether a prescriber's total claims will increase by at least 20% in the following year.

\- Evaluate targeting effectiveness using lift and precision at the top decile.

\- Create model-based targeting tiers and an interactive analytics dashboard.



\## Dataset



\*\*Source:\*\* CMS Medicare Part D Prescriber data, accessed through a public Hugging Face dataset mirror.



\*\*Scope:\*\* Texas, Internal Medicine specialty, 2022–2023.



\- 2023 Internal Medicine prescribers: 6,277

\- 2022 Internal Medicine prescribers: 6,121

\- Prescribers matched across both years: 5,284

\- Observed rising prescribers: 1,539



The dataset includes prescribing claims, drug costs, days supplied, prescription fills, drug diversity, and prescriber identifiers.



\## Methodology



\### 1. Prescriber Segmentation



Aggregated provider–drug-level records into prescriber-level features, including total claims, total drug cost, unique drugs, prescription fills, and normalized prescribing metrics.



Applied feature transformations, standardization, and K-Means clustering to identify four behavioral segments:



| Segment | Description |

|---|---|

| 0 | Cost-Intensive Prescribers |

| 1 | Low-Volume Prescribers |

| 2 | High-Volume Prescribers |

| 3 | Minimal-Activity Prescribers |



PCA was used to visualize cluster separation.



\### 2. High-Spending Prescriber Classification



Defined high-spending prescribers as those in the top 20% of observed 2023 total drug cost.



Trained and evaluated Random Forest and Logistic Regression classifiers using prescribing features that exclude direct drug-cost measures.



| Model | F1 Score | ROC-AUC |

|---|---:|---:|

| Random Forest | 0.8025 | 0.9617 |

| Logistic Regression | 0.7832 | 0.9582 |



This task classifies same-year spending behavior; it does not forecast future spending.



\### 3. Rising Prescriber Prediction



Matched prescribers across 2022 and 2023 using their NPI identifiers.



Defined a rising prescriber as one whose total claims increased by at least 20% from 2022 to 2023.



Used only 2022 prescribing features as model inputs, with 2023 claims used to construct the outcome label.



| Metric | Random Forest | Logistic Regression |

|---|---:|---:|

| Precision | 0.4453 | 0.4899 |

| Recall | 0.7922 | 0.7110 |

| F1 Score | 0.5701 | 0.5801 |

| ROC-AUC | 0.7549 | 0.7753 |

| Precision@10% | 0.7170 | 0.6604 |

| Lift@10% | 2.4605 | 2.2663 |



Random Forest was selected for the targeting demonstration because it achieved higher top-decile precision and lift.



\### 4. Model-Based Priority Tiers



Prescribers are ranked by their predicted growth scores and grouped into illustrative targeting tiers:



\- Tier 1: Top 10%

\- Tier 2: Next 20%

\- Tier 3: Remaining 70%



These tiers demonstrate analytical prioritization and are not validated measures of sales effectiveness.



\## Streamlit Dashboard



The application includes three sections:



\*\*Overview \& Segmentation:\*\* Prescriber KPIs, behavioral segments, segment profiles, PCA visualizations, and downloadable results.



\*\*High-Spending Targeting:\*\* Classification model comparison, ROC curve, confusion matrix, feature importance, targeting filters, and live predictions.



\*\*Rising Prescriber Prediction:\*\* Growth model comparison, decile lift chart, priority tiers, ranked prescriber lists, CSV export, and live prediction.



\## Technology Stack



Python, Pandas, NumPy, Scikit-learn, Matplotlib, Streamlit, Joblib, and PyArrow.



\## Running Locally



Install the dependencies:



```bash

pip install -r requirements.txt

```



Start the dashboard:



```bash

streamlit run app.py

```



Open the local address displayed by Streamlit, usually `http://localhost:8501`.



The dashboard requires the processed Parquet datasets, trained models, and visualization files referenced by `app.py`.



\## Project Limitations



\- Analysis is restricted to Texas Internal Medicine prescribers.

\- Growth prediction uses a 2022-to-2023 transition and a random held-out prescriber test set.

\- The project does not establish generalization to later calendar years.

\- Matching across years excludes prescribers not present in both datasets.

\- Full-dataset dashboard rankings include training-set predictions and must not be interpreted as independent test performance.

\- The models predict prescribing activity patterns, not clinical quality, treatment appropriateness, or actual sales outcomes.

\- Model scores have not been separately calibrated.



\## Data Attribution



The underlying prescribing data originates from the CMS Medicare Part D public-use data program and was obtained through a third-party public dataset mirror. This repository is an educational portfolio project and is not affiliated with or endorsed by CMS.



\## Project Purpose



Developed as a data science portfolio project to demonstrate prescriber segmentation, predictive modeling, model evaluation, and interactive analytics relevant to healthcare and pharmaceutical commercial intelligence.

