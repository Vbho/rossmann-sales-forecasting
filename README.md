# Rossmann Drug Store Sales Forecasting
**Author:** Vaishnavi Jitendra Bhor  
**LinkedIn:** [linkedin.com/in/vaishnavi-bhor-business-analyst](https://www.linkedin.com/in/vaishnavi-bhor-business-analyst/)  
**Email:** vaishnavibhor123@gmail.com  
**Dataset:** [Rossmann Store Sales — Kaggle](https://www.kaggle.com/competitions/rossmann-store-sales/data)

---

## Overview

Rossmann operates over 3,000 drug stores across Europe. Store managers need accurate sales forecasts up to six weeks ahead to plan staffing, order stock, and schedule promotions. Getting it wrong has direct cost consequences — over-ordering ties up cash, under-staffing hurts customer satisfaction, and missed promotion windows result in lost revenue at scale.

This project approaches the problem as a real operational forecasting challenge. The dataset covers 1,115 stores in Germany from January 2013 to July 2015. The test set covers August and September 2015 — the exact six-week window store managers needed to plan for. The output is designed to be directly usable by an operations team, not just a model evaluation metric.

---

## Central Business Question

> *Which factors drive daily sales variation across 1,115 German drug stores, and can that variation be forecast six weeks ahead accurately enough to support operational decisions on staffing, stock, and promotions?*

---

## Repository Structure

```
rossmann-sales-forecasting/
│
├── README.md
├── Project_Observations_and_Findings.md
│
├── data/
│   ├── DA2024_train.csv                        (raw training data — 1,017,209 records)
│   ├── DA2024_stores.csv                       (store metadata — 1,115 stores)
│   ├── DA2024_test.csv                         (test set — 41,088 records, Sales unknown)
│   ├── rossmann_train_features.csv             (cleaned + engineered — 844,338 rows)
│   ├── rossmann_test_features.csv              (test set with same feature engineering)
│   ├── store_clusters.csv                      (K-Means cluster assignment per store)
│   ├── rossmann_train_clustered.csv            (training data with cluster labels)
│   ├── model_comparison_results.csv            (RMSPE and R² for all models)
│   ├── gbm_feature_importance.csv              (GBM feature importance scores)
│   ├── rossmann_predictions.csv                (41,088 sales predictions for Aug–Sep 2015)
│   └── rossmann_store_predictions_summary.csv  (6-week forecast summary per store)
│
├── analysis/
│   ├── 01_data_quality_features.py
│   └── 02_eda_full.py
│
├── ml_models/
│   ├── ml1_clustering.py
│   ├── ml2_3_4_forecasting.py
│   └── ml5_predictions_on_test.py
│
├── excel/
│   └── Rossmann_Analysis_Complete.xlsx
│
└── charts/
    ├── E1_sales_overview.png
    ├── E2_store_type_analysis.png
    ├── E3_seasonality.png
    ├── E4_promotion_analysis.png
    ├── E5_competition_analysis.png
    ├── E6_correlation_matrix.png
    ├── ML1_clustering.png
    ├── ML2_model_comparison.png
    ├── ML3_feature_importance.png
    ├── ML4_prediction_analysis.png
    └── ML5_predicted_sales_analysis.png
```

> **Note on raw data files:** `DA2024_train.csv` exceeds GitHub's 25MB browser upload limit. Download the three raw Kaggle files directly from the [competition page](https://www.kaggle.com/competitions/rossmann-store-sales/data) and place them in the `data/` folder before running the scripts.

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl

python analysis/01_data_quality_features.py
python analysis/02_eda_full.py
python ml_models/ml1_clustering.py
python ml_models/ml2_3_4_forecasting.py
python ml_models/ml5_predictions_on_test.py
```

`ml2_3_4_forecasting.py` includes a `DEV_MODE` flag at the top (currently set to `True`). In development mode the script uses a 40% training sample and runs in approximately five minutes. Setting `DEV_MODE = False` trains on the full dataset and takes around 20 minutes. Full-data GBM typically reaches 18–22% RMSPE on this dataset.

Scripts must be run in the order listed above, as each produces output files used by subsequent scripts.

---

## Analytical Framework

Sales variation across stores and days was examined through a structured hypothesis framework before any modelling was undertaken. Each driver was tested independently before moving to the next.

```
Why do sales vary so much across stores and days?

Store characteristics
  Store type (a/b/c/d)?             Yes — Type B: €10,060/day vs Type D: €5,738/day
  Assortment level (a/b/c)?         Yes — Assortment B drives highest spend per visit
  Overall sales volume profile?     Yes — high-volume stores have different seasonal patterns

Timing
  Day of week?                      Yes — Monday is consistently the strongest sales day
  Month and season?                 Yes — December peaks, July dips across all years
  Year-on-year trend?               Yes — 2014 revenue dipped due to store closures, not demand

Promotions
  Daily promotion (Promo)?          Yes — promo days average 81% higher sales
  Mailing scheme (Promo2)?          Marginal negative correlation — less effective than Promo

Competition
  Distance to nearest competitor?   Partial — stores in less competitive areas earn a premium
  Recency of competitor opening?    Yes — recent entrants suppress sales more than established ones

External context
  State holidays?                   Yes — Type B and Assortment B stores spike on state holidays
  School holidays?                  Moderate positive effect across all store types
```

---

## Data Quality

The raw files were audited before any analysis was run. The issues below were identified and corrected at the data preparation stage.

| Issue | Scale | Fix Applied |
|---|---|---|
| Two empty unnamed columns in stores.csv | 1,115 rows | Dropped |
| Missing CompetitionDistance | 3 stores | Filled with median (2,325m) |
| CompetitionOpenSince missing for stores with no competitor | 354 stores | Filled with 0 |
| Promo2Since missing for non-participating stores | 544 stores | Filled with 0 |
| StateHoliday: string "0" mixed with integer 0 | 986,159 rows | Standardised to 0/1/2/3 |
| Open=1 but Sales=0 (data entry errors) | 54 records | Removed |
| Open status unknown in test set | 11 rows | Filled with 1 |

The StateHoliday inconsistency is particularly worth noting. In Python, the string `"0"` and the integer `0` are not equal. Filtering `df[df['StateHoliday'] == 0]` without fixing this returns approximately half the expected rows, with no error raised. Any holiday-related analysis or model feature built on uncorrected data would be systematically wrong.

---

## Feature Engineering

The original dataset contained 9 columns. 16 new features were engineered before model training. Three produced the most significant impact on model accuracy.

**`CompetitionOpen`** — months elapsed since the nearest competitor opened, calculated from the competitor's opening date rather than just their distance. A competitor in their launch phase is actively drawing customers through opening promotions. One established for four years is already part of the local market landscape — customers have formed stable preferences. This time-based variable captures the competitive disruption curve that static distance cannot represent. In GBM feature importance rankings, `CompetitionOpen` consistently outperforms raw `CompetitionDistance`.

**`LogCompetitionDistance`** — log-transformed competition distance. The raw variable ranges from 20m to 75,860m — nearly a 4,000x spread. Gradient boosting splits on thresholds, and with that distribution most of the signal sits at one extreme of the scale. Log-transforming to a 3–11 range distributes the signal more evenly, allowing the model to detect the isolation premium: stores with distant competition tend to achieve higher average daily sales, likely due to reduced price pressure in their catchment area.

**`IsPromo2Month`** — a binary flag for whether the current calendar month falls within a store's active Promo2 mailing interval. The raw `Promo2` column indicates only whether a store participates in the scheme. This feature distinguishes between participation and active promotion, capturing actual scheme activity at the transaction level.

---

## Machine Learning

### ML1 — Store Clustering

Before any forecasting model was trained, the 1,115 stores were segmented by sales profile using K-Means clustering. Optimal cluster count was determined as K=2 by Silhouette Score (0.293).

| Cluster | Label | Stores | Avg Daily Sales |
|---|---|---|---|
| 0 | Mid-Volume | 821 | €5,888 |
| 1 | High-Volume | 294 | €9,857 |

Training a single model across all stores forces the algorithm to simultaneously describe stores with fundamentally different characteristics. A store averaging €5,000 per day and one averaging €30,000 per day have different seasonal shapes, different promotion response rates, and different competitive sensitivities. Training within clusters keeps each model within a comparable population and avoids the averaging effect that degrades accuracy for both groups.

### ML2–4 — Forecasting Models

Three model types were trained separately on each cluster and evaluated against a held-out validation set representing the final six weeks of training data — matching the length of the actual test period.

| Model | Cluster | RMSPE | R² |
|---|---|---|---|
| **GradientBoosting** | Mid-Volume | **29.6%** | **0.571** |
| **GradientBoosting** | High-Volume | **22.3%** | **0.727** |
| Random Forest | Mid-Volume | 32.7% | 0.461 |
| Random Forest | High-Volume | 27.6% | 0.521 |
| Neural Network | Mid-Volume | 34.2% | 0.453 |
| Neural Network | High-Volume | 32.7% | 0.298 |
| GBM — global training (no clustering) | All stores | 36.9% | 0.550 |

**Evaluation metric — RMSPE** (Root Mean Square Percentage Error) was selected as the evaluation metric. It penalises percentage errors rather than absolute errors, which is appropriate for retail forecasting: a 20% error at a store doing €40,000 per day is operationally ten times more costly than the same percentage error at a €4,000-per-day store. RMSPE is also the official Kaggle metric for this competition.

**Why not ARIMA:** With 2.5 years of data per store, ARIMA parameter estimation becomes unstable and prone to overfitting. GradientBoostingRegressor in scikit-learn produces results comparable to XGBoost for this type of structured tabular forecasting problem.

**Improvement from clustering:** Per-cluster training reduced average GBM RMSPE from 36.9% (global) to 25.95% (per-cluster average) — a 29.7% reduction. The improvement is consistent across all three model types, confirming that the gain comes from the clustering approach rather than from any model-specific factor.

### ML5 — Test Set Predictions

The best-performing model (GBM) was retrained on the complete training dataset and used to generate predictions for all 41,088 records in the test set covering August 1 to September 17, 2015.

Two output files were produced:

- `rossmann_predictions.csv` — one row per store per day with the predicted sales figure
- `rossmann_store_predictions_summary.csv` — one row per store with the six-week predicted total and daily average

The summary file is structured for direct operational use. A store manager can open it, read off the expected six-week revenue and average daily sales for their store, and use those figures to set staffing levels and place stock orders without needing to process or aggregate the daily data themselves.

---

## Key Findings

**Promotions are the most impactful operational lever.** Daily promotions (Promo) generate 81% higher average sales than non-promo days (€7,991 vs €4,406). The Promo2 mailing scheme shows a marginal negative correlation with sales despite wider participation — it appears to attract more price-sensitive customers who spend less per visit than those who respond to direct daily promotions.

**Store Type B is disproportionately high-performing.** Only 17 of 1,115 stores are Type B, yet they average €10,060 per day — nearly double the network average. The combination of Type B store format and Assortment C produces the highest average transaction value across all store configurations.

**The recency of a competitor's opening matters more than their distance.** A competitor who opened within the last six months suppresses sales more significantly than one established for three or more years. Sales recover gradually as customer preferences stabilise. This is reflected in `CompetitionOpen` ranking above raw `CompetitionDistance` in feature importance.

**The 2014 revenue decline is structural, not demand-driven.** Network revenue fell from approximately €2.0B in 2013 to €1.9B in 2014. Store-level analysis confirms this reflects store closures rather than declining per-store performance — individual stores that remained open maintained consistent daily sales through the period.

**Monday consistently produces the highest sales volume.** The gap between Monday and all other trading days exceeds most seasonal effects in magnitude. Monday shift allocation represents the single highest-impact staffing decision at network level.

---

## Recommendations

| Priority | Action | Rationale |
|---|---|---|
| 1 | Schedule Promo activity on Mondays | Highest base demand combined with strongest promo uplift maximises revenue per promotion spend |
| 2 | Prioritise Promo support for stores within 500m of competitors | Competition recency analysis identifies this as the segment with the most acute sales suppression |
| 3 | Reassess Promo2 scheme effectiveness | Marginal negative correlation with sales suggests the mailing budget may produce better returns elsewhere |
| 4 | Staff Type B stores approximately 40% above Type A baseline | Customer count and ATV analysis confirm significantly higher throughput requirements |
| 5 | Begin stock pre-ordering from Week 46 | Weeks 48–50 are consistently the highest-revenue period — stockouts during this window are disproportionately costly |

---

## Tools

Python — pandas, numpy, matplotlib, seaborn, scikit-learn  
Excel — openpyxl  
Dataset — Rossmann Store Sales, Kaggle (public dataset)

---

*Vaishnavi Jitendra Bhor — vaishnavibhor123@gmail.com*  
*[LinkedIn](https://www.linkedin.com/in/vaishnavi-bhor-business-analyst/) | MSc Business Analytics, University of Manchester*  
*Open to Business Analyst, Data Analyst, and consulting roles in the UK and Europe*
