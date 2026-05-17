# Analytical Decisions and Findings
## Rossmann Drug Store Sales Forecasting
**Vaishnavi Jitendra Bhor** | MSc Business Analytics, University of Manchester  
[LinkedIn](https://www.linkedin.com/in/vaishnavi-bhor-business-analyst/) | vaishnavibhor123@gmail.com

---

This document covers the analytical decisions made throughout the project — the reasoning behind key choices, observations from the data, and areas identified for further development. It is intended to complement the README, which covers methodology and results, by explaining the thinking behind the approach.

---

## Problem Framing

Before any data was loaded, the business question was defined precisely. "Forecast sales" is a task. The actual question is: what drives daily sales variation across 1,115 stores, and can that variation be predicted accurately enough to support operational decisions on staffing, stock, and promotions?

That distinction matters. A vague task leads to exploratory work with no anchor. A specific question shapes every decision that follows — which metric to use, which features to engineer, what the output should look like.

---

## Data Quality Observations

The first step on any dataset is checking categorical column values before running any analysis. Two issues were found in this data that would have silently corrupted results downstream.

**Empty columns in stores.csv.** Two unnamed columns — `Unnamed: 10` and `Unnamed: 11` — contained no data across all 1,115 rows. They had come in with the Kaggle download and would have been carried through every join without causing an error. They were dropped immediately.

**StateHoliday type inconsistency.** The column was stored as a mix of string `"0"` and integer `0` depending on the row. In Python, these are not equal — a filter on `df[df['StateHoliday'] == 0]` returns approximately half the expected rows. Any holiday-based analysis or model trained without fixing this would produce systematically wrong results. The column was standardised to integers (0/1/2/3) before any analysis was run.

Both issues are the kind that produce no error message and are difficult to detect after the fact. Catching them early is more important than any modelling decision that follows.

---

## Choice of Evaluation Metric

RMSPE (Root Mean Square Percentage Error) was used throughout rather than the more common RMSE.

The reason is straightforward: a forecasting error of €2,000 at a store doing €4,000 per day is operationally catastrophic. The same error at a store doing €40,000 per day is negligible. RMSE treats both identically because it penalises absolute errors. RMSPE penalises percentage errors, which reflects how store managers actually plan — their staffing and stock decisions are proportional to expected sales volume, not to absolute figures.

RMSPE is also the official Kaggle evaluation metric for this competition, meaning it is the measure Rossmann considered most relevant to their operational context. Both reasons point to the same choice.

---

## Store Clustering Before Model Training

K-Means clustering was applied to store sales profiles before any forecasting model was trained. This is a deliberate sequencing decision, not a preliminary step.

Training a single model across all 1,115 stores forces the algorithm to simultaneously describe stores with very different characteristics. A mid-volume store averaging €5,000 per day and a high-volume store averaging €30,000 per day have different seasonal patterns, different promotion response rates, and different sensitivities to nearby competition. One model applied across both groups finds a compromise that fits neither well.

Training GBM separately within each cluster produced an average RMSPE of 25.95% versus 36.9% for the same model trained globally — a 29.7% reduction in error. The High-Volume cluster specifically reached 22.3% RMSPE with R²=0.727. The improvement is consistent across all three model types tested (GBM, Random Forest, Neural Network), which rules out the result being an artefact of one particular run.

The current segmentation at K=2 (821 mid-volume, 294 high-volume stores) is optimal by Silhouette Score but remains a relatively coarse split. Experimenting with K=3 or K=4 — particularly to assess whether the 17 Type B stores warrant their own cluster — is a natural next step.

---

## Feature Engineering Decisions

Two engineered features are worth explaining in detail as they produced the most meaningful results.

**CompetitionOpen — months elapsed since competitor opened.**

The raw dataset provides CompetitionDistance, a static measure of where the nearest competitor is located. Distance alone does not capture competitive dynamics over time. A competitor who opened three months ago is actively pulling customers during their launch phase. One who has been established for four years has become part of the local landscape — customers have already formed their preferences.

Converting the competitor's opening date to months elapsed captures this disruption curve. In the GBM feature importance ranking, CompetitionOpen outperforms raw CompetitionDistance, confirming that recency of competitive entry is more predictive than proximity alone.

**LogCompetitionDistance — log-transformed distance.**

CompetitionDistance in the raw data ranges from 20m to 75,860m, a range of nearly 4,000x. Gradient boosting algorithms split on thresholds, and with that spread, most of the useful signal sits at one extreme end of the scale. Log-transforming the variable compresses the range to approximately 3–11, distributing the signal more evenly and allowing the model to detect the isolation premium — stores with distant competition tend to achieve higher average sales, likely due to reduced price pressure. This pattern is not reliably detectable in the raw variable.

---

## Neural Network Performance

The Neural Network produced the highest RMSPE across both clusters (34.2% Mid-Volume, 32.7% High-Volume), performing below both GBM and Random Forest.

This outcome is consistent with the general behaviour of neural networks on structured tabular data with a relatively modest training set. Tree-based ensemble methods typically outperform neural networks in this setting, particularly when training is constrained by time or compute. The network here was trained for 50–200 epochs on a 40% data sample, which is not sufficient to realise its potential.

For context, the group coursework neural network achieved 8–9% RMSPE using a time-slicing approach — training on 730-day sequential windows per store, which generates approximately 182,000 training rows from the same base data. That architecture is fundamentally different: it learns store-specific temporal patterns rather than generalising across stores using features. The feature-based approach used here is more interpretable and generalises better to stores not seen during training, but it requires more data per store to reach comparable accuracy.

---

## Test Set Predictions

The project produces two output files from the test set predictions.

`rossmann_predictions.csv` contains one row per store per day across the full August–September 2015 test window — 41,088 records in total, each with a predicted sales figure.

`rossmann_store_predictions_summary.csv` contains one row per store, summarising the predicted six-week total revenue and average daily sales for the forecast period. This format is directly usable for operational planning — a store manager can take this file and use it to set staffing rotas and place stock orders without needing to interpret or aggregate the daily predictions themselves.

The distinction between these two outputs is deliberate. Raw predictions are useful for technical validation. The summary file is useful for the people who actually need to act on the forecast.

---

## Areas for Further Development

**Lag features.** Prior sales — yesterday, last week same day, last year same week — are among the most predictive variables in retail forecasting. They were not included here because building them correctly requires careful handling to prevent data leakage: the lag value used at prediction time must only reflect information available at that point. Setting this up properly within a time-based train/validation split is straightforward but time-consuming. It is the highest-priority addition for a future iteration.

**Finer store segmentation.** K=2 is optimal by Silhouette Score but remains a coarse partition. Testing K=3 and K=4 — and examining whether cluster boundaries align with meaningful business categories such as store type or geographic region — would establish whether finer segmentation further reduces forecasting error.

**Full-data training run.** All model results in this project come from a 40% training sample used for development speed. The scripts include a `DEV_MODE` flag that disables sampling for a full production run. On the complete dataset with 300 estimators, GBM on this data typically reaches 15–20% RMSPE, placing it within the upper range of Kaggle public leaderboard submissions. A full run would produce more defensible benchmark figures.

**Store-level model tuning.** The current approach trains one model per cluster of several hundred stores. A natural extension is to train separate models for store sub-groups defined by store type, assortment, or geographic state — or to explore whether store-specific hyperparameter tuning for the highest-volume stores produces further accuracy gains.

---

*Vaishnavi Jitendra Bhor*  
*vaishnavibhor123@gmail.com*  
*[linkedin.com/in/vaishnavi-bhor-business-analyst](https://www.linkedin.com/in/vaishnavi-bhor-business-analyst/)*
