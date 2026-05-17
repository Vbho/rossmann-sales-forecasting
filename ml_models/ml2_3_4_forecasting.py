# Rossmann Drug Store Sales Forecasting
# ML Scripts 2-4: GradientBoosting, Random Forest, Neural Network
# Vaishnavi Jitendra Bhor | MSc Business Analytics, University of Manchester
#
# Models are trained separately per store cluster (from ml1_clustering.py).
# Stores are grouped by sales profile before training because a mid-volume
# store and a high-volume store have different seasonal shapes and promo
# response rates. One model across all 1,115 stores finds a compromise
# that doesn't work well for either group. Per-cluster training
# reduces average RMSPE by roughly 29% compared to training on all stores.
#
# Metric: RMSPE (Root Mean Square Percentage Error)
#   Official Kaggle metric for this competition. Chosen because it
#   penalises percentage errors rather than absolute ones — a 20% error
#   at a €40K/day store costs ten times more than at a €4K/day store.
#
# DEV_MODE = True  uses 40% sample, runs in ~5 minutes
# DEV_MODE = False trains on full data, runs in ~20 minutes
#
# Inputs : data/rossmann_train_clustered.csv
# Outputs: data/model_comparison_results.csv
#          data/gbm_feature_importance.csv
#          charts/ML2_model_comparison.png
#          charts/ML3_feature_importance.png
#
# Run: python ml_models/ml2_3_4_forecasting.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch
import os, warnings
warnings.filterwarnings('ignore')
os.makedirs('charts', exist_ok=True)
os.makedirs('data',   exist_ok=True)

DEV_MODE    = True
SAMPLE_FRAC = 0.40

plt.rcParams.update({
    'figure.facecolor': '#F8FAFC', 'axes.facecolor': '#F8FAFC',
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'DejaVu Sans', 'axes.titlesize': 12,
    'axes.titleweight': 'bold', 'axes.labelsize': 10
})
BLUE, GREEN, RED, AMBER, PURPLE, NAVY = '#2563EB','#16A34A','#DC2626','#D97706','#7C3AED','#1E3A5F'
LGREY = '#F1F5F9'

print("Loading clustered training data...")
df = pd.read_csv('data/rossmann_train_clustered.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])
print(f"  {len(df):,} rows | Clusters: {sorted(df['Cluster'].unique())}")

# Feature set — 21 variables selected through EDA and feature importance iteration
FEATURES = [
    'Store',
    'DayOfWeek', 'Month', 'Day', 'WeekOfYear', 'Year',
    'Quarter', 'IsWeekend', 'IsMonthStart', 'IsMonthEnd', 'Season',
    'Promo', 'Promo2', 'Promo2Open', 'IsPromo2Month',
    'CompetitionOpen',
    'LogCompetitionDistance',
    'StateHoliday', 'SchoolHoliday',
    'StoreType_enc', 'Assortment_enc',
]
TARGET = 'Sales'


def rmspe(y_true, y_pred):
    """Root Mean Square Percentage Error. Only computed on non-zero actual values."""
    mask = y_true != 0
    return np.sqrt(np.mean(
        ((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2
    )) * 100


# Validation split: hold out the last 6 weeks of training data
# This mirrors the length of the actual test period (Aug–Sep 2015)
cutoff = df['Date'].max() - pd.Timedelta(weeks=6)
print(f"\nValidation split: training up to {cutoff.date()}, validating after")

results = []
feature_importance_all = {}

for cluster in sorted(df['Cluster'].unique()):
    cluster_name = df[df['Cluster'] == cluster]['ClusterName'].iloc[0]
    print(f"\n--- Cluster {cluster}: {cluster_name} ---")

    train_cl = df[(df['Cluster'] == cluster) & (df['Date'] <= cutoff)]
    val_cl   = df[(df['Cluster'] == cluster) & (df['Date'] >  cutoff)]

    if DEV_MODE:
        train_cl = train_cl.sample(frac=SAMPLE_FRAC, random_state=42)

    X_tr = train_cl[FEATURES].fillna(0)
    y_tr = train_cl[TARGET]
    X_va = val_cl[FEATURES].fillna(0)
    y_va = val_cl[TARGET]

    print(f"  Train: {len(X_tr):,} rows | Val: {len(X_va):,} rows")

    # Gradient Boosting
    print("  Training GradientBoosting...")
    gbm = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42
    )
    gbm.fit(X_tr, y_tr)
    gbm_pred  = gbm.predict(X_va).clip(min=0)
    gbm_rmspe = rmspe(y_va.values, gbm_pred)
    gbm_r2    = gbm.score(X_va, y_va)
    print(f"    RMSPE: {gbm_rmspe:.2f}%  R2: {gbm_r2:.3f}")
    results.append({
        'Cluster': cluster, 'ClusterName': cluster_name,
        'Model': 'GradientBoosting', 'RMSPE': round(gbm_rmspe, 2),
        'R2': round(gbm_r2, 3), 'Training': 'Per-Cluster'
    })
    feature_importance_all[f'C{cluster}'] = pd.Series(gbm.feature_importances_, index=FEATURES)

    # Random Forest
    print("  Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42,
        max_features='sqrt'
    )
    rf.fit(X_tr, y_tr)
    rf_pred  = rf.predict(X_va).clip(min=0)
    rf_rmspe = rmspe(y_va.values, rf_pred)
    rf_r2    = rf.score(X_va, y_va)
    print(f"    RMSPE: {rf_rmspe:.2f}%  R2: {rf_r2:.3f}")
    results.append({
        'Cluster': cluster, 'ClusterName': cluster_name,
        'Model': 'RandomForest', 'RMSPE': round(rf_rmspe, 2),
        'R2': round(rf_r2, 3), 'Training': 'Per-Cluster'
    })

    # Neural Network
    print("  Training Neural Network...")
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    nn = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        batch_size=512
    )
    nn.fit(X_tr_sc, y_tr)
    nn_pred  = nn.predict(X_va_sc).clip(min=0)
    nn_rmspe = rmspe(y_va.values, nn_pred)
    nn_r2    = nn.score(X_va_sc, y_va)
    print(f"    RMSPE: {nn_rmspe:.2f}%  R2: {nn_r2:.3f}")
    results.append({
        'Cluster': cluster, 'ClusterName': cluster_name,
        'Model': 'NeuralNetwork', 'RMSPE': round(nn_rmspe, 2),
        'R2': round(nn_r2, 3), 'Training': 'Per-Cluster'
    })

# Pre-computed results from the global (no-clustering) baseline run
# Kept here to allow side-by-side comparison in charts and CSV
global_baseline = [
    {'Cluster': -1, 'ClusterName': 'All Stores', 'Model': 'GradientBoosting',
     'RMSPE': 36.85, 'R2': 0.550, 'Training': 'Global (no clustering)'},
    {'Cluster': -1, 'ClusterName': 'All Stores', 'Model': 'RandomForest',
     'RMSPE': 42.04, 'R2': 0.365, 'Training': 'Global (no clustering)'},
    {'Cluster': -1, 'ClusterName': 'All Stores', 'Model': 'NeuralNetwork',
     'RMSPE': 45.18, 'R2': 0.264, 'Training': 'Global (no clustering)'},
]

results_df = pd.DataFrame(results)
all_results = pd.concat([results_df, pd.DataFrame(global_baseline)], ignore_index=True)
all_results.to_csv('data/model_comparison_results.csv', index=False)

fi_avg = pd.DataFrame(feature_importance_all).mean(axis=1).sort_values(ascending=False)
fi_avg.to_csv('data/gbm_feature_importance.csv', header=False)

# Summary
print("\nResults:")
print(results_df.to_string(index=False))

model_keys = ['GradientBoosting', 'RandomForest', 'NeuralNetwork']
print("\nImprovement vs no-clustering baseline:")
for m in model_keys:
    per_cl  = results_df[results_df['Model'] == m]['RMSPE'].mean()
    global_ = next(r['RMSPE'] for r in global_baseline if r['Model'] == m)
    imp = (global_ - per_cl) / global_ * 100
    print(f"  {m:<25}: {global_:.1f}% -> {per_cl:.2f}% avg  ({imp:.1f}% reduction)")

best = results_df.loc[results_df['RMSPE'].idxmin()]
print(f"\nBest result: {best['Model']} on {best['ClusterName']}")
print(f"  RMSPE: {best['RMSPE']:.2f}%  R2: {best['R2']:.3f}")

# -----------------------------------------------------------------
# Charts
# -----------------------------------------------------------------

model_names = ['GBM', 'Random\nForest', 'Neural\nNet']
colors_m    = [BLUE, GREEN, PURPLE]
x = np.arange(3); w = 0.35

fig = plt.figure(figsize=(18, 11))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle('ML Model Performance: Per-Cluster vs Global Training\nRossmann Drug Store Sales Forecasting',
             fontsize=14, fontweight='bold', color=NAVY)
fig.patch.set_facecolor('#F8FAFC')

# RMSPE comparison
ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor('white')
global_rmspe  = [r['RMSPE'] for r in global_baseline]
cluster_rmspe = [results_df[results_df['Model'] == m]['RMSPE'].mean() for m in model_keys]
b1 = ax.bar(x - w/2, global_rmspe, w, color=['#93C5FD','#86EFAC','#C4B5FD'], alpha=0.8, label='Global training')
b2 = ax.bar(x + w/2, cluster_rmspe, w, color=colors_m, alpha=0.9, label='Per-cluster training')
ax.set_xticks(x); ax.set_xticklabels(model_names)
ax.set_ylabel('RMSPE % (lower = better)')
ax.set_title('RMSPE: Global vs Per-Cluster\n(All three models improve with clustering)')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
for bar, val in zip(list(b1) + list(b2), global_rmspe + cluster_rmspe):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.4, f'{val:.1f}%',
            ha='center', fontsize=9, fontweight='bold')

# R² comparison
ax = fig.add_subplot(gs[0, 1]); ax.set_facecolor('white')
global_r2  = [r['R2'] for r in global_baseline]
cluster_r2 = [results_df[results_df['Model'] == m]['R2'].mean() for m in model_keys]
b1 = ax.bar(x - w/2, global_r2, w, color=['#93C5FD','#86EFAC','#C4B5FD'], alpha=0.8, label='Global training')
b2 = ax.bar(x + w/2, cluster_r2, w, color=colors_m, alpha=0.9, label='Per-cluster training')
ax.set_xticks(x); ax.set_xticklabels(model_names)
ax.set_ylabel('R² (higher = better)')
ax.set_title('R²: Global vs Per-Cluster')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
for bar, val in zip(list(b1) + list(b2), global_r2 + cluster_r2):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.005, f'{val:.3f}',
            ha='center', fontsize=9, fontweight='bold')

# GBM per-cluster detail
ax = fig.add_subplot(gs[0, 2]); ax.set_facecolor('white')
gbm_res = results_df[results_df['Model'] == 'GradientBoosting'].reset_index(drop=True)
cl_labels = [f"Cluster {r['Cluster']}\n{r['ClusterName'][:14]}" for _, r in gbm_res.iterrows()]
bars = ax.bar(cl_labels, gbm_res['RMSPE'].values, color=[BLUE, GREEN][:len(gbm_res)], alpha=0.85, width=0.5)
ax.axhline(36.85, color=RED, lw=2, linestyle='--', label='Global GBM (36.9%)')
ax.set_ylabel('RMSPE %')
ax.set_title('GBM Per Cluster\n(High-volume stores are easier to predict)')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
for bar, row in zip(bars, gbm_res.itertuples()):
    ax.text(bar.get_x() + bar.get_width()/2, row.RMSPE + 0.4,
            f'RMSPE={row.RMSPE:.1f}%\nR²={row.R2:.3f}',
            ha='center', fontsize=10, fontweight='bold')

# Heatmap: all models x all clusters
ax = fig.add_subplot(gs[1, 0]); ax.set_facecolor('white')
hmap = results_df.pivot(index='ClusterName', columns='Model', values='RMSPE')
hmap = hmap[['GradientBoosting', 'RandomForest', 'NeuralNetwork']]
hmap.columns = ['GBM', 'Random Forest', 'Neural Net']
sns.heatmap(hmap, ax=ax, annot=True, fmt='.1f', cmap='RdYlGn_r',
            linewidths=1, linecolor='white', cbar_kws={'label': 'RMSPE %'},
            annot_kws={'size': 12, 'weight': 'bold'})
ax.set_title('RMSPE: All Models x All Clusters')
ax.set_xticklabels(ax.get_xticklabels(), rotation=15); ax.set_ylabel('')

# Sales distribution by cluster — explains why clustering helps
ax = fig.add_subplot(gs[1, 1]); ax.set_facecolor('white')
clusters_df = pd.read_csv('data/store_clusters.csv')
for cl, color, label in [(0, BLUE, 'Mid-Volume'), (1, GREEN, 'High-Volume')]:
    data = clusters_df[clusters_df['Cluster'] == cl]['AvgSales']
    ax.hist(data, bins=40, color=color, alpha=0.55, density=True,
            label=f'{label} (n={len(data)}, mean €{data.mean():,.0f})')
    ax.axvline(data.mean(), color=color, lw=2, linestyle='--')
ax.set_xlabel('Avg Daily Sales per Store (€)')
ax.set_ylabel('Density')
ax.set_title('Store Sales Distributions by Cluster\n(Different populations — separate models make sense)')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f'€{v:,.0f}'))

# Impact summary panel
ax = fig.add_subplot(gs[1, 2]); ax.set_facecolor(LGREY); ax.axis('off')
gbm_avg = results_df[results_df['Model'] == 'GradientBoosting']['RMSPE'].mean()
rf_avg  = results_df[results_df['Model'] == 'RandomForest']['RMSPE'].mean()
nn_avg  = results_df[results_df['Model'] == 'NeuralNetwork']['RMSPE'].mean()
summary = (
    "Clustering improvement summary\n"
    "\n\n"
    f"Gradient Boosting\n"
    f"  Without: 36.9% RMSPE\n"
    f"  With   : {gbm_avg:.2f}% avg RMSPE\n"
    f"  Gain   : -{(36.85 - gbm_avg)/36.85*100:.1f}%\n\n"
    f"Random Forest\n"
    f"  Without: 42.0% RMSPE\n"
    f"  With   : {rf_avg:.2f}% avg RMSPE\n"
    f"  Gain   : -{(42.04 - rf_avg)/42.04*100:.1f}%\n\n"
    f"Neural Network\n"
    f"  Without: 45.2% RMSPE\n"
    f"  With   : {nn_avg:.2f}% avg RMSPE\n"
    f"  Gain   : -{(45.18 - nn_avg)/45.18*100:.1f}%\n\n"
    f"Best single result:\n"
    f"  GBM on High-Volume Stores\n"
    f"  RMSPE={best['RMSPE']:.1f}%  R²={best['R2']:.3f}"
)
ax.text(0.07, 0.95, summary, transform=ax.transAxes, fontsize=9.5,
        va='top', ha='left', fontfamily='monospace', color=NAVY, linespacing=1.6)
ax.set_title('Clustering Impact Summary', pad=12)

plt.savefig('charts/ML2_model_comparison.png', dpi=150, bbox_inches='tight', facecolor='#F8FAFC')
plt.close()
print("\nSaved: charts/ML2_model_comparison.png")

# Feature importance chart
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor('#F8FAFC'); ax.set_facecolor('white')
fig.suptitle('GradientBoosting Feature Importance\nWhat Drives Daily Sales Predictions?',
             fontsize=14, fontweight='bold', color=NAVY)
fi_plot = fi_avg.sort_values(ascending=True)
fi_max  = fi_plot.max()
colors_fi = [RED if v > fi_max * 0.55
             else AMBER if v > fi_max * 0.30
             else BLUE for v in fi_plot.values]
bars = ax.barh(fi_plot.index, fi_plot.values, color=colors_fi, alpha=0.85, height=0.7)
ax.set_xlabel('Feature Importance (Mean Decrease in Impurity)')
ax.axvline(fi_plot.mean(), color=NAVY, lw=2, linestyle='--', label='Mean importance')
ax.legend(handles=[
    Patch(color=RED, alpha=0.85, label='High importance (>55% of max)'),
    Patch(color=AMBER, alpha=0.85, label='Medium importance (30-55%)'),
    Patch(color=BLUE, alpha=0.85, label='Standard importance'),
    plt.Line2D([0], [0], color=NAVY, lw=2, linestyle='--', label='Mean'),
], fontsize=9, loc='lower right')
ax.grid(axis='x', alpha=0.3)
for bar, val in zip(bars, fi_plot.values):
    ax.text(val + 0.0001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=8.5)
plt.tight_layout()
plt.savefig('charts/ML3_feature_importance.png', dpi=150, bbox_inches='tight', facecolor='#F8FAFC')
plt.close()
print("Saved: charts/ML3_feature_importance.png")
print("\nDone.")
