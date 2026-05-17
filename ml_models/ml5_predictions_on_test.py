# Rossmann Drug Store Sales Forecasting
# ML Script 5: Sales Predictions on Test Set
# Vaishnavi Jitendra Bhor | MSc Business Analytics, University of Manchester
#
# Trains the best model (GradientBoosting) on the full training set and
# generates daily sales predictions for all 41,088 records in the test set
# covering August 1 to September 17, 2015.
#
# Two output files are produced:
#   rossmann_predictions.csv              — daily prediction per store per day
#   rossmann_store_predictions_summary.csv — 6-week total and average per store
#
# The summary file is what a store manager would actually use to plan
# staffing rotas and stock orders for the six-week window.
#
# Inputs : data/rossmann_train_features.csv
#          data/rossmann_test_features.csv
#          data/store_clusters.csv
#
# Run: python ml_models/ml5_predictions_on_test.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import warnings, os
warnings.filterwarnings('ignore')
os.makedirs('charts', exist_ok=True)
os.makedirs('data',   exist_ok=True)

plt.rcParams.update({
    'figure.facecolor':'#F8FAFC','axes.facecolor':'#F8FAFC',
    'axes.spines.top':False,'axes.spines.right':False,
    'font.family':'DejaVu Sans','axes.titlesize':12,'axes.titleweight':'bold'
})
BLUE='#2563EB';GREEN='#16A34A';RED='#DC2626';AMBER='#D97706'
PURPLE='#7C3AED';NAVY='#1E3A5F'

print("=" * 60)
print("ML SCRIPT 5: TEST SET SALES PREDICTIONS")
print("=" * 60)

#  Load data 

train = pd.read_csv('data/rossmann_train_features.csv', low_memory=False)
test  = pd.read_csv('data/rossmann_test_features.csv',  low_memory=False)
clusters = pd.read_csv('data/store_clusters.csv')

train['Date'] = pd.to_datetime(train['Date'])
test['Date']  = pd.to_datetime(test['Date'])

print(f"  Training rows  : {len(train):,}")
print(f"  Test rows      : {len(test):,}")
print(f"  Train period   : {train['Date'].min().date()} to {train['Date'].max().date()}")
print(f"  Test period    : {test['Date'].min().date()} to {test['Date'].max().date()}")

# Merge cluster labels
train = train.merge(clusters[['Store','Cluster']], on='Store', how='left')
test  = test.merge(clusters[['Store','Cluster']],  on='Store', how='left')
test['Cluster'] = test['Cluster'].fillna(0).astype(int)

#  Feature set 
FEATURES = [
    'Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
    'Month', 'Day', 'WeekOfYear', 'Year', 'Quarter', 'IsWeekend',
    'IsMonthStart', 'IsMonthEnd', 'Season', 'CompetitionOpen',
    'LogCompetitionDistance', 'Promo2Open', 'IsPromo2Month',
    'StoreType_enc', 'Assortment_enc', 'Promo2', 'Cluster'
]
TARGET = 'Sales'

#  Train best model on FULL training data 

print("  Note: Training on all 844,338 rows (no sampling) for best accuracy")
X_train = train[FEATURES].fillna(0)
y_train = train[TARGET]

print(f"  Training samples: {len(X_train):,}")
print(f"  Features used:    {len(FEATURES)}")

# Best model config from ML2-4 comparison
gbm = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    min_samples_leaf=20,
    subsample=0.8,
    random_state=42,
    verbose=0
)

print("  Fitting model (this takes ~5 minutes on full data)...")
gbm.fit(X_train, y_train)
print(f"  ✓ Model trained on {len(X_train):,} rows")

#  Generate predictions on test set 

X_test = test[FEATURES].fillna(0)

# Stores closed on test days → set prediction to 0
test_preds = gbm.predict(X_test).clip(min=0)

# Override: stores marked as closed get 0 sales
closed_mask = test['Open'] == 0
test_preds[closed_mask] = 0
n_closed = closed_mask.sum()
print(f"  Predictions generated: {len(test_preds):,}")
print(f"  Closed store days:     {n_closed:,} → Sales set to 0")
print(f"  Open store predictions: {(~closed_mask).sum():,}")
print(f"\n  Prediction summary:")
open_preds = test_preds[~closed_mask]
print(f"    Mean predicted sales : €{open_preds.mean():,.0f}")
print(f"    Median predicted sales: €{np.median(open_preds):,.0f}")
print(f"    Min predicted sales  : €{open_preds.min():,.0f}")
print(f"    Max predicted sales  : €{open_preds.max():,.0f}")

#  Save predictions 

output = test[['Store','Date','DayOfWeek','Open','Promo',
               'StateHoliday','SchoolHoliday','Cluster']].copy()
output['Predicted_Sales'] = test_preds.round(0).astype(int)
output['Date'] = output['Date'].dt.strftime('%Y-%m-%d')

output.to_csv('data/rossmann_predictions.csv', index=False)
print(f"  ✓ data/rossmann_predictions.csv — {len(output):,} rows")

# Store-level summary
store_summary = output[output['Open']==1].groupby('Store').agg(
    Days_Predicted=('Date','count'),
    Avg_Daily_Sales=('Predicted_Sales','mean'),
    Total_Sales_6wk=('Predicted_Sales','sum')
).reset_index().round(0)
store_summary.to_csv('data/rossmann_store_predictions_summary.csv', index=False)
print(f"  ✓ data/rossmann_store_predictions_summary.csv — {len(store_summary):,} stores")

#  Validation: compare train tail vs test predictions 

last_6wk_train = train[train['Date'] >= train['Date'].max() - pd.Timedelta(weeks=6)]
train_avg = last_6wk_train.groupby('DayOfWeek')['Sales'].mean()
test_avg  = output[output['Open']==1].copy()
test_avg['Date'] = pd.to_datetime(test_avg['Date'])
test_avg['DayOfWeek'] = test_avg['DayOfWeek']
test_dow  = test_avg.groupby('DayOfWeek')['Predicted_Sales'].mean()

print(f"\n  DoW avg comparison (last 6 wk train vs 6 wk predictions):")
days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
print(f"  {'Day':<6} {'Train Avg':>12} {'Pred Avg':>12} {'Diff %':>10}")
for d in range(1,8):
    t = train_avg.get(d, 0)
    p = test_dow.get(d, 0)
    diff = (p-t)/t*100 if t>0 else 0
    print(f"  {days[d-1]:<6} €{t:>10,.0f} €{p:>10,.0f} {diff:>+9.1f}%")

#  Charts 

fig = plt.figure(figsize=(18,10))
gs  = gridspec.GridSpec(2,3,figure=fig,hspace=0.4,wspace=0.35)
fig.suptitle('6-Week Sales Predictions on Test Set  |  Aug–Sep 2015  |  1,115 Stores',
             fontsize=14,fontweight='bold',color=NAVY)
fig.patch.set_facecolor('#F8FAFC')

# Prediction distribution
ax=fig.add_subplot(gs[0,0]); ax.set_facecolor('white')
ax.hist(open_preds[open_preds<20000],bins=80,color=BLUE,alpha=0.7,edgecolor='white',lw=0.3)
ax.axvline(open_preds.mean(),color=RED,lw=2,linestyle='--',label=f"Mean: €{open_preds.mean():,.0f}")
ax.axvline(np.median(open_preds),color=GREEN,lw=2,linestyle='--',label=f"Median: €{np.median(open_preds):,.0f}")
ax.set_xlabel('Predicted Daily Sales (€)'); ax.set_ylabel('Number of Stores')
ax.set_title('Predicted Sales Distribution\n(Aug-Sep 2015 test period)')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Predicted sales by day of week
ax=fig.add_subplot(gs[0,1]); ax.set_facecolor('white')
pred_dow = output[output['Open']==1].groupby('DayOfWeek')['Predicted_Sales'].mean()
ax.bar(days,pred_dow.values,
       color=[GREEN if i<5 else AMBER for i in range(7)],alpha=0.85)
ax.set_ylabel('Avg Predicted Sales (€)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,p:f'€{v:,.0f}'))
ax.set_title('Predicted Sales by Day of Week\n(Monday peak matches historical pattern)')
ax.grid(axis='y',alpha=0.3)

# Daily predicted total across all stores
ax=fig.add_subplot(gs[0,2]); ax.set_facecolor('white')
output_open = output[output['Open']==1].copy()
output_open['Date_dt'] = pd.to_datetime(output_open['Date'])
daily_total = output_open.groupby('Date_dt')['Predicted_Sales'].sum() / 1e6
ax.fill_between(range(len(daily_total)),daily_total.values,alpha=0.15,color=PURPLE)
ax.plot(range(len(daily_total)),daily_total.values,color=PURPLE,lw=2)
ax.set_xticks(range(0,len(daily_total),7))
ax.set_xticklabels(daily_total.index[::7].strftime('%d %b'),rotation=45,ha='right')
ax.set_ylabel('Total Predicted Sales (€M)')
ax.set_title('Daily Predicted Revenue — All Stores\n(6-week forecast window)')
ax.grid(alpha=0.3)

# Train vs predicted comparison
ax=fig.add_subplot(gs[1,:2]); ax.set_facecolor('white')
train_monthly = train.groupby(train['Date'].dt.to_period('M').astype(str))['Sales'].sum()/1e6
output_open['Month'] = pd.to_datetime(output_open['Date']).dt.to_period('M').astype(str)
pred_monthly  = output_open.groupby('Month')['Predicted_Sales'].sum()/1e6
all_monthly   = pd.concat([train_monthly.rename('Historical'), pred_monthly.rename('Predicted')])
hist_idx = list(range(len(train_monthly)))
pred_idx = list(range(len(train_monthly), len(train_monthly)+len(pred_monthly)))
ax.fill_between(hist_idx,train_monthly.values,alpha=0.15,color=BLUE)
ax.plot(hist_idx,train_monthly.values,color=BLUE,lw=2,label='Historical sales')
ax.fill_between(pred_idx,pred_monthly.values,alpha=0.2,color=GREEN)
ax.plot(pred_idx,pred_monthly.values,color=GREEN,lw=2.5,linestyle='--',
        marker='o',ms=7,label='Predicted (test set)')
ax.axvline(len(train_monthly)-0.5,color=RED,lw=2,linestyle=':',label='Prediction start')
n_ticks = len(train_monthly) + len(pred_monthly)
all_labels = list(train_monthly.index) + list(pred_monthly.index)
ax.set_xticks(range(0,n_ticks,4))
ax.set_xticklabels([all_labels[i] for i in range(0,n_ticks,4)],rotation=45,ha='right')
ax.set_ylabel('Total Monthly Sales (€M)')
ax.set_title('Historical vs Predicted Monthly Sales\n(Blue = actual history | Green = 6-week prediction)')
ax.legend(fontsize=10); ax.grid(alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,p:f'€{v:.0f}M'))

# Top 20 stores by predicted revenue
ax=fig.add_subplot(gs[1,2]); ax.set_facecolor('white')
top20 = store_summary.nlargest(20,'Total_Sales_6wk')
ax.barh(top20['Store'].astype(str),
        top20['Total_Sales_6wk']/1e3,
        color=BLUE,alpha=0.85,height=0.7)
ax.set_xlabel('Total 6-Week Predicted Sales (€K)')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,p:f'€{v:.0f}K'))
ax.set_title('Top 20 Stores by Predicted Revenue\n(6-week total)')
ax.grid(axis='x',alpha=0.3)

plt.savefig('charts/ML4_prediction_analysis.png',dpi=150,bbox_inches='tight')
plt.close()
print("  ✓ charts/ML4_prediction_analysis.png")

print("\n" + "="*60)
print("ML SCRIPT 5 COMPLETE")
print("="*60)
print(f"""
  Test set predicted : {len(output):,} records
  Stores covered     : {output['Store'].nunique():,}
  Date range         : Aug 01 – Sep 17, 2015
  Avg store revenue  : €{store_summary['Total_Sales_6wk'].mean():,.0f} over 6 weeks
  Top store revenue  : €{store_summary['Total_Sales_6wk'].max():,.0f} over 6 weeks
  Files saved:
    data/rossmann_predictions.csv
    data/rossmann_store_predictions_summary.csv
    charts/ML4_prediction_analysis.png
""")
