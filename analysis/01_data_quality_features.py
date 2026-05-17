# Rossmann Drug Store Sales Forecasting
# Script 1: Data Quality Audit and Feature Engineering
# Vaishnavi Jitendra Bhor | MSc Business Analytics, University of Manchester
#
# Inputs:
#   data/DA2024_train.csv   — 1,017,209 daily sales records
#   data/DA2024_stores.csv  — 1,115 store metadata rows
#   data/DA2024_test.csv    — 41,088 rows, Sales column unknown
#
# Outputs:
#   data/rossmann_train_features.csv  — 844,338 rows, 34 columns
#   data/rossmann_test_features.csv   — 41,088 rows, 33 columns
#
# Run from repo root: python analysis/01_data_quality_features.py

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('data',   exist_ok=True)
os.makedirs('charts', exist_ok=True)

print("Loading raw data...")
train  = pd.read_csv('data/DA2024_train.csv',  low_memory=False)
stores = pd.read_csv('data/DA2024_stores.csv')
test   = pd.read_csv('data/DA2024_test.csv',   low_memory=False)

print(f"  Train  : {len(train):,} rows, {len(train.columns)} columns")
print(f"  Stores : {len(stores):,} rows, {len(stores.columns)} columns")
print(f"  Test   : {len(test):,} rows, {len(test.columns)} columns")

train['Date'] = pd.to_datetime(train['Date'], dayfirst=True)
test['Date']  = pd.to_datetime(test['Date'],  dayfirst=True)

print(f"\n  Training period : {train['Date'].min().date()} to {train['Date'].max().date()}")
print(f"  Test period     : {test['Date'].min().date()} to {test['Date'].max().date()}")

# -----------------------------------------------------------------
# Data quality checks
# Checking these before any analysis — the silent errors are the
# dangerous ones because they don't throw exceptions.
# -----------------------------------------------------------------

print("\nRunning data quality checks...")

# StateHoliday has a type inconsistency — some rows stored as
# string "0", others as integer 0. They're not equal in Python,
# so any filter on this column returns roughly half the expected rows.
print(f"  StateHoliday values: {train['StateHoliday'].value_counts().to_dict()}")

# Records where the store is marked open but Sales = 0
# These are data entry errors — 54 records out of 1M.
anomaly = train[(train['Open'] == 1) & (train['Sales'] == 0)]
print(f"  Open=1 / Sales=0 records: {len(anomaly):,} (will be removed)")

# 180 stores have no records for July–December 2014.
# Keeping them in — GBM handles sparse data well, and imputing
# six months per store would introduce more noise than signal.
mid = train[(train['Date'] >= '2014-07-01') & (train['Date'] <= '2014-12-31')]
active_mid = mid[mid['Open'] == 1]['Store'].nunique()
print(f"  Stores active Jul–Dec 2014: {active_mid} of 1,115 ({1115-active_mid} have a gap in this period)")

print(f"  Test set Open nulls: {test['Open'].isna().sum()} (will fill with 1 — assume open)")

# -----------------------------------------------------------------
# Fix stores.csv
# -----------------------------------------------------------------

print("\nCleaning stores data...")

# Two completely empty columns — safe to drop
unnamed = [c for c in stores.columns if 'Unnamed' in c]
stores.drop(columns=unnamed, inplace=True)
print(f"  Dropped {len(unnamed)} empty unnamed columns: {unnamed}")

# 3 stores missing CompetitionDistance — fill with median
med_dist = stores['CompetitionDistance'].median()
n_missing_dist = stores['CompetitionDistance'].isna().sum()
stores['CompetitionDistance'].fillna(med_dist, inplace=True)
print(f"  CompetitionDistance: {n_missing_dist} missing values filled with median ({med_dist:,.0f}m)")

# Stores with no competitor have no CompetitionOpenSince date
# Filling with 0 means "no competitor" rather than "unknown"
n_missing_since = stores['CompetitionOpenSinceMonth'].isna().sum()
stores['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
stores['CompetitionOpenSinceYear'].fillna(0, inplace=True)
print(f"  CompetitionOpenSince: {n_missing_since} missing filled with 0 (no competitor)")

# Promo2Since nulls are stores that don't participate in the scheme
n_missing_promo2 = stores['Promo2SinceWeek'].isna().sum()
stores['Promo2SinceWeek'].fillna(0, inplace=True)
stores['Promo2SinceYear'].fillna(0, inplace=True)
stores['PromoInterval'].fillna('None', inplace=True)
print(f"  Promo2Since: {n_missing_promo2} missing filled with 0 (non-participants)")

# -----------------------------------------------------------------
# Fix train and test
# -----------------------------------------------------------------

print("\nCleaning training and test data...")

# Standardise StateHoliday to integers: 0=none, 1=public, 2=Easter, 3=Christmas
holiday_map = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 0: 0, 1: 1, 2: 2, 3: 3}
train['StateHoliday'] = train['StateHoliday'].map(
    lambda x: holiday_map.get(str(x).strip(), 0)).astype(int)
test['StateHoliday'] = test['StateHoliday'].map(
    lambda x: holiday_map.get(str(x).strip(), 0)).astype(int)
print("  StateHoliday standardised: 0=none, 1=public holiday, 2=Easter, 3=Christmas")

# Fill the 11 Open nulls in test with 1 — the test days were
# presumably selected because stores were expected to be open
n_open_null = test['Open'].isna().sum()
test['Open'].fillna(1, inplace=True)
print(f"  Test Open: {n_open_null} missing values filled with 1")

# Remove the 54 Open=1/Sales=0 records
rows_before = len(train)
train = train[~((train['Open'] == 1) & (train['Sales'] == 0))].copy()
print(f"  Removed {rows_before - len(train)} Open=1/Sales=0 records")

# Only train on days when stores were actually open
# Closed days have Sales=0 by definition — there's nothing to learn
train_open = train[train['Open'] == 1].copy()
print(f"  Training rows (open stores): {len(train_open):,}")
print(f"  Closed days excluded: {len(train) - len(train_open):,}")

# -----------------------------------------------------------------
# Merge store metadata into train and test
# -----------------------------------------------------------------

print("\nMerging store metadata...")
train_merged = train_open.merge(stores, on='Store', how='left')
test_merged  = test.merge(stores, on='Store', how='left')
print(f"  Train after merge: {train_merged.shape}")
print(f"  Test after merge : {test_merged.shape}")

# -----------------------------------------------------------------
# Feature engineering
# 16 new columns built from existing data
# -----------------------------------------------------------------

print("\nEngineering features...")

def engineer_features(df):
    df = df.copy()

    # Temporal features — day, week, month, year patterns
    df['Year']         = df['Date'].dt.year
    df['Month']        = df['Date'].dt.month
    df['Day']          = df['Date'].dt.day
    df['WeekOfYear']   = df['Date'].dt.isocalendar().week.astype(int)
    df['DayOfYear']    = df['Date'].dt.dayofyear
    df['Quarter']      = df['Date'].dt.quarter
    df['IsWeekend']    = (df['DayOfWeek'] >= 6).astype(int)
    df['IsMonthStart'] = (df['Day'] <= 5).astype(int)
    df['IsMonthEnd']   = (df['Day'] >= 25).astype(int)

    # Season: 1=Spring, 2=Summer, 3=Autumn, 4=Winter
    df['Season'] = df['Month'].map({
        12: 4, 1: 4, 2: 4,
        3: 1,  4: 1, 5: 1,
        6: 2,  7: 2, 8: 2,
        9: 3, 10: 3, 11: 3
    })

    # Competition features
    # Months since the nearest competitor opened.
    # Recent entrants (small value) suppress sales more than established ones.
    df['CompetitionOpen'] = (
        12 * (df['Year']  - df['CompetitionOpenSinceYear'])
        +    (df['Month'] - df['CompetitionOpenSinceMonth'])
    ).clip(lower=0).fillna(0)

    # Log-transform distance: raw range is 20m–75,860m (4,000x).
    # Log compression makes the isolation premium detectable by the model.
    df['LogCompetitionDistance'] = np.log1p(df['CompetitionDistance'])

    # Promotion features
    # How long has this store been enrolled in Promo2?
    df['Promo2Open'] = (
        12 * (df['Year']  - df['Promo2SinceYear'])
        +    (df['WeekOfYear'] - df['Promo2SinceWeek']) / 4.0
    ).clip(lower=0)
    df['Promo2Open'] = df['Promo2Open'].where(df['Promo2'] == 1, 0).fillna(0)

    # Is the current month an active Promo2 mailing month for this store?
    # More precise than the scheme participation flag alone.
    month_abbr = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                  7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    df['_mabbr'] = df['Month'].map(month_abbr)
    df['IsPromo2Month'] = df.apply(
        lambda r: 1 if (r['Promo2'] == 1
                        and isinstance(r['PromoInterval'], str)
                        and r['_mabbr'] in r['PromoInterval'])
        else 0, axis=1)
    df.drop(columns=['_mabbr'], inplace=True)

    # Ordinal encodings for categorical store attributes
    df['StoreType_enc']  = df['StoreType'].map({'a': 0, 'b': 1, 'c': 2, 'd': 3}).fillna(0).astype(int)
    df['Assortment_enc'] = df['Assortment'].map({'a': 0, 'b': 1, 'c': 2}).fillna(0).astype(int)

    return df

train_fe = engineer_features(train_merged)
test_fe  = engineer_features(test_merged)

new_features = [
    'Year','Month','Day','WeekOfYear','DayOfYear','Quarter',
    'IsWeekend','IsMonthStart','IsMonthEnd','Season',
    'CompetitionOpen','LogCompetitionDistance',
    'Promo2Open','IsPromo2Month','StoreType_enc','Assortment_enc'
]
print(f"  New features ({len(new_features)}): {new_features}")
print(f"  Train shape: {train_fe.shape}")
print(f"  Test shape : {test_fe.shape}")

# -----------------------------------------------------------------
# Save outputs
# -----------------------------------------------------------------

print("\nSaving outputs...")
train_fe.to_csv('data/rossmann_train_features.csv', index=False)
test_fe.to_csv('data/rossmann_test_features.csv',  index=False)
print(f"  Saved: rossmann_train_features.csv ({len(train_fe):,} rows, {train_fe.shape[1]} columns)")
print(f"  Saved: rossmann_test_features.csv  ({len(test_fe):,} rows, {test_fe.shape[1]} columns)")

print("\nSummary:")
print(f"  Raw records loaded       : 1,017,209")
print(f"  Anomalies removed        : {rows_before - len(train)}")
print(f"  Closed-day rows excluded : {len(train) - len(train_open):,}")
print(f"  Final training rows      : {len(train_fe):,}")
print(f"  Test rows                : {len(test_fe):,}")
print(f"  Features                 : {train_fe.shape[1]} ({len(new_features)} newly engineered)")
