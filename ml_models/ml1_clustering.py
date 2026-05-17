# Rossmann Drug Store Sales Forecasting
# ML Script 1: K-Means Store Clustering
# Vaishnavi Jitendra Bhor | MSc Business Analytics, University of Manchester
#
# Segments 1,115 stores by sales profile before any forecasting models are trained.
# The idea is that mid-volume and high-volume stores have different seasonal
# patterns and promo response rates, so training within clusters gives better
# results than one model across all stores.
#
# Inputs : data/rossmann_train_features.csv
# Outputs: data/store_clusters.csv
#          data/rossmann_train_clustered.csv
#          charts/ML1_clustering.png
#
# Run: python ml_models/ml1_clustering.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings; warnings.filterwarnings('ignore')

plt.rcParams.update({'figure.facecolor':'#F8FAFC','axes.facecolor':'#F8FAFC',
    'axes.spines.top':False,'axes.spines.right':False,
    'font.family':'DejaVu Sans','axes.titlesize':12,'axes.titleweight':'bold'})
BLUE='#2563EB';GREEN='#16A34A';RED='#DC2626';AMBER='#D97706';NAVY='#1E3A5F'

print("="*60)
print("ML MODEL 1: K-MEANS STORE CLUSTERING")
print("="*60)

df = pd.read_csv('data/rossmann_train_features.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])

# Build store-level feature matrix for clustering
print("Building store-level feature matrix...")
store_feats = df.groupby('Store').agg(
    AvgSales      = ('Sales','mean'),
    StdSales      = ('Sales','std'),
    MedianSales   = ('Sales','median'),
    MaxSales      = ('Sales','max'),
    MinSales      = ('Sales','min'),
    AvgCustomers  = ('Customers','mean'),
    PromoRate     = ('Promo','mean'),
    AvgATV        = ('ATV','mean') if 'ATV' in df.columns else ('Sales','mean'),
    StoreType     = ('StoreType_enc','first'),
    Assortment    = ('Assortment_enc','first'),
    CompDist      = ('CompetitionDistance','first'),
    CompOpen      = ('CompetitionOpen','mean'),
    Promo2        = ('Promo2','first'),
).reset_index()
store_feats['CV'] = store_feats['StdSales'] / store_feats['AvgSales']
store_feats.fillna(0, inplace=True)
print(f"  Store feature matrix: {store_feats.shape}")

# Scale
cluster_cols = ['AvgSales','StdSales','MedianSales','MaxSales',
                'AvgCustomers','PromoRate','CV','CompDist',
                'StoreType','Assortment']
X = store_feats[cluster_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal K
print("Finding optimal K...")
inertias, silhouettes = [], []
K_range = range(2, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    print(f"  K={k}: Silhouette={silhouette_score(X_scaled,labels):.3f}  Inertia={km.inertia_:,.0f}")

best_k = K_range[np.argmax(silhouettes)]
print(f"\n  Optimal K = {best_k} (best Silhouette Score)")

# Fit final model
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
store_feats['Cluster'] = km_final.fit_predict(X_scaled)

# Profile clusters
print("Cluster profiles:")
profile = store_feats.groupby('Cluster')[
    ['AvgSales','AvgCustomers','PromoRate','CompDist','CV']
].mean().round(0)
print(profile.to_string())

cluster_names = {}
for c in range(best_k):
    avg = store_feats[store_feats['Cluster']==c]['AvgSales'].mean()
    n   = (store_feats['Cluster']==c).sum()
    if avg > store_feats['AvgSales'].mean() * 1.3:
        name = 'HIGH-VOLUME STORES'
    elif avg < store_feats['AvgSales'].mean() * 0.7:
        name = 'LOW-VOLUME STORES'
    else:
        name = 'MID-VOLUME STORES'
    cluster_names[c] = name
    print(f"  Cluster {c}: {name} ({n} stores, avg €{avg:,.0f}/day)")

store_feats['ClusterName'] = store_feats['Cluster'].map(cluster_names)

# Save cluster assignments
cluster_out = store_feats[['Store','Cluster','ClusterName','AvgSales','StdSales','CV']]
cluster_out.to_csv('data/store_clusters.csv', index=False)
print(f"\n  ✓ store_clusters.csv saved")

# Merge cluster back to main data
df = df.merge(store_feats[['Store','Cluster','ClusterName']], on='Store', how='left')
df.to_csv('data/rossmann_train_clustered.csv', index=False)
print(f"  ✓ rossmann_train_clustered.csv saved ({len(df):,} rows)")

# Charts
print("Generating charts...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'K-Means Store Clustering (K={best_k})  |  Segmenting 1,115 Stores by Sales Profile',
             fontsize=13, fontweight='bold', color=NAVY)
fig.patch.set_facecolor('#F8FAFC')

# Elbow + Silhouette
ax = axes[0]; ax.set_facecolor('white')
ax2 = ax.twinx()
ax.plot(list(K_range), inertias, color=BLUE, lw=2.2, marker='o', ms=6, label='Inertia')
ax2.plot(list(K_range), silhouettes, color=GREEN, lw=2.2, marker='s', ms=6, label='Silhouette')
ax.axvline(best_k, color=RED, lw=2, linestyle='--', label=f'Optimal K={best_k}')
ax.set_xlabel('Number of Clusters (K)')
ax.set_ylabel('Inertia', color=BLUE); ax2.set_ylabel('Silhouette Score', color=GREEN)
ax2.spines['top'].set_visible(False)
ax.set_title(f'Optimal K Selection\n(K={best_k} maximises Silhouette Score)')
lines1, lbl1 = ax.get_legend_handles_labels()
lines2, lbl2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, lbl1+lbl2, fontsize=8); ax.grid(alpha=0.3)

# PCA visualisation
ax = axes[1]; ax.set_facecolor('white')
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
colors_cl = [BLUE, GREEN, RED, AMBER][:best_k]
for c in range(best_k):
    mask = store_feats['Cluster'] == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               color=colors_cl[c], alpha=0.7, s=30,
               label=f'C{c}: {cluster_names[c][:15]} ({mask.sum()})',
               edgecolors='white', lw=0.5)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)')
ax.set_title(f'PCA 2D Cluster Visualisation\n(Var explained: {sum(pca.explained_variance_ratio_)*100:.0f}%)')
ax.legend(fontsize=7); ax.grid(alpha=0.2)

# Sales distribution per cluster
ax = axes[2]; ax.set_facecolor('white')
for c in range(best_k):
    cluster_sales = df[df['Cluster']==c]['Sales']
    ax.hist(cluster_sales[cluster_sales<20000], bins=60,
            color=colors_cl[c], alpha=0.55,
            label=f'C{c}: {cluster_names[c][:12]}', density=True)
ax.set_xlabel('Daily Sales (€)'); ax.set_ylabel('Density')
ax.set_title('Sales Distribution by Cluster\n(Distinct peaks confirm separation)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('charts/ML1_clustering.png', dpi=150, bbox_inches='tight', facecolor='#F8FAFC')
plt.close()
print("  ✓ ML1_clustering.png")

print(f"\n✓ ML MODEL 1 COMPLETE")
print(f"  Clusters: {best_k} | Silhouette: {max(silhouettes):.3f}")
print(f"  Cluster assignments saved — will train separate models per cluster")
