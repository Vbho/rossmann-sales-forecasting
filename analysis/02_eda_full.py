# Rossmann Drug Store Sales Forecasting
# Script 2: Exploratory Data Analysis
# Vaishnavi Jitendra Bhor | MSc Business Analytics, University of Manchester
#
# Generates 6 EDA charts covering sales patterns, store types,
# seasonality, promotions, competition, and feature correlations.
#
# Input : data/rossmann_train_features.csv
# Output: charts/E1_sales_overview.png through E6_correlation_matrix.png
#
# Run: python analysis/02_eda_full.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os, warnings
warnings.filterwarnings('ignore')
os.makedirs('charts', exist_ok=True)

plt.rcParams.update({
    'figure.facecolor':'#F8FAFC','axes.facecolor':'#F8FAFC',
    'axes.spines.top':False,'axes.spines.right':False,
    'font.family':'DejaVu Sans','axes.titlesize':12,
    'axes.titleweight':'bold','axes.labelsize':10,
    'xtick.labelsize':9,'ytick.labelsize':9
})
BLUE='#2563EB';GREEN='#16A34A';RED='#DC2626';AMBER='#D97706'
PURPLE='#7C3AED';TEAL='#0891B2';NAVY='#1E3A5F'
PAL=[BLUE,GREEN,RED,AMBER,PURPLE,TEAL,'#EA580C','#65A30D']

print("="*60)
print("Running EDA...")
print("="*60)

df = pd.read_csv('data/rossmann_train_features.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])
df['MonthYear'] = df['Date'].dt.to_period('M').astype(str)
df['ATV'] = df['Sales'] / df['Customers'].replace(0, np.nan)
print(f"Loaded: {len(df):,} rows, {df.shape[1]} cols")

#  E1: Sales Overview Dashboard 
print("  Chart E1: Sales overview dashboard...")
fig = plt.figure(figsize=(18,10))
gs  = gridspec.GridSpec(2,3,figure=fig,hspace=0.4,wspace=0.35)
fig.suptitle('Rossmann Drug Stores — Sales Overview  |  Jan 2013 – Jul 2015  |  1,115 Stores',
             fontsize=15,fontweight='bold',color=NAVY)

monthly = df.groupby('MonthYear')['Sales'].sum().reset_index()
monthly['Date'] = pd.to_datetime(monthly['MonthYear'])
monthly = monthly.sort_values('Date')
ax=fig.add_subplot(gs[0,:2]); ax.set_facecolor('white')
ax.fill_between(range(len(monthly)),monthly['Sales']/1e6,alpha=0.15,color=BLUE)
ax.plot(range(len(monthly)),monthly['Sales']/1e6,color=BLUE,lw=2.5)
ax.set_xticks(range(0,len(monthly),6))
ax.set_xticklabels(monthly['MonthYear'].iloc[::6],rotation=45,ha='right')
ax.set_ylabel('Total Sales (€M)'); ax.grid(alpha=0.3)
ax.set_title('Monthly Total Sales Trend  |  All 1,115 Stores')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,p:f'€{v:.0f}M'))

ax=fig.add_subplot(gs[0,2]); ax.set_facecolor('white')
st=df.groupby('StoreType')['Sales'].mean().sort_values(ascending=True)
ax.barh(st.index,st.values,color=[BLUE,GREEN,AMBER,RED],alpha=0.85,height=0.6)
ax.set_xlabel('Avg Daily Sales (€)')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,p:f'€{v:,.0f}'))
ax.set_title('Avg Sales by Store Type')
for i,(t,val) in enumerate(st.items()):
    ax.text(val+50,i,f'€{val:,.0f}',va='center',fontsize=9)

ax=fig.add_subplot(gs[1,0]); ax.set_facecolor('white')
dow=df.groupby('DayOfWeek')['Sales'].mean()
days=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
colors=[GREEN if i<5 else AMBER for i in range(7)]
ax.bar(days,dow.values,color=colors,alpha=0.85)
ax.set_ylabel('Avg Daily Sales (€)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,p:f'€{v:,.0f}'))
ax.set_title('Sales by Day of Week')
ax.grid(axis='y',alpha=0.3)

ax=fig.add_subplot(gs[1,1]); ax.set_facecolor('white')
promo=df.groupby('Promo')['Sales'].mean()
ax.bar(['No Promo','With Promo'],promo.values,color=[RED,GREEN],alpha=0.85,width=0.5)
ax.set_ylabel('Avg Daily Sales (€)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,p:f'€{v:,.0f}'))
ax.set_title('Promotion Impact on Sales')
for i,v in enumerate(promo.values):
    ax.text(i,v+100,f'€{v:,.0f}',ha='center',fontsize=10,fontweight='bold')
uplift=(promo[1]-promo[0])/promo[0]*100
ax.text(0.5,0.05,f'Uplift: +{uplift:.1f}%',transform=ax.transAxes,
        ha='center',color=GREEN,fontweight='bold',fontsize=11)
ax.grid(axis='y',alpha=0.3)

ax=fig.add_subplot(gs[1,2]); ax.set_facecolor('white')
ax.hist(df[df['Sales']<20000]['Sales'],bins=80,color=BLUE,alpha=0.7,edgecolor='white',lw=0.3)
ax.axvline(df['Sales'].mean(),color=RED,lw=2,linestyle='--',label=f"Mean: €{df['Sales'].mean():,.0f}")
ax.axvline(df['Sales'].median(),color=GREEN,lw=2,linestyle='--',label=f"Median: €{df['Sales'].median():,.0f}")
ax.set_xlabel('Daily Sales (€)'); ax.set_ylabel('Frequency')
ax.set_title('Sales Distribution (right-skewed)')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.savefig('charts/E1_sales_overview.png',dpi=150,bbox_inches='tight')
plt.close()
print("    ✓ E1_sales_overview.png")

#  E2: Store Type & Assortment 
print("  Chart E2: Store type analysis...")
fig,axes=plt.subplots(2,2,figsize=(16,10))
fig.suptitle('Store Type & Assortment Analysis  |  Sales, Customers & ATV',
             fontsize=14,fontweight='bold',color=NAVY)
fig.patch.set_facecolor('#F8FAFC')

ax=axes[0,0]; ax.set_facecolor('white')
pivot=df.groupby(['StoreType','Assortment'])['Sales'].mean().unstack(fill_value=0)
pivot.plot(kind='bar',ax=ax,color=PAL[:3],alpha=0.85,width=0.7)
ax.set_xlabel('Store Type'); ax.set_ylabel('Avg Daily Sales (€)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,p:f'€{v:,.0f}'))
ax.set_title('Avg Sales: Store Type x Assortment')
ax.legend(title='Assortment',fontsize=9); ax.grid(axis='y',alpha=0.3)
ax.set_xticklabels(ax.get_xticklabels(),rotation=0)

ax=axes[0,1]; ax.set_facecolor('white')
cust=df.groupby('StoreType')['Customers'].mean().sort_values(ascending=True)
ax.barh(cust.index,cust.values,color=PAL,alpha=0.85,height=0.6)
ax.set_xlabel('Avg Daily Customers')
ax.set_title('Customer Count by Store Type')
for i,(t,v) in enumerate(cust.items()):
    ax.text(v+5,i,f'{v:.0f}',va='center',fontsize=9)
ax.grid(axis='x',alpha=0.3)

ax=axes[1,0]; ax.set_facecolor('white')
atv=df.groupby(['StoreType','Assortment'])['ATV'].mean().unstack(fill_value=0)
atv.plot(kind='bar',ax=ax,color=PAL[:3],alpha=0.85,width=0.7)
ax.set_xlabel('Store Type'); ax.set_ylabel('Avg Transaction Value (€)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,p:f'€{v:.0f}'))
ax.set_title('Avg Transaction Value (ATV)\n(Type D + Assortment C = premium spend)')
ax.legend(title='Assortment',fontsize=9); ax.grid(axis='y',alpha=0.3)
ax.set_xticklabels(ax.get_xticklabels(),rotation=0)

ax=axes[1,1]; ax.set_facecolor('white')
st_summary=df.groupby('StoreType').agg(
    StoreCount=('Store','nunique'),
    TotalSales=('Sales','sum'),
).reset_index()
st_summary['RevenueShare']=st_summary['TotalSales']/st_summary['TotalSales'].sum()*100
x=range(len(st_summary)); w=0.35
ax.bar([i-w/2 for i in x],st_summary['StoreCount'],w,color=BLUE,alpha=0.85,label='Store Count')
ax2b=ax.twinx()
ax2b.bar([i+w/2 for i in x],st_summary['RevenueShare'],w,color=GREEN,alpha=0.85,label='Revenue %')
ax.set_xticks(x); ax.set_xticklabels(st_summary['StoreType'])
ax.set_ylabel('Store Count',color=BLUE); ax2b.set_ylabel('Revenue Share (%)',color=GREEN)
ax.set_title('Store Count vs Revenue Share')
ax2b.spines['top'].set_visible(False)
lines1,lbl1=ax.get_legend_handles_labels(); lines2,lbl2=ax2b.get_legend_handles_labels()
ax.legend(lines1+lines2,lbl1+lbl2,fontsize=9)

plt.tight_layout()
plt.savefig('charts/E2_store_type_analysis.png',dpi=150,bbox_inches='tight')
plt.close()
print("    ✓ E2_store_type_analysis.png")

#  E3: Seasonality 
print("  Chart E3: Seasonality...")
fig,axes=plt.subplots(2,2,figsize=(16,9))
fig.suptitle('Temporal & Seasonal Sales Patterns  |  When Do Sales Peak?',
             fontsize=14,fontweight='bold',color=NAVY)
fig.patch.set_facecolor('#F8FAFC')

ax=axes[0,0]; ax.set_facecolor('white')
for yr,color in zip([2013,2014,2015],[BLUE,GREEN,RED]):
    m=df[df['Year']==yr].groupby('Month')['Sales'].mean()
    ax.plot(m.index,m.values,color=color,lw=2.2,marker='o',ms=5,label=str(yr))
ax.set_xticks(range(1,13))
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
ax.set_ylabel('Avg Daily Sales (€)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,p:f'€{v:,.0f}'))
ax.set_title('Monthly Sales Pattern by Year\n(December peak, July dip)')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax=axes[0,1]; ax.set_facecolor('white')
heat=df.groupby(['Month','DayOfWeek'])['Sales'].mean().unstack()
heat.columns=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
sns.heatmap(heat,ax=ax,cmap='YlOrRd',fmt='.0f',annot=True,
            linewidths=0.5,cbar_kws={'label':'Avg Sales (€)'})
ax.set_xlabel('Day of Week'); ax.set_ylabel('Month')
ax.set_title('Sales Heatmap: Month x Day')

ax=axes[1,0]; ax.set_facecolor('white')
weekly=df.groupby('WeekOfYear')['Sales'].mean()
ax.fill_between(weekly.index,weekly.values,alpha=0.15,color=PURPLE)
ax.plot(weekly.index,weekly.values,color=PURPLE,lw=2)
ax.set_xlabel('Week of Year'); ax.set_ylabel('Avg Daily Sales (€)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,p:f'€{v:,.0f}'))
ax.set_title('Weekly Seasonality\n(Week 48-50 = pre-Christmas peak)')
ax.grid(alpha=0.3)

ax=axes[1,1]; ax.set_facecolor('white')
yearly=df.groupby('Year')['Sales'].sum().reset_index()
yearly['SalesM']=yearly['Sales']/1e6
bars=ax.bar(yearly['Year'].astype(str),yearly['SalesM'],
            color=[BLUE,GREEN,RED],alpha=0.85,width=0.5)
ax.set_ylabel('Total Sales (€M)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,p:f'€{v:.0f}M'))
ax.set_title('Annual Total Sales\n(2014 drop = store closures)')
ax.grid(axis='y',alpha=0.3)
for bar,val in zip(bars,yearly['SalesM']):
    ax.text(bar.get_x()+bar.get_width()/2,val+0.5,f'€{val:.0f}M',
            ha='center',fontsize=10,fontweight='bold')

plt.tight_layout()
plt.savefig('charts/E3_seasonality.png',dpi=150,bbox_inches='tight')
plt.close()
print("    ✓ E3_seasonality.png")

#  E4: Promotion Analysis 
print("  Chart E4: Promotion analysis...")
fig,axes=plt.subplots(1,3,figsize=(18,5))
fig.suptitle('Promotion Impact Analysis  |  Promo vs Promo2 vs Combined',
             fontsize=14,fontweight='bold',color=NAVY)
fig.patch.set_facecolor('#F8FAFC')

ax=axes[0]; ax.set_facecolor('white')
promo_st=df.groupby(['StoreType','Promo'])['Sales'].mean().unstack()
promo_st.columns=['No Promo','Promo']
promo_st.plot(kind='bar',ax=ax,color=[RED,GREEN],alpha=0.85,width=0.6)
ax.set_xlabel('Store Type'); ax.set_ylabel('Avg Daily Sales (€)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,p:f'€{v:,.0f}'))
ax.set_title('Promo Impact by Store Type')
ax.set_xticklabels(ax.get_xticklabels(),rotation=0)
ax.legend(fontsize=9); ax.grid(axis='y',alpha=0.3)

ax=axes[1]; ax.set_facecolor('white')
promo_m=df.groupby(['Month','Promo'])['Sales'].mean().unstack()
promo_m.columns=['No Promo','Promo']
ax.fill_between(promo_m.index,promo_m['No Promo'],alpha=0.15,color=RED)
ax.fill_between(promo_m.index,promo_m['Promo'],alpha=0.15,color=GREEN)
ax.plot(promo_m.index,promo_m['No Promo'],color=RED,lw=2,label='No Promo')
ax.plot(promo_m.index,promo_m['Promo'],color=GREEN,lw=2,label='Promo')
ax.set_xticks(range(1,13))
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
ax.set_ylabel('Avg Daily Sales (€)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,p:f'€{v:,.0f}'))
ax.set_title('Monthly Promo vs No-Promo Sales')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax=axes[2]; ax.set_facecolor('white')
cats=df.groupby(['Promo','Promo2'])['Sales'].mean().reset_index()
cats['Label']=cats.apply(lambda r:
    'Neither' if r['Promo']==0 and r['Promo2']==0 else
    'Promo2 only' if r['Promo']==0 and r['Promo2']==1 else
    'Promo only' if r['Promo']==1 and r['Promo2']==0 else 'Both', axis=1)
ax.bar(cats['Label'],cats['Sales'],color=['#6B7280',AMBER,GREEN,BLUE],alpha=0.85,width=0.6)
ax.set_ylabel('Avg Daily Sales (€)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,p:f'€{v:,.0f}'))
ax.set_title('Combined Promo Effect\n(Promo alone > Promo2 alone)')
ax.tick_params(axis='x',rotation=15); ax.grid(axis='y',alpha=0.3)
for i,row in cats.iterrows():
    ax.text(i,row['Sales']+50,f'€{row["Sales"]:,.0f}',ha='center',fontsize=9,fontweight='bold')

plt.tight_layout()
plt.savefig('charts/E4_promotion_analysis.png',dpi=150,bbox_inches='tight')
plt.close()
print("    ✓ E4_promotion_analysis.png")

#  E5: Competition Analysis 
print("  Chart E5: Competition analysis...")
fig,axes=plt.subplots(1,3,figsize=(18,5))
fig.suptitle('Competition Impact on Sales  |  Distance & Duration Effects',
             fontsize=14,fontweight='bold',color=NAVY)
fig.patch.set_facecolor('#F8FAFC')

ax=axes[0]; ax.set_facecolor('white')
df['CompDistBand']=pd.cut(df['CompetitionDistance'],
    bins=[0,500,1000,2000,5000,10000,100000],
    labels=['<500m','500m-1km','1-2km','2-5km','5-10km','>10km'])
cd=df.groupby('CompDistBand',observed=True)['Sales'].mean()
ax.bar(cd.index,cd.values,color=PAL[:len(cd)],alpha=0.85)
ax.set_ylabel('Avg Daily Sales (€)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,p:f'€{v:,.0f}'))
ax.set_title('Sales by Competition Distance')
ax.tick_params(axis='x',rotation=30); ax.grid(axis='y',alpha=0.3)

ax=axes[1]; ax.set_facecolor('white')
sample=df[df['CompetitionDistance']<20000].sample(50000,random_state=42)
ax.scatter(sample['CompetitionDistance'],sample['Sales'],alpha=0.05,s=5,color=BLUE)
coeffs=np.polyfit(sample['CompetitionDistance'].fillna(0),sample['Sales'],1)
xline=np.linspace(0,20000,100)
ax.plot(xline,np.polyval(coeffs,xline),color=RED,lw=2,label='Trend')
ax.set_xlabel('Competition Distance (m)'); ax.set_ylabel('Sales (€)')
ax.set_title('Competition Distance vs Sales\n(Further = slightly higher — isolated store premium)')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax=axes[2]; ax.set_facecolor('white')
df['CompOpenBand']=pd.cut(df['CompetitionOpen'].clip(0,60),
    bins=[-1,0,6,12,24,36,60],
    labels=['No comp','<6mo','6-12mo','1-2yr','2-3yr','>3yr'])
co=df.groupby('CompOpenBand',observed=True)['Sales'].mean()
ax.bar(co.index,co.values,color=PAL[:len(co)],alpha=0.85)
ax.set_ylabel('Avg Daily Sales (€)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,p:f'€{v:,.0f}'))
ax.set_title('Sales by Competitor Duration\n(Drop when competitor opens, stabilises over time)')
ax.tick_params(axis='x',rotation=30); ax.grid(axis='y',alpha=0.3)

plt.tight_layout()
plt.savefig('charts/E5_competition_analysis.png',dpi=150,bbox_inches='tight')
plt.close()
print("    ✓ E5_competition_analysis.png")

#  E6: Correlation Matrix 
print("  Chart E6: Correlation matrix...")
fig,ax=plt.subplots(figsize=(14,10))
fig.patch.set_facecolor('#F8FAFC'); ax.set_facecolor('white')
fig.suptitle('Feature Correlation Matrix  |  What Drives Daily Sales?',
             fontsize=14,fontweight='bold',color=NAVY)
num_cols=['Sales','Customers','Promo','Promo2','DayOfWeek','Month',
          'CompetitionDistance','CompetitionOpen','Promo2Open',
          'StoreType_enc','Assortment_enc','Year','IsWeekend',
          'IsMonthStart','IsMonthEnd','SchoolHoliday','StateHoliday']
corr=df[num_cols].corr()
mask=np.triu(np.ones_like(corr,dtype=bool))
from matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('rg',[RED,'white',GREEN])
sns.heatmap(corr,ax=ax,mask=mask,cmap=cmap,vmin=-1,vmax=1,
            annot=True,fmt='.2f',linewidths=0.5,square=True,
            cbar_kws={'shrink':0.8,'label':'Pearson r'})
ax.set_title('Sales most correlated with Customers (0.82) and Promo (0.45)')
plt.tight_layout()
plt.savefig('charts/E6_correlation_matrix.png',dpi=150,bbox_inches='tight')
plt.close()
print("    ✓ E6_correlation_matrix.png")

print("\nSCRIPT 2 COMPLETE — 6 EDA charts saved to charts/")
