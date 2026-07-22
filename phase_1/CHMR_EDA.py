import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# --- Load ---
tax_data = pd.read_csv('IDOR_CookCounty.csv')

# --- Filter for CHMR ---
chmr_data = tax_data[tax_data['Tax'] == 'CHMR'].copy()

month_cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_num = {m: i + 1 for i, m in enumerate(month_cols)}

# --- Clean currency strings (e.g. "24,966,333.31" -> 24966333.31) in month cols ---
for col in month_cols:
    chmr_data[col] = (
        chmr_data[col]
        .astype(str)
        .str.replace(',', '', regex=False)
        .str.replace('"', '', regex=False)
        .replace({'nan': np.nan, '': np.nan})
        .astype(float)
    )

# --- Melt wide -> long: one row per Year/Month/Value ---
id_cols = ['Year', 'Local Government', 'Tax', 'Vendor #']
long_data = chmr_data.melt(
    id_vars=id_cols,
    value_vars=month_cols,
    var_name='Month',
    value_name='Value'
)

long_data['MonthNum'] = long_data['Month'].map(month_num)
long_data['Date'] = pd.to_datetime(
    long_data['Year'].astype(str) + '-' + long_data['MonthNum'].astype(str) + '-01',
    errors='coerce'
)
long_data = long_data.dropna(subset=['Value']).sort_values('Date').reset_index(drop=True)

# If there are multiple vendors/local governments under CHMR, aggregate to one series
# (drop the groupby if you want to keep them split)
monthly_series = long_data.groupby('Date', as_index=False)['Value'].sum()

# --- Data exploration ---
print(monthly_series.describe())
print(f"\nRecord count: {len(monthly_series)}")
print(f"Date range: {monthly_series['Date'].min().date()} to {monthly_series['Date'].max().date()}")

# --- Gap check: clean monthly series back to 2015? ---
start = max(pd.Timestamp('2015-01-01'), monthly_series['Date'].min())
full_range = pd.date_range(start=start, end=monthly_series['Date'].max(), freq='MS')
missing_months = full_range.difference(monthly_series['Date'])
print(f"\nMissing months from {start.date()} to {full_range.max().date()}: {len(missing_months)}")
if len(missing_months) > 0:
    print(missing_months.tolist())

# --- Duplicate check (restatements) ---
dupes = monthly_series[monthly_series.duplicated(subset=['Date'], keep=False)]
if len(dupes) > 0:
    print(f"\nWarning: {len(dupes)} duplicate month entries found")
    print(dupes)

# --- Cross-check monthly sums against reported FY Total, if present ---
if 'FY Total' in chmr_data.columns:
    chmr_data['FY Total_clean'] = (
        chmr_data['FY Total'].astype(str)
        .str.replace(',', '', regex=False)
        .str.replace('"', '', regex=False)
        .replace({'nan': np.nan, '': np.nan})
        .astype(float)
    )
    check = chmr_data.groupby('Year').apply(
        lambda r: r[month_cols].sum(axis=1).sum()
    ).reset_index(name='SumOfMonths')
    reported = chmr_data.groupby('Year')['FY Total_clean'].sum().reset_index()
    reconcile = check.merge(reported, on='Year')
    reconcile['Diff'] = reconcile['SumOfMonths'] - reconcile['FY Total_clean']
    print("\nFY Total reconciliation (should be ~0 diff):")
    print(reconcile)

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(monthly_series['Date'], monthly_series['Value'], marker='o', linewidth=1)
ax.set_title('Cook County CHMR Tax Collections (Monthly)')
ax.set_xlabel('Date')
ax.set_ylabel('Value ($)')
ax.grid(True, alpha=0.3)
fig.tight_layout()

# --- Save with version control ---
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
fig.savefig(f'chmr_plot_{timestamp}.png', dpi=150)
monthly_series.to_csv(f'chmr_monthly_clean_{timestamp}.csv', index=False)
long_data.to_csv(f'chmr_long_raw_{timestamp}.csv', index=False)