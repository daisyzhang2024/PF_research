import pandas as pd

df = pd.read_csv('CPI-W.csv', skiprows=11)

month_cols = ['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec']
month_num = {m: i for i, m in enumerate(month_cols, start=1)}

# Wide -> long: one row per Year+Month
long_df = df.melt(id_vars='Year', value_vars=month_cols,
                   var_name='Month', value_name='CPI')

# Build the date column (2015-01-01, 2015-02-01, ...)
long_df['Date'] = pd.to_datetime(
    long_df['Year'].astype(str) + '-' + long_df['Month'].map(month_num).astype(str) + '-01'
)

long_df = long_df.sort_values('Date').set_index('Date')[['CPI']]

long_df.to_csv('cpiW_by_date.csv')
print(long_df)