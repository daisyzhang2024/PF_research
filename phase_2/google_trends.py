import pandas as pd
from trendspy import Trends

tr = Trends()

# Query Google Trends
df = tr.interest_over_time(
    keywords=["shopping chicago"],
    timeframe="2015-01-01 2021-12-01",
    geo="US-IL-602"
)

# 1. Ensure index is DatetimeIndex
df.index = pd.to_datetime(df.index)

# 2. Resample monthly on the index directly (no column key needed)
monthly_trends = df.resample('MS')['shopping chicago'].mean().reset_index()

# 3. Rename columns cleanly
monthly_trends.columns = ['date', 'chicago_search_sentiment']

# 4. Save
monthly_trends.to_csv("chicago_sentiment.csv", index=False)

print("Saved chicago_sentiment.csv successfully!")
print(monthly_trends.head())