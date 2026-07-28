import pandas as pd
from trendspy import Trends

def fetch_la_sentiment():
    tr = Trends()

    # Query Google Trends for LA Metro DMA (US-CA-803)
    # Using keywords relevant to LA shopping search activity
    df = tr.interest_over_time(
        keywords=["shopping los angeles"],
        timeframe="2018-04-01 2026-04-01",
        geo="US-CA-803"
    )

    # 1. Ensure index is a DatetimeIndex
    df.index = pd.to_datetime(df.index)

    # 2. Resample monthly ('MS' = Month Start) and calculate mean search interest
    monthly_trends = df.resample('MS')['shopping los angeles'].mean().reset_index()

    # 3. Cleanly rename columns
    monthly_trends.columns = ['date', 'la_search_sentiment']

    # 4. Export to CSV
    output_filename = "la_sentiment.csv"
    monthly_trends.to_csv(output_filename, index=False)

    print(f"Saved {output_filename} successfully!\n")
    print(monthly_trends.head())

if __name__ == "__main__":
    fetch_la_sentiment()