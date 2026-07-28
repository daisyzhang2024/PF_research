import time
import pandas as pd
import requests

def fetch_gdelt_with_retry():
    print("Querying GDELT API directly with retry logic...")
    
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    
    params = {
        "query": '"los angeles" shopping',
        "mode": "timelinetone",
        "startdatetime": "20180401000000",
        "enddatetime": "20260401000000",
        "format": "json"
    }
    
    # 1. Custom User-Agent prevents GDELT from blocking python-requests
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    max_retries = 4
    wait_time = 30  # Initial wait time in seconds

    data = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Sending request (Attempt {attempt}/{max_retries})...")
            response = requests.get(url, params=params, headers=headers, timeout=20)
            
            # Handle rate limit explicitly
            if response.status_code == 429:
                print(f"⚠️ Hit GDELT Rate Limit (429). Cooling down for {wait_time}s before retrying...")
                time.sleep(wait_time)
                wait_time *= 2  # Double the wait time on next attempt
                continue
                
            response.raise_for_status()
            data = response.json()
            break  # Request succeeded!
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")
            if attempt < max_retries:
                print(f"Waiting {wait_time}s...")
                time.sleep(wait_time)

    if not data or 'timeline' not in data or not data['timeline']:
        print("❌ Could not retrieve data. Try running again in 5–10 minutes.")
        return

    # 2. Extract and clean timeline data
    timeline = data['timeline'][0]['data']
    df = pd.DataFrame(timeline)

    df['date'] = pd.to_datetime(df['date'])
    df['la_news_sentiment'] = pd.to_numeric(df['value'], errors='coerce')

    # 3. Resample to Monthly Average ('MS' = Month Start)
    monthly_df = df.resample('MS', on='date')['la_news_sentiment'].mean().reset_index()

    # 4. Save
    output_filename = "la_news_sentiment.csv"
    monthly_df.to_csv(output_filename, index=False)

    print(f"\n✅ Saved {output_filename} successfully with {len(monthly_df)} monthly records!\n")
    print(monthly_df.head())
    print("\nRecent rows:")
    print(monthly_df.tail())

if __name__ == "__main__":
    fetch_gdelt_with_retry()