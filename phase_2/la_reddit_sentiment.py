import pandas as pd
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

def fetch_arctic_shift_data():
    url = "https://arctic-shift.com/api/posts/search"
    
    params = {
        "q": "shopping OR store OR retail",
        "subreddit": "LosAngeles",
        "after": "2018-04-01",
        "before": "2026-04-01",
        "limit": 1000
    }

    print("Fetching Reddit submissions from Arctic Shift API...")
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        posts = response.json().get('data', [])
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return

    if not posts:
        print("⚠️ No posts returned.")
        return

    sentiment_data = []
    for post in posts:
        title = post.get('title', '')
        created_utc = post.get('created_utc')
        if title and created_utc:
            score = sia.polarity_scores(title)['compound']
            sentiment_data.append({'created_utc': created_utc, 'sentiment_score': score})

    df_sentiment = pd.DataFrame(sentiment_data)
    df_sentiment['date'] = pd.to_datetime(df_sentiment['created_utc'], unit='s')

    monthly_sentiment = df_sentiment.resample('MS', on='date')['sentiment_score'].mean().reset_index()
    monthly_sentiment.rename(columns={'sentiment_score': 'la_consumer_sentiment'}, inplace=True)

    monthly_sentiment.to_csv("la_reddit_sentiment.csv", index=False)
    print("✅ Saved la_reddit_sentiment.csv successfully!\n")
    print(monthly_sentiment.head())

if __name__ == "__main__":
    fetch_arctic_shift_data()