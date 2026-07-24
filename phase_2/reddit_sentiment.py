import pandas as pd
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Example: Pulling posts mentioning Chicago retail keywords
# For historical data back to 2015, PullPush / Pushshift API endpoints work well:
url = "https://api.pullpush.io/reddit/search/submission/"

params = {
    "q": "chicago (store OR shop OR retail OR shopping)",
    "subreddit": "chicago",
    "after": "2015-01-01",
    "before": "2021-12-01",
    "size": 1000
}

# Fetch posts and calculate compound sentiment score (-1 to +1)
response = requests.get(url, params=params).json()
posts = response.get('data', [])

sentiment_data = []
for post in posts:
    title = post.get('title', '')
    created_utc = post.get('created_utc')
    score = sia.polarity_scores(title)['compound']
    sentiment_data.append({'created_utc': created_utc, 'sentiment_score': score})

df_sentiment = pd.DataFrame(sentiment_data)
df_sentiment['date'] = pd.to_datetime(df_sentiment['created_utc'], unit='s')

# Resample to Monthly Average
monthly_sentiment = df_sentiment.resample('MS', on='date')['sentiment_score'].mean().reset_index()
monthly_sentiment.rename(columns={'sentiment_score': 'chicago_consumer_sentiment'}, inplace=True)