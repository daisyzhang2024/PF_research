from trendspy import Trends
import time

tr = Trends(request_delay=5.0)  # 2.0 often isn't enough anymore

def fetch_with_backoff(tr, keywords, timeframe, geo, max_retries=6):
    for attempt in range(max_retries):
        try:
            return tr.interest_over_time(keywords=keywords, timeframe=timeframe, geo=geo)
        except Exception as e:
            if "429" in str(e):
                wait = 30 * (2 ** attempt)  # 30, 60, 120, 240...
                print(f"429 hit, backing off {wait}s (attempt {attempt+1})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Exceeded retries — Google is still blocking this IP")

df = fetch_with_backoff(tr, ["shopping chicago"], "2022-01-01 2026-04-01", "US-IL-602")