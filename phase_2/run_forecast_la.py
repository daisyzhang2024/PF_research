"""
run_forecast_la.py
===================
Uses the best combo from backtesting -- yoy_logdiff | RandomForest, per
la_delevel_results.csv (re-check that file if you rerun run_backtest_la.py
on updated data, in case the winner changes) -- to predict LA County
Sales Tax Revenue for one additional month, WITHOUT re-running the full
backtest. Fast to import and call repeatedly (e.g. from a notebook).

IMPORTANT -- read before using:
  This model predicts month t's revenue using month t's own macro inputs
  (unemployment, gas price, metro ridership, etc.), so it can only
  forecast a month for which you already know those macro values
  ("nowcasting" -- e.g. predicting last month's or this month's revenue
  before the tax data itself is published, using macro data that's
  already out).

  It CANNOT forecast several months ahead unless you also supply
  forecasted macro inputs for those future months, chaining forward
  month by month. Naively holding macro inputs flat is not recommended
  -- errors compound silently.

  Because the winning combo here is year-over-year (not month-over-month
  like CHMR), the model anchors its prediction to the actual value from
  12 months prior, so your data file needs to already contain that
  same-month-last-year row.

Run:  python3 run_forecast_la.py
"""

from la_model_lib import prepare_full_dataset, forecast_next_month

if __name__ == "__main__":
    df = prepare_full_dataset()

    # ---- Replace with the actual known values for the month you're
    #      forecasting once available ----
    next_month_features = {
        "month": 5,                      # May
        "unemp": 5.5,
        "GasPrice": 5.90,
        "metro_ridership": 6100000,
        "CPI_U": 356.0,
        "la_search_sentiment": 55,
        "la_news_sentiment": 0.3,
    }

    predicted_revenue = forecast_next_month(df, next_month_features)
    print(f"Predicted LA County Sales Tax Revenue for month={next_month_features['month']}: ${predicted_revenue:,.0f}")


# Predicted LA County Sales Tax Revenue for month=5: $47,221,219