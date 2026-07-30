"""
run_forecast.py
================
Uses the best combo from backtesting (mom_logdiff | RandomForest) to
predict CHMR for one additional month, WITHOUT re-running the full
backtest. Fast to import and call repeatedly (e.g. from a notebook).

IMPORTANT -- read before using:
  This model predicts month t's CHMR using month t's own macro inputs
  (unemployment, gas price, CTA ridership, etc.), so it can only forecast
  a month for which you already know those macro values ("nowcasting" --
  e.g. predicting last month's or this month's CHMR before the tax data
  itself is published, using macro data that's already out).

  It CANNOT forecast several months ahead unless you also supply
  forecasted macro inputs for those future months (e.g. from BLS/EIA
  projections), chaining forward month by month. Naively holding macro
  inputs flat is not recommended -- errors compound silently.

Run:  python3 run_forecast.py
"""

from chmr_model_lib import prepare_full_dataset, forecast_next_month

if __name__ == "__main__":
    df = prepare_full_dataset()

    # ---- Replace with the actual known values for the month you're
    #      forecasting once available ----
    next_month_features = {
        "month": 5,                      # May
        "unemp": 5.2,
        "CPI_U": 310.0,
        "GasPrice": 4.10,
        "CDD": 20.0,
        "CTA": 29500000.0,
        "osm_shop_count": 8180.0,
        "osm_net_new": 90.0,
        "chicago_search_sentiment": 70.0,
    }

    predicted_CHMR = forecast_next_month(df, next_month_features)
    print(f"Predicted CHMR for month={next_month_features['month']}: ${predicted_CHMR:,.0f}")
