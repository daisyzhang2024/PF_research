"""
run_forecast_nyc.py
====================
Uses the best combo from backtesting -- yoy_logdiff | RandomForest, per
nyc_delevel_results.csv (re-check that file if you rerun run_backtest_nyc.py
on updated data, in case the winner changes) -- to predict NYC sales_tax
revenue for one additional month, WITHOUT re-running the full backtest.
Fast to import and call repeatedly (e.g. from a notebook).

IMPORTANT -- read before using:
  This model predicts month t's revenue using month t's own macro inputs
  (unemployment, gas price, subway/bus ridership, search-interest), so
  it can only forecast a month for which you already know those macro
  values ("nowcasting" -- e.g. predicting last month's or this month's
  revenue before the tax data itself is published, using macro data
  that's already out).

  It CANNOT forecast several months ahead unless you also supply
  forecasted macro inputs for those future months, chaining forward
  month by month. Naively holding macro inputs flat is not recommended
  -- errors compound silently.

  Because the winning combo here is year-over-year (not month-over-month),
  the model anchors its prediction to the actual value from 12 months
  prior, so your data file needs to already contain that
  same-month-last-year row.

  Note: NYC's benchmark thresholds (R2>0.93, MAPE<3.5%, OOS RMSE<$8M)
  were carried over from the CHMR/LA scripts but the RMSE threshold in
  particular doesn't transfer -- NYC's monthly sales_tax revenue runs
  ~$600M-$1.1B vs. LA County's ~$35-70M, so an $8M RMSE bar is far too
  strict here. Judge NYC results primarily on MAPE_%/R2 instead, or
  rescale the RMSE threshold to NYC's revenue base if you want a hard
  pass/fail cutoff.

Run:  python3 run_forecast_nyc.py
"""

from nyc_model_lib import prepare_full_dataset, forecast_next_month

if __name__ == "__main__":
    df = prepare_full_dataset()

    # ---- Replace with the actual known values for the month you're
    #      forecasting once available ----
    next_month_features = {
        "month": 6,                      # June
        "unemp": 5.3,
        "GasPrice": 4.30,
        "ridership": 148000000,
        "nyc_search": 62,
    }

    predicted_revenue = forecast_next_month(df, next_month_features)
    print(f"Predicted NYC Sales Tax Revenue for month={next_month_features['month']}: ${predicted_revenue:,.0f}")

# Predicted NYC Sales Tax Revenue for month=6: $1,166,995,190