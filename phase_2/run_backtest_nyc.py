"""
run_backtest_nyc.py
====================
Evaluates all (target-formulation x model) combinations for NYC
"sales_tax" revenue via rolling-origin backtest and saves the comparison
tables. Same de-leveling approach as the CHMR (Cook County) / LA County
versions:

  1. Model de-leveled targets instead of raw dollar level:
       trend_dev   : log(target) minus a fitted log-linear growth trend
       mom_logdiff : month-over-month log-change
       yoy_logdiff : year-over-year log-change
  2. Add calendar month dummies (Feb..Dec, Jan baseline) to every model.
  3. Reconstruct absolute-dollar revenue from each de-leveled prediction
     and evaluate ONLY on reconstructed levels (R^2, MAPE, RMSE).
  4. Rolling-origin (walk-forward) backtest over the last N_TEST months
     for Ridge, RandomForest, and GradientBoosting.

Note: this dataset only runs 2019-01 to 2026-05 (89 months, vs. CHMR's
136 and LA's 97), and it has fewer macro predictors than LA (no CPI, no
news-sentiment series -- just unemp, GasPrice, ridership, nyc_search).
With N_TEST=24, the yoy_logdiff / trend_dev combos are trained on as
few as ~53 months in the earliest fold -- worth keeping in mind when
comparing MAPE/R2 against the CHMR/LA results.

Run:  python3 run_backtest_nyc.py
Outputs (written next to this script):
  nyc_delevel_results.csv           - metrics for all 9 combos
  nyc_month_dummy_effects.csv       - seasonal coefficients vs. January
  nyc_best_backtest_predictions.csv - actual vs. predicted for the winner
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from nyc_model_lib import (
    prepare_full_dataset, feature_cols_for, backtest, N_TEST,
)

if __name__ == "__main__":
    df = prepare_full_dataset()

    results = {}
    detail = {}
    for target_kind in ["trend_dev", "mom_logdiff", "yoy_logdiff"]:
        for model_name in ["Ridge", "RandomForest", "GradientBoosting"]:
            key = f"{target_kind} | {model_name}"
            try:
                metrics, det = backtest(df, target_kind, model_name)
                results[key] = metrics
                detail[key] = det
            except Exception as e:
                results[key] = {"error": str(e)}

    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values("MAPE_%")
    print("\n=== Rolling-origin backtest (last {} months), reconstructed levels ===\n".format(N_TEST))
    print(results_df.round(3).to_string())

    print("\n--- vs. benchmark thresholds (R2>0.93, MAPE<3.5%, OOS RMSE<$8M) ---")
    for key, row in results_df.iterrows():
        if "error" in row and not pd.isna(row.get("error", np.nan)):
            continue
        passed = (row["R2"] > 0.93) and (row["MAPE_%"] < 3.5) and (row["RMSE"] < 8_000_000)
        print(f"{key:35s} {'PASS' if passed else 'fail'}")

    results_df.to_csv("nyc_delevel_results.csv")

    # ---- month-dummy coefficients from a full-sample OLS on trend_dev ----
    feats = feature_cols_for("trend_dev", df.columns)
    fit_df = df.dropna(subset=feats + ["trend_dev"])
    X = sm.add_constant(fit_df[feats].astype(float))
    ols = sm.OLS(fit_df["trend_dev"].astype(float), X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    month_coefs = ols.params[[c for c in feats if c.startswith("m_")]]
    month_pvals = ols.pvalues[[c for c in feats if c.startswith("m_")]]
    month_table = pd.DataFrame({"coef_log_dev": month_coefs, "p_value": month_pvals})
    month_table["approx_pct_effect_vs_Jan"] = (np.exp(month_table["coef_log_dev"]) - 1) * 100
    month_table.to_csv("nyc_month_dummy_effects.csv")
    print("\n=== Month effects (relative to January baseline), from HAC-robust OLS on trend deviation ===\n")
    print(month_table.round(4).to_string())

    best_key = results_df.index[0]
    detail[best_key].to_csv("nyc_best_backtest_predictions.csv", index=False)
    print(f"\nBest performing combo: {best_key}")

# === Rolling-origin backtest (last 24 months), reconstructed levels ===

#                                    R2  MAPE_%          RMSE
# yoy_logdiff | RandomForest      0.765   3.264  4.350058e+07
# yoy_logdiff | Ridge             0.658   4.489  5.246215e+07
# yoy_logdiff | GradientBoosting  0.319   4.872  7.397342e+07
# mom_logdiff | Ridge             0.237   7.437  7.831297e+07
# mom_logdiff | GradientBoosting  0.042   7.469  8.776514e+07
# trend_dev | Ridge               0.065   8.545  8.670441e+07
# trend_dev | GradientBoosting   -0.019   8.820  9.053423e+07
# trend_dev | RandomForest       -0.140   9.501  9.575116e+07
# mom_logdiff | RandomForest     -0.897  11.646  1.234885e+08

# --- vs. benchmark thresholds (R2>0.93, MAPE<3.5%, OOS RMSE<$8M) ---
# yoy_logdiff | RandomForest          fail
# yoy_logdiff | Ridge                 fail
# yoy_logdiff | GradientBoosting      fail
# mom_logdiff | Ridge                 fail
# mom_logdiff | GradientBoosting      fail
# trend_dev | Ridge                   fail
# trend_dev | GradientBoosting        fail
# trend_dev | RandomForest            fail
# mom_logdiff | RandomForest          fail

# === Month effects (relative to January baseline), from HAC-robust OLS on trend deviation ===

#       coef_log_dev  p_value  approx_pct_effect_vs_Jan
# m_2        -0.1463   0.0021                  -13.6104
# m_3         0.0022   0.9800                    0.2167
# m_4        -0.1325   0.0178                  -12.4092
# m_5        -0.1637   0.0032                  -15.1031
# m_6         0.0400   0.5041                    4.0854
# m_7        -0.0926   0.2110                   -8.8481
# m_8        -0.1000   0.1566                   -9.5189
# m_9         0.0285   0.5012                    2.8885
# m_10       -0.0978   0.1108                   -9.3214
# m_11       -0.0755   0.1888                   -7.2737
# m_12        0.0950   0.3056                    9.9692