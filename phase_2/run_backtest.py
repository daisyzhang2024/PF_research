"""
run_backtest.py
================
Evaluates all (target-formulation x model) combinations via rolling-
origin backtest and saves the comparison tables. This is the "does the
de-leveling approach work, and which combo is best" script.

Reframing implemented (see chmr_model_lib.py for the functions):
  1. Model de-leveled targets instead of raw CHMR level:
       trend_dev   : log(CHMR) minus a fitted log-linear growth trend
       mom_logdiff : month-over-month log-change
       yoy_logdiff : year-over-year log-change
  2. Add calendar month dummies (Feb..Dec, Jan baseline) to every model.
  3. Reconstruct absolute-dollar CHMR from each de-leveled prediction and
     evaluate ONLY on reconstructed levels (R^2, MAPE, RMSE), against
     benchmarks R^2 > 0.93, MAPE < 3.5%, OOS RMSE < $8M.
  4. Rolling-origin (walk-forward) backtest over the last N_TEST months
     for Ridge, RandomForest, and GradientBoosting.

Run:  python3 run_backtest.py
Outputs (written next to this script):
  chmr_delevel_results.csv           - metrics for all 9 combos
  chmr_month_dummy_effects.csv       - seasonal coefficients vs. January
  chmr_best_backtest_predictions.csv - actual vs. predicted for the winner
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from chmr_model_lib import (
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

    results_df.to_csv("chmr_delevel_results.csv")

    # ---- month-dummy coefficients from a full-sample OLS on trend_dev ----
    feats = feature_cols_for("trend_dev", df.columns)
    fit_df = df.dropna(subset=feats + ["trend_dev"])
    X = sm.add_constant(fit_df[feats].astype(float))
    ols = sm.OLS(fit_df["trend_dev"].astype(float), X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    month_coefs = ols.params[[c for c in feats if c.startswith("m_")]]
    month_pvals = ols.pvalues[[c for c in feats if c.startswith("m_")]]
    month_table = pd.DataFrame({"coef_log_dev": month_coefs, "p_value": month_pvals})
    month_table["approx_pct_effect_vs_Jan"] = (np.exp(month_table["coef_log_dev"]) - 1) * 100
    month_table.to_csv("chmr_month_dummy_effects.csv")
    print("\n=== Month effects (relative to January baseline), from HAC-robust OLS on trend deviation ===\n")
    print(month_table.round(4).to_string())

    best_key = results_df.index[0]
    detail[best_key].to_csv("chmr_best_backtest_predictions.csv", index=False)
    print(f"\nBest performing combo: {best_key}")
