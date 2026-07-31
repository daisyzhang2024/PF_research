"""
run_backtest_la.py
===================
Evaluates all (target-formulation x model) combinations for LA County
"Sales Tax Revenue" via rolling-origin backtest and saves the comparison
tables. Same de-leveling approach as the CHMR (Cook County) version:

  1. Model de-leveled targets instead of raw dollar level:
       trend_dev   : log(target) minus a fitted log-linear growth trend
       mom_logdiff : month-over-month log-change
       yoy_logdiff : year-over-year log-change
  2. Add calendar month dummies (Feb..Dec, Jan baseline) to every model.
  3. Reconstruct absolute-dollar revenue from each de-leveled prediction
     and evaluate ONLY on reconstructed levels (R^2, MAPE, RMSE).
  4. Rolling-origin (walk-forward) backtest over the last N_TEST months
     for Ridge, RandomForest, and GradientBoosting.

Note: this dataset only runs 2018-04 to 2026-04 (97 months, vs. CHMR's
136), so the yoy_logdiff / trend_dev combos have less history to learn
from -- worth keeping in mind when comparing MAPE/R2 against the CHMR
results.

Run:  python3 run_backtest_la.py
Outputs (written next to this script):
  la_delevel_results.csv           - metrics for all 9 combos
  la_month_dummy_effects.csv       - seasonal coefficients vs. January
  la_best_backtest_predictions.csv - actual vs. predicted for the winner
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from la_model_lib import (
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

    results_df.to_csv("la_delevel_results.csv")

    # ---- month-dummy coefficients from a full-sample OLS on trend_dev ----
    feats = feature_cols_for("trend_dev", df.columns)
    fit_df = df.dropna(subset=feats + ["trend_dev"])
    X = sm.add_constant(fit_df[feats].astype(float))
    ols = sm.OLS(fit_df["trend_dev"].astype(float), X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    month_coefs = ols.params[[c for c in feats if c.startswith("m_")]]
    month_pvals = ols.pvalues[[c for c in feats if c.startswith("m_")]]
    month_table = pd.DataFrame({"coef_log_dev": month_coefs, "p_value": month_pvals})
    month_table["approx_pct_effect_vs_Jan"] = (np.exp(month_table["coef_log_dev"]) - 1) * 100
    month_table.to_csv("la_month_dummy_effects.csv")
    print("\n=== Month effects (relative to January baseline), from HAC-robust OLS on trend deviation ===\n")
    print(month_table.round(4).to_string())

    best_key = results_df.index[0]
    detail[best_key].to_csv("la_best_backtest_predictions.csv", index=False)
    print(f"\nBest performing combo: {best_key}")


#                                    R2  MAPE_%          RMSE
# yoy_logdiff | RandomForest      0.348   7.941  4.685268e+06
# yoy_logdiff | Ridge             0.110   9.332  5.472604e+06
# yoy_logdiff | GradientBoosting -0.059  10.230  5.969484e+06
# trend_dev | Ridge               0.011  11.626  5.768829e+06
# trend_dev | GradientBoosting   -0.030  12.084  5.887206e+06
# trend_dev | RandomForest       -0.075  12.307  6.013943e+06
# mom_logdiff | Ridge            -0.929  14.630  8.057369e+06
# mom_logdiff | RandomForest     -1.115  15.903  8.435268e+06
# mom_logdiff | GradientBoosting -2.064  18.320  1.015333e+07

# --- vs. benchmark thresholds (R2>0.93, MAPE<3.5%, OOS RMSE<$8M) ---
# yoy_logdiff | RandomForest          fail
# yoy_logdiff | Ridge                 fail
# yoy_logdiff | GradientBoosting      fail
# trend_dev | Ridge                   fail
# trend_dev | GradientBoosting        fail
# trend_dev | RandomForest            fail
# mom_logdiff | Ridge                 fail
# mom_logdiff | RandomForest          fail
# mom_logdiff | GradientBoosting      fail

# === Month effects (relative to January baseline), from HAC-robust OLS on trend deviation ===

#       coef_log_dev  p_value  approx_pct_effect_vs_Jan
# m_2        -0.1134   0.0233                  -10.7199
# m_3        -0.0308   0.6427                   -3.0290
# m_4        -0.0425   0.2988                   -4.1589
# m_5         0.1425   0.0138                   15.3209
# m_6        -0.0890   0.1143                   -8.5164
# m_7         0.0729   0.0895                    7.5673
# m_8         0.0801   0.0725                    8.3397
# m_9        -0.0525   0.2623                   -5.1127
# m_10        0.0142   0.7260                    1.4300
# m_11        0.0107   0.8302                    1.0788
# m_12        0.2209   0.0076                   24.7175

# Best performing combo: yoy_logdiff | RandomForest