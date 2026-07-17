"""
CHMR Sales Tax Revenue Prediction — Best Model Auto-Select
============================================================
Tries a small set of candidate models (Ridge, Random Forest, Gradient
Boosting, XGBoost), picks whichever generalizes best on walk-forward
(time-series) cross-validation, and reports full diagnostics for that
one winning model only.

Input CSV schema (one row per month):
    date, unemp, CPI_U, GasPrice, CDD, CHMR

Usage:
    python chmr_best_model.py --csv combined_CC_v2.csv --test_months 12
    python chmr_best_model.py --csv combined_CC_v2.csv --test_months 12 --exclude_test_months 2021-01,2021-02
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. LOAD + CLEAN
# ---------------------------------------------------------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Parse percent strings like "5.70%" -> 5.70 (float). Checks is_numeric_dtype
    # rather than dtype == object, since pandas' newer StringDtype doesn't
    # compare equal to `object` but still needs the same parsing.
    if not pd.api.types.is_numeric_dtype(df["unemp"]):
        df["unemp"] = df["unemp"].astype(str).str.rstrip("%").astype(float)

    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y", errors="coerce")
    if df["date"].isna().any():
        df["date"] = pd.to_datetime(df["date"].astype(str), errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Autoregressive / lag features — all built with .shift(), so they only
    # look backward in time (no leakage from the row being predicted).
    df["CHMR_lag1"] = df["CHMR"].shift(1)
    df["CHMR_lag2"] = df["CHMR"].shift(2)
    df["CHMR_lag3"] = df["CHMR"].shift(3)
    df["CHMR_lag12"] = df["CHMR"].shift(12)   # same month, prior year
    df["CHMR_lag13"] = df["CHMR"].shift(13)   # same month, prior year, one month earlier
    df["CHMR_roll3_mean"] = df["CHMR"].shift(1).rolling(window=3).mean()
    df["CHMR_roll12_mean"] = df["CHMR"].shift(1).rolling(window=12).mean()

    # Growth-rate feature built entirely from already-known lagged values
    # (lag1 vs lag13) — NOT pct_change(12) on the raw column, which would use
    # the current row's CHMR and leak the target into its own feature.
    df["CHMR_prev_yoy_pct"] = (df["CHMR_lag1"] - df["CHMR_lag13"]) / df["CHMR_lag13"]

    df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLS = [
    "unemp", "CPI_U", "GasPrice", "CDD",
    "month", "quarter", "month_sin", "month_cos",
    "CHMR_lag1", "CHMR_lag2", "CHMR_lag3", "CHMR_lag12",
    "CHMR_roll3_mean", "CHMR_roll12_mean", "CHMR_prev_yoy_pct",
]
TARGET_COL = "CHMR"


# ---------------------------------------------------------------------------
# 3. TRAIN/TEST SPLIT (chronological — never random for time series)
# ---------------------------------------------------------------------------
def time_split(df: pd.DataFrame, test_months: int):
    if test_months >= len(df):
        raise ValueError("test_months must be smaller than the number of rows available")
    return df.iloc[:-test_months].reset_index(drop=True), df.iloc[-test_months:].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. METRICS
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    smape = np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)
    ) * 100
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE_%": mape, "sMAPE_%": smape}


def print_metrics(label: str, metrics: dict):
    print(f"\n  {label}")
    print(f"  {'-' * len(label)}")
    for k, v in metrics.items():
        print(f"    {k:10s}: {v:.4f}" if k == "R2" else f"    {k:10s}: {v:,.3f}")


# ---------------------------------------------------------------------------
# 5. CANDIDATE MODELS
# ---------------------------------------------------------------------------
def build_candidates():
    candidates = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=5.0)),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=5, min_samples_leaf=3,
            max_features="sqrt", random_state=42, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=150, max_depth=2, learning_rate=0.03,
            subsample=0.7, random_state=42,
        ),
    }
    if HAS_XGB:
        candidates["XGBoost"] = XGBRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7, reg_lambda=2.0,
            random_state=42, n_jobs=-1,
        )
    return candidates


def rolling_origin_backtest(model, X: pd.DataFrame, y: pd.Series, horizon: int,
                             min_train: int, step: int) -> dict:
    """
    Simulates the *actual* task the model will be judged on — train on
    everything up to some point, then predict the next `horizon` months as a
    block — repeated at several points walking forward through training
    history, and averaged. This is a closer proxy for genuine forward-looking
    accuracy than standard TimeSeriesSplit CV, which validates on short,
    scattered folds rather than one full-length forecast horizon at a time.
    Only uses data already in the training set — the real test set is never
    touched during model selection.
    """
    n = len(X)
    cutoffs = list(range(min_train, n - horizon + 1, step))
    if not cutoffs:
        # Not enough history for a proper backtest window; fall back to a
        # single split using whatever's available.
        cutoffs = [max(1, n - horizon)]

    all_true, all_pred = [], []
    for cutoff in cutoffs:
        model.fit(X.iloc[:cutoff], y.iloc[:cutoff])
        preds = model.predict(X.iloc[cutoff:cutoff + horizon])
        all_true.extend(y.iloc[cutoff:cutoff + horizon].values)
        all_pred.extend(preds)

    return compute_metrics(np.array(all_true), np.array(all_pred))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="combined_CC_v2.csv")
    parser.add_argument("--test_months", type=int, default=12,
                         help="Number of most recent months held out as OOS test set")
    parser.add_argument("--backtest_min_train", type=int, default=24,
                         help="Minimum months of history before the first backtest window "
                              "(needs enough data to fit lag features + a stable model)")
    parser.add_argument("--backtest_step", type=int, default=6,
                         help="Months to advance the backtest origin between windows")
    parser.add_argument("--exclude_test_months", type=str, default="",
                         help="Comma-separated YYYY-MM months to exclude from OOS metric "
                              "scoring only (e.g. known one-off anomalies). They stay in the "
                              "per-month table, just not in the aggregate metrics. "
                              "e.g. --exclude_test_months 2021-01,2021-02")
    args = parser.parse_args()
    exclude_months = {m.strip() for m in args.exclude_test_months.split(",") if m.strip()}

    print(f"Loading {args.csv} ...")
    raw = load_data(args.csv)
    print(f"  Loaded {len(raw)} rows spanning {raw['date'].min().date()} to {raw['date'].max().date()}")

    feat = build_features(raw)
    print(f"  {len(feat)} rows remain after lag feature engineering "
          f"(first 12 months dropped due to CHMR_lag12/lag13)")

    train_df, test_df = time_split(feat, args.test_months)
    print(f"  Train: {train_df['date'].min().date()} to {train_df['date'].max().date()} ({len(train_df)} months)")
    print(f"  Test (OOS): {test_df['date'].min().date()} to {test_df['date'].max().date()} ({len(test_df)} months)")

    train_max, test_above = train_df[TARGET_COL].max(), (test_df[TARGET_COL] > train_df[TARGET_COL].max()).mean() * 100
    if test_above > 0:
        print(f"\n  Note: {test_above:.0f}% of test months exceed the training max ({train_max:,.0f}). "
              f"Tree models can't extrapolate above that, which will bias them toward underprediction.")

    X_train, X_test = train_df[FEATURE_COLS], test_df[FEATURE_COLS]
    y_train, y_test = train_df[TARGET_COL], test_df[TARGET_COL]

    # ---- Silently evaluate every candidate via rolling-origin backtest, pick the winner ----
    # Each candidate is repeatedly trained on "everything up through some point" and asked
    # to forecast the next `test_months`-long block, walking forward through training
    # history. This is the same task shape as the real held-out evaluation, so whichever
    # model wins here should be a much better bet to also win on the real test set than
    # picking based on small-fold CV (which is what caused GradientBoosting to look best
    # on CV but lose to Ridge on the actual 2021 holdout last time).
    print(f"\nEvaluating candidate models (rolling-origin backtest, "
          f"{args.test_months}-month forecast blocks)...")
    candidates = build_candidates()
    backtest_results = {}
    for name, model in candidates.items():
        backtest_results[name] = rolling_origin_backtest(
            model, X_train, y_train,
            horizon=args.test_months,
            min_train=args.backtest_min_train,
            step=args.backtest_step,
        )
        m = backtest_results[name]
        print(f"  {name:20s} backtest RMSE: {m['RMSE']:>13,.0f}   MAPE: {m['MAPE_%']:.2f}%")

    best_name = min(backtest_results, key=lambda n: backtest_results[n]["RMSE"])
    best_model = candidates[best_name]
    print(f"\n>>> Selected model: {best_name} (lowest rolling-origin backtest RMSE)")

    # ---- Fit winner on full training set, evaluate in-sample + OOS ----
    best_model.fit(X_train, y_train)
    train_preds = best_model.predict(X_train)
    test_preds = best_model.predict(X_test)

    in_sample_metrics = compute_metrics(y_train, train_preds)
    oos_metrics = compute_metrics(y_test, test_preds)

    print(f"\n{'=' * 60}\n{best_name} — Full Report\n{'=' * 60}")
    print_metrics("In-sample (train)", in_sample_metrics)
    print_metrics("Out-of-sample (test, unseen months)", oos_metrics)

    # ---- Per-month error breakdown ----
    print(f"\n{'=' * 60}\nPER-MONTH ERROR (sorted by worst absolute % error)\n{'=' * 60}")
    month_diag = pd.DataFrame({
        "date": test_df["date"].dt.strftime("%Y-%m").values,
        "actual": y_test.values,
        "predicted": test_preds,
    })
    month_diag["abs_pct_error"] = (
        (month_diag["predicted"] - month_diag["actual"]).abs() / month_diag["actual"] * 100
    )
    print(month_diag.sort_values("abs_pct_error", ascending=False).to_string(
        index=False,
        formatters={
            "actual": lambda x: f"{x:,.0f}",
            "predicted": lambda x: f"{x:,.0f}",
            "abs_pct_error": lambda x: f"{x:.1f}%",
        },
    ))
    month_diag.to_csv("chmr_per_month_error.csv", index=False)
    print("\nSaved: chmr_per_month_error.csv")

    # ---- Steady-state metrics (excluding known anomalous months) ----
    if exclude_months:
        mask = ~month_diag["date"].isin(exclude_months)
        found = sorted(exclude_months & set(month_diag["date"]))
        print(f"\n{'=' * 60}\nSTEADY-STATE METRICS (excluding: {', '.join(found)})\n{'=' * 60}")
        if mask.sum() == 0:
            print("  All test months were excluded — nothing left to score.")
        else:
            steady_metrics = compute_metrics(month_diag.loc[mask, "actual"], month_diag.loc[mask, "predicted"])
            print_metrics(f"{best_name} ({mask.sum()} of {len(month_diag)} months retained)", steady_metrics)

    # ---- Feature importance / coefficients + plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=FEATURE_COLS).sort_values()
        importances.plot(kind="barh", ax=axes[0], color="steelblue")
        axes[0].set_title(f"Feature Importance — {best_name}")
        axes[0].set_xlabel("Importance")
    else:
        ridge_step = best_model.named_steps["ridge"]
        coefs = pd.Series(ridge_step.coef_, index=FEATURE_COLS).sort_values()
        coefs.plot(kind="barh", ax=axes[0], color="steelblue")
        axes[0].set_title(f"Standardized Coefficients — {best_name}")
        axes[0].set_xlabel("Coefficient (on scaled features)")

    axes[1].plot(train_df["date"], y_train, label="Train (actual)", color="gray", alpha=0.6)
    axes[1].plot(test_df["date"], y_test, label="Test (actual)", color="black", marker="o")
    axes[1].plot(test_df["date"], test_preds, label=f"Test (predicted, {best_name})",
                 color="red", marker="x", linestyle="--")
    axes[1].set_title("Actual vs Predicted CHMR (OOS period)")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("chmr_model_results.png", dpi=150)
    print("\nSaved plot: chmr_model_results.png")

    pd.DataFrame({"date": test_df["date"], "actual": y_test.values, "predicted": test_preds}) \
        .to_csv("chmr_oos_predictions.csv", index=False)
    print("Saved predictions: chmr_oos_predictions.csv")


if __name__ == "__main__":
    main()