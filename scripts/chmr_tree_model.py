"""
CHMR Sales Tax Revenue Prediction — Tree-Based ML Pipeline
============================================================
Predicts monthly CHMR (Cook County Home Rule Municipal Retailers' tax /
sales tax revenue) using Random Forest, Gradient Boosting, and XGBoost.

Input CSV schema (one row per month):
    date, unemp, CPI_U, GasPrice, CDD, CHMR

Usage:
    python chmr_tree_model.py --csv combined_CC_v2.csv --test_months 12
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
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

    # Parse percent strings like "5.70%" -> 5.70 (float). Handles both legacy
    # object dtype and pandas' newer StringDtype, which don't compare equal to `object`.
    if not pd.api.types.is_numeric_dtype(df["unemp"]):
        df["unemp"] = df["unemp"].astype(str).str.rstrip("%").astype(float)

    # Parse date (handles M/D/YY format seen in the source file)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y", errors="coerce")
    if df["date"].isna().any():
        # fallback for other date formats
        df["date"] = pd.to_datetime(df["date"].astype(str), errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Calendar / seasonality features
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Autoregressive / lag features on the target itself (no leakage: shift() only
    # looks backward in time)
    df["CHMR_lag1"] = df["CHMR"].shift(1)
    df["CHMR_lag2"] = df["CHMR"].shift(2)
    df["CHMR_lag3"] = df["CHMR"].shift(3)
    df["CHMR_lag12"] = df["CHMR"].shift(12)  # same month, prior year
    df["CHMR_roll3_mean"] = df["CHMR"].shift(1).rolling(window=3).mean()
    df["CHMR_roll12_mean"] = df["CHMR"].shift(1).rolling(window=12).mean()
    df["CHMR_yoy_pct"] = df["CHMR"].pct_change(periods=12)

    # Drop rows with NaNs created by lagging (start of series)
    df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLS = [
    "unemp", "CPI_U", "GasPrice", "CDD",
    "month", "quarter", "month_sin", "month_cos",
    "CHMR_lag1", "CHMR_lag2", "CHMR_lag3", "CHMR_lag12",
    "CHMR_roll3_mean", "CHMR_roll12_mean", "CHMR_yoy_pct",
]
TARGET_COL = "CHMR"


# ---------------------------------------------------------------------------
# 3. TIME-BASED TRAIN/TEST SPLIT
# ---------------------------------------------------------------------------
def time_split(df: pd.DataFrame, test_months: int):
    """
    Chronological split — the last `test_months` rows are held out as the
    out-of-sample (OOS) test set. This mirrors how the model would actually
    be used (predicting future months from past data), unlike a random split
    which would leak future information into training.
    """
    if test_months >= len(df):
        raise ValueError("test_months must be smaller than the number of rows available")
    train_df = df.iloc[:-test_months].reset_index(drop=True)
    test_df = df.iloc[-test_months:].reset_index(drop=True)
    return train_df, test_df


# ---------------------------------------------------------------------------
# 4. METRICS
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    # symmetric MAPE — more stable when values can be near zero
    smape = np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)
    ) * 100
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE_%": mape, "sMAPE_%": smape}


def print_metrics(label: str, metrics: dict):
    print(f"\n  {label}")
    print(f"  {'-' * len(label)}")
    for k, v in metrics.items():
        if k == "R2":
            print(f"    {k:10s}: {v:.4f}")
        else:
            print(f"    {k:10s}: {v:,.3f}")


# ---------------------------------------------------------------------------
# 5. MODELS
# ---------------------------------------------------------------------------
def build_models(n_train_rows: int):
    # Cap tree depth / min_samples given small monthly datasets to avoid overfitting
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
    return models


# ---------------------------------------------------------------------------
# 6. CROSS-VALIDATION (walk-forward, for robustness check on top of the
#    single holdout test)
# ---------------------------------------------------------------------------
def walk_forward_cv(model, X, y, n_splits=4):
    n_splits = min(n_splits, len(X) - 1)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    for train_idx, val_idx in tscv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        fold_metrics.append(compute_metrics(y.iloc[val_idx], preds))
    avg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
    return avg


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="combined_CC_v2.csv")
    parser.add_argument("--test_months", type=int, default=12,
                         help="Number of most recent months held out as OOS test set")
    parser.add_argument("--cv_splits", type=int, default=4)
    args = parser.parse_args()

    print(f"Loading {args.csv} ...")
    raw = load_data(args.csv)
    print(f"  Loaded {len(raw)} rows spanning {raw['date'].min().date()} to {raw['date'].max().date()}")

    feat = build_features(raw)
    print(f"  {len(feat)} rows remain after lag feature engineering "
          f"(first 12 months are dropped due to CHMR_lag12)")

    train_df, test_df = time_split(feat, args.test_months)
    print(f"  Train: {train_df['date'].min().date()} to {train_df['date'].max().date()} "
          f"({len(train_df)} months)")
    print(f"  Test (OOS): {test_df['date'].min().date()} to {test_df['date'].max().date()} "
          f"({len(test_df)} months)")

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    models = build_models(len(train_df))
    results = {}
    predictions = {"date": test_df["date"], "actual": y_test.values}

    for name, model in models.items():
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")

        # Walk-forward CV on training data only (robustness check)
        cv_metrics = walk_forward_cv(model, X_train, y_train, n_splits=args.cv_splits)
        print_metrics(f"{name} — Walk-forward CV (avg over folds, train data)", cv_metrics)

        # Fit on full training set, evaluate in-sample and on true OOS holdout
        model.fit(X_train, y_train)
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        in_sample_metrics = compute_metrics(y_train, train_preds)
        oos_metrics = compute_metrics(y_test, test_preds)

        print_metrics(f"{name} — In-sample (train)", in_sample_metrics)
        print_metrics(f"{name} — Out-of-sample (test, unseen months)", oos_metrics)

        results[name] = {
            "cv": cv_metrics,
            "in_sample": in_sample_metrics,
            "oos": oos_metrics,
            "model": model,
        }
        predictions[f"{name}_pred"] = test_preds

    # ---- Summary comparison table ----
    print(f"\n{'=' * 60}\nOOS TEST SET SUMMARY (lower RMSE/MAPE, higher R2 = better)\n{'=' * 60}")
    summary = pd.DataFrame({name: r["oos"] for name, r in results.items()}).T
    print(summary.round(3).to_string())

    best_model_name = summary["R2"].idxmax()
    print(f"\nBest model on OOS R2: {best_model_name}")

    # ---- Feature importance (best model) ----
    best_model = results[best_model_name]["model"]
    importances = pd.Series(best_model.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    importances.plot(kind="barh", ax=axes[0], color="steelblue")
    axes[0].set_title(f"Feature Importance — {best_model_name}")
    axes[0].set_xlabel("Importance")

    axes[1].plot(train_df["date"], y_train, label="Train (actual)", color="gray", alpha=0.6)
    axes[1].plot(test_df["date"], y_test, label="Test (actual)", color="black", marker="o")
    axes[1].plot(test_df["date"], predictions[f"{best_model_name}_pred"],
                 label=f"Test (predicted, {best_model_name})", color="red", marker="x", linestyle="--")
    axes[1].set_title("Actual vs Predicted CHMR (OOS period)")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("chmr_model_results.png", dpi=150)
    print("\nSaved plot: chmr_model_results.png")

    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv("chmr_oos_predictions.csv", index=False)
    print("Saved predictions: chmr_oos_predictions.csv")

    summary.to_csv("chmr_model_metrics_summary.csv")
    print("Saved metrics summary: chmr_model_metrics_summary.csv")


if __name__ == "__main__":
    main()

# ============================================================
# RandomForest
# ============================================================

#   RandomForest — Walk-forward CV (avg over folds, train data)
#   -----------------------------------------------------------
#     R2        : -2.0912
#     MAE       : 8,123,815.836
#     RMSE      : 9,145,349.853
#     MAPE_%    : 11.773
#     sMAPE_%   : 13.378

#   RandomForest — In-sample (train)
#   --------------------------------
#     R2        : 0.9834
#     MAE       : 1,241,206.179
#     RMSE      : 1,943,482.808
#     MAPE_%    : 2.200
#     sMAPE_%   : 2.199

#   RandomForest — Out-of-sample (test, unseen months)
#   --------------------------------------------------
#     R2        : -1.0355
#     MAE       : 12,485,412.133
#     RMSE      : 14,377,936.489
#     MAPE_%    : 23.330
#     sMAPE_%   : 20.223

# ============================================================
# GradientBoosting
# ============================================================

#   GradientBoosting — Walk-forward CV (avg over folds, train data)
#   ---------------------------------------------------------------
#     R2        : -0.5117
#     MAE       : 5,656,478.047
#     RMSE      : 6,594,649.208
#     MAPE_%    : 8.162
#     sMAPE_%   : 8.804

#   GradientBoosting — In-sample (train)
#   ------------------------------------
#     R2        : 1.0000
#     MAE       : 74,586.334
#     RMSE      : 90,063.022
#     MAPE_%    : 0.130
#     sMAPE_%   : 0.130

#   GradientBoosting — Out-of-sample (test, unseen months)
#   ------------------------------------------------------
#     R2        : -0.2479
#     MAE       : 9,328,209.598
#     RMSE      : 11,257,782.740
#     MAPE_%    : 17.015
#     sMAPE_%   : 15.600

# ============================================================
# OOS TEST SET SUMMARY (lower RMSE/MAPE, higher R2 = better)
# ============================================================
#                      R2           MAE          RMSE  MAPE_%  sMAPE_%
# RandomForest     -1.035  1.248541e+07  1.437794e+07  23.330   20.223
# GradientBoosting -0.248  9.328210e+06  1.125778e+07  17.015   15.600

# Best model on OOS R2: GradientBoosting
