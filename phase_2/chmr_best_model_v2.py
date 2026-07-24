"""
CHMR Sales Tax Revenue Prediction — Best Model Auto-Select (v2)
============================================================
Extended model featuring local Chicago indicators: CTA ridership, OpenStreetMap
retail density metrics, and local consumer search sentiment.

Usage:
    python chmr_best_model_v2.py --csv combined_CC_v5.csv --test_months 12
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

    # Standardize date column name
    date_col = "date_dt" if "date_dt" in df.columns else "date"
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")

    # Clean percent string in unemp if necessary
    if not pd.api.types.is_numeric_dtype(df["unemp"]):
        df["unemp"] = df["unemp"].astype(str).str.rstrip("%").astype(float)

    # Fill NaN in osm_net_new for the first month
    if "osm_net_new" in df.columns:
        df["osm_net_new"] = df["osm_net_new"].fillna(0)

    df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Calendar / Cyclical
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Target (CHMR) Lags & Moving Averages
    df["CHMR_lag1"] = df["CHMR"].shift(1)
    df["CHMR_lag2"] = df["CHMR"].shift(2)
    df["CHMR_lag3"] = df["CHMR"].shift(3)
    df["CHMR_lag12"] = df["CHMR"].shift(12)
    df["CHMR_lag13"] = df["CHMR"].shift(13)
    df["CHMR_roll3_mean"] = df["CHMR"].shift(1).rolling(window=3).mean()
    df["CHMR_roll12_mean"] = df["CHMR"].shift(1).rolling(window=12).mean()
    df["CHMR_prev_yoy_pct"] = (df["CHMR_lag1"] - df["CHMR_lag13"]) / df["CHMR_lag13"]

    # --- NEW FEATURE ENGINERING FOR EXTENDED DATA ---
    # CTA Transit Mobility
    if "CTA" in df.columns:
        df["CTA_lag1"] = df["CTA"].shift(1)
        df["CTA_lag12"] = df["CTA"].shift(12)
        df["CTA_prev_yoy_pct"] = (df["CTA_lag1"] - df["CTA_lag12"]) / (df["CTA_lag12"] + 1e-9)

    # OpenStreetMap Retail Infrastructure
    if "osm_shop_count" in df.columns:
        df["osm_shop_count_lag1"] = df["osm_shop_count"].shift(1)
    if "osm_net_new" in df.columns:
        df["osm_net_new_lag1"] = df["osm_net_new"].shift(1)

    # Chicago Search Sentiment
    sentiment_col = "chicago_search_sentiment" if "chicago_search_sentiment" in df.columns else "chicago_sentiment"
    if sentiment_col in df.columns:
        df["sentiment_lag1"] = df[sentiment_col].shift(1)
        df["sentiment_roll3_mean"] = df[sentiment_col].shift(1).rolling(window=3).mean()

    df = df.dropna().reset_index(drop=True)
    return df


# Define explicit feature set incorporating all macro and local Chicago features
FEATURE_COLS = [
    # Macro / Original
    "unemp", "CPI_U", "GasPrice", "CDD",
    "month", "quarter", "month_sin", "month_cos",
    "CHMR_lag1", "CHMR_lag2", "CHMR_lag3", "CHMR_lag12",
    "CHMR_roll3_mean", "CHMR_roll12_mean", "CHMR_prev_yoy_pct",
    # New Local Chicago Features
    "CTA_lag1", "CTA_prev_yoy_pct",
    "osm_shop_count_lag1", "osm_net_new_lag1",
    "sentiment_lag1", "sentiment_roll3_mean"
]

TARGET_COL = "CHMR"


# ---------------------------------------------------------------------------
# 3. METRICS & TIME SPLIT
# ---------------------------------------------------------------------------
def time_split(df: pd.DataFrame, test_months: int):
    return df.iloc[:-test_months].reset_index(drop=True), df.iloc[-test_months:].reset_index(drop=True)


def compute_metrics(y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)) * 100
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE_%": mape, "sMAPE_%": smape}


def print_metrics(label: str, metrics: dict):
    print(f"\n  {label}")
    print(f"  {'-' * len(label)}")
    for k, v in metrics.items():
        print(f"    {k:10s}: {v:.4f}" if k == "R2" else f"    {k:10s}: {v:,.3f}")


# ---------------------------------------------------------------------------
# 4. CANDIDATE MODELS & BACKTESTING
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
    n = len(X)
    cutoffs = list(range(min_train, n - horizon + 1, step))
    if not cutoffs:
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
    parser.add_argument("--csv", type=str, default="combined_CC_v5.csv")
    parser.add_argument("--test_months", type=int, default=12)
    parser.add_argument("--backtest_min_train", type=int, default=24)
    parser.add_argument("--backtest_step", type=int, default=6)
    args = parser.parse_args()

    print(f"Loading {args.csv} ...")
    raw = load_data(args.csv)
    print(f"  Loaded {len(raw)} rows spanning {raw['date'].min().date()} to {raw['date'].max().date()}")

    feat = build_features(raw)
    
    # Filter available feature columns dynamically
    available_features = [col for col in FEATURE_COLS if col in feat.columns]
    print(f"  Using {len(available_features)} feature columns.")

    train_df, test_df = time_split(feat, args.test_months)
    print(f"  Train: {train_df['date'].min().date()} to {train_df['date'].max().date()} ({len(train_df)} months)")
    print(f"  Test (OOS): {test_df['date'].min().date()} to {test_df['date'].max().date()} ({len(test_df)} months)")

    X_train, X_test = train_df[available_features], test_df[available_features]
    y_train, y_test = train_df[TARGET_COL], test_df[TARGET_COL]

    # Model evaluation via rolling-origin backtest
    print(f"\nEvaluating candidate models with Chicago local features...")
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

    # Train final model
    best_model.fit(X_train, y_train)
    train_preds = best_model.predict(X_train)
    test_preds = best_model.predict(X_test)

    print(f"\n{'=' * 60}\n{best_name} — Extended Model Diagnostics\n{'=' * 60}")
    print_metrics("In-sample (train)", compute_metrics(y_train, train_preds))
    print_metrics("Out-of-sample (test)", compute_metrics(y_test, test_preds))

    # Feature Importance / Coefficients Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=available_features).sort_values()
        importances.plot(kind="barh", ax=axes[0], color="teal")
        axes[0].set_title(f"Feature Importance — {best_name}")
    else:
        ridge_step = best_model.named_steps["ridge"]
        coefs = pd.Series(ridge_step.coef_, index=available_features).sort_values()
        coefs.plot(kind="barh", ax=axes[0], color="teal")
        axes[0].set_title(f"Standardized Coefficients — {best_name}")

    axes[1].plot(train_df["date"], y_train, label="Train (actual)", color="gray", alpha=0.6)
    axes[1].plot(test_df["date"], y_test, label="Test (actual)", color="black", marker="o")
    axes[1].plot(test_df["date"], test_preds, label=f"Test (predicted)", color="crimson", marker="x", linestyle="--")
    axes[1].set_title("Actual vs Predicted CHMR (Extended Features)")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("chmr_extended_model_results.png", dpi=150)
    print("\nSaved plot: chmr_extended_model_results.png")


if __name__ == "__main__":
    main()

# Evaluating candidate models with Chicago local features...
#   Ridge                backtest RMSE:    11,068,627   MAPE: 13.56% --> actual best model selected by Gemini
#   RandomForest         backtest RMSE:     5,346,510   MAPE: 6.13%
#   GradientBoosting     backtest RMSE:     4,323,774   MAPE: 4.79%

# >>> Selected model: GradientBoosting (lowest rolling-origin backtest RMSE)

# ============================================================
# GradientBoosting — Extended Model Diagnostics
# ============================================================

#   In-sample (train)
#   -----------------
#     R2        : 0.9848
#     MAE       : 1,424,444.097
#     RMSE      : 1,790,379.176
#     MAPE_%    : 2.497
#     sMAPE_%   : 2.480

#   Out-of-sample (test)
#   --------------------
#     R2        : -0.4485
#     MAE       : 10,515,920.867
#     RMSE      : 12,128,770.559
#     MAPE_%    : 19.199
#     sMAPE_%   : 17.506