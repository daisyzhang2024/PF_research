"""
nyc_model_lib.py
================
Reusable functions for NYC sales tax revenue modeling. Same approach as
chmr_model_lib.py / la_model_lib.py, adapted to this dataset's columns:
  target      : "sales_tax"
  date column : "date"
  macro cols  : unemp, GasPrice, ridership, nyc_search

Note: unlike the LA dataset, this one has no CPI or news-sentiment
column -- just 4 macro predictors (unemployment, gas price, transit
ridership, and Google search-interest sentiment).

This file has NO top-level execution -- importing it does not run the
backtest or any forecast. Use run_backtest_nyc.py to re-evaluate all
target/model combos, or run_forecast_nyc.py to predict a single future
month with the best combo.

Modeling approach (see run_backtest_nyc.py docstring for full rationale):
  - Model de-leveled targets (trend deviation, MoM log-change, YoY
    log-change) instead of the raw dollar level, then reconstruct the
    level afterward.
  - Add calendar month dummies (Feb..Dec, Jan is baseline) to capture
    seasonality explicitly.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

DATA_PATH = "combined_nyc.csv"
DATE_COL = "date"
TARGET_COL = "sales_tax"
N_TEST = 24          # rolling-origin backtest window (months held out, walked forward)
RANDOM_STATE = 42

MACRO_COLS = ["unemp", "GasPrice", "ridership", "nyc_search"]


# ---------------------------------------------------------------------
# 1. Load & basic feature engineering
# ---------------------------------------------------------------------

def load_data(path=DATA_PATH):
    df = pd.read_csv(path, thousands=",")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df["t"] = np.arange(len(df))                 # linear time index for trend
    df["month"] = df[DATE_COL].dt.month
    df["log_target"] = np.log(df[TARGET_COL])
    return df


def add_month_dummies(df):
    # drop_first=True drops January (month==1) as the baseline category
    dummies = pd.get_dummies(df["month"], prefix="m", drop_first=True)
    return pd.concat([df, dummies], axis=1)


def add_macro_changes(df):
    # macro predictors as both level and MoM/YoY change so the
    # differenced targets (MoM/YoY) have differenced predictors too
    out = df.copy()
    for c in MACRO_COLS:
        out[f"{c}_chg1"] = out[c].diff(1)
        out[f"{c}_chg12"] = out[c].diff(12)
    return out


# ---------------------------------------------------------------------
# 2. Target construction
# ---------------------------------------------------------------------

def build_targets(df):
    """
    Returns df with two additional target columns:
      mom_logdiff : log(target_t) - log(target_{t-1})
      yoy_logdiff : log(target_t) - log(target_{t-12})
    (trend_dev is added separately via fit_trend, since it needs to be
    refit per-fold during backtesting to avoid leakage.)
    """
    df = df.copy()
    df["mom_logdiff"] = df["log_target"].diff(1)
    df["yoy_logdiff"] = df["log_target"].diff(12)
    return df


def prepare_full_dataset(path=DATA_PATH):
    """Convenience wrapper: load + all feature/target engineering in one call."""
    df = load_data(path)
    df = add_month_dummies(df)
    df = add_macro_changes(df)
    df = build_targets(df)
    full_trend_fn = fit_trend(df["t"].values, df["log_target"].values)
    df["trend_dev"] = df["log_target"] - full_trend_fn(df["t"].values)
    return df


def fit_trend(t_train, y_train_log):
    """OLS log(target) ~ t on the TRAIN slice only; returns predict(t) fn."""
    X = sm.add_constant(t_train)
    model = sm.OLS(y_train_log, X).fit()
    return lambda t_new: model.predict(sm.add_constant(t_new, has_constant="add"))


# ---------------------------------------------------------------------
# 3. Models
# ---------------------------------------------------------------------

def get_models():
    return {
        "Ridge": Ridge(alpha=5.0, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, max_depth=5, min_samples_leaf=3,
            random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            random_state=RANDOM_STATE),
    }


# ---------------------------------------------------------------------
# 4. Rolling-origin backtest for each (target-formulation x model)
# ---------------------------------------------------------------------

def feature_cols_for(target_kind, df_columns):
    """df_columns: the .columns of your prepared df (needed to find month-dummy cols)."""
    macro = MACRO_COLS + [f"{c}_chg1" for c in MACRO_COLS]
    if target_kind == "yoy_logdiff":
        macro = macro + [f"{c}_chg12" for c in MACRO_COLS]
    month_dummies = [c for c in df_columns if c.startswith("m_")]
    return macro + month_dummies


def evaluate_levels(y_true, y_pred):
    err = y_true - y_pred
    rmse = np.sqrt(np.mean(err ** 2))
    mape = np.mean(np.abs(err) / np.abs(y_true)) * 100
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {"R2": r2, "MAPE_%": mape, "RMSE": rmse}


def backtest(df, target_kind, model_name, n_test=N_TEST):
    """
    Walk-forward: for each held-out month i (last n_test months), fit
    on all data strictly before i, predict month i, reconstruct the
    absolute target level, then move to i+1 (expanding window).
    """
    n = len(df)
    start = n - n_test
    feats = feature_cols_for(target_kind, df.columns)
    preds_level, actual_level, dates = [], [], []

    for i in range(start, n):
        train = df.iloc[:i].dropna(subset=feats + [target_kind]).copy()
        test_row = df.iloc[[i]]

        model = get_models()[model_name]
        X_train = train[feats].values
        y_train = train[target_kind].values

        scaler = StandardScaler().fit(X_train)
        model.fit(scaler.transform(X_train), y_train)

        X_test = test_row[feats].values
        pred_target = model.predict(scaler.transform(X_test))[0]

        # --- reconstruct absolute level ---
        if target_kind == "trend_dev":
            trend_fn = fit_trend(train["t"].values, train["log_target"].values)
            trend_at_i = trend_fn(np.array([df["t"].iloc[i]]))[0]
            log_level_pred = trend_at_i + pred_target
        elif target_kind == "mom_logdiff":
            log_level_pred = df["log_target"].iloc[i - 1] + pred_target
        elif target_kind == "yoy_logdiff":
            log_level_pred = df["log_target"].iloc[i - 12] + pred_target
        else:
            raise ValueError(target_kind)

        preds_level.append(np.exp(log_level_pred))
        actual_level.append(df[TARGET_COL].iloc[i])
        dates.append(df[DATE_COL].iloc[i])

    metrics = evaluate_levels(np.array(actual_level), np.array(preds_level))
    return metrics, pd.DataFrame({
        "date": dates, "actual": actual_level, "predicted": preds_level
    })


# ---------------------------------------------------------------------
# 5. Forecast a single future/unseen month (nowcasting: requires that
#    month's macro inputs already be known -- see run_forecast_nyc.py)
# ---------------------------------------------------------------------

def forecast_next_month(df, next_month_features: dict, target_kind="yoy_logdiff",
                         model_name="RandomForest"):
    """
    Refits <target_kind> | <model_name> on ALL available history, then
    predicts the target for one additional month.

    IMPORTANT: target_kind/model_name should match whatever run_backtest_nyc.py
    found to be the best-performing combo (check nyc_delevel_results.csv --
    check that file after running the backtest, since the default args
    here may not match the actual winner for this dataset).

    next_month_features must supply values for that month for:
      unemp, GasPrice, ridership, nyc_search, and 'month' (1-12).
    These must be values you already know/have for that month --
    NOT values you're also trying to forecast (see run_forecast_nyc.py
    notes on multi-month-ahead forecasting).

    If target_kind == "yoy_logdiff", next_month_features should describe
    the month you're forecasting (e.g. May 2026), and df must already
    contain the actual same-month-last-year row (e.g. May 2025) so the
    year-over-year change and reconstruction have something to anchor to.

    Returns the predicted target level (in dollars).
    """
    feats = feature_cols_for(target_kind, df.columns)

    last_row = df.iloc[-1]
    new_row = {}
    for c in MACRO_COLS:
        new_row[c] = next_month_features[c]
        new_row[f"{c}_chg1"] = next_month_features[c] - last_row[c]
        if target_kind == "yoy_logdiff":
            row_12mo_ago = df.iloc[-12]
            new_row[f"{c}_chg12"] = next_month_features[c] - row_12mo_ago[c]
        else:
            new_row[f"{c}_chg12"] = np.nan  # unused for this target_kind
    for m in range(2, 13):
        new_row[f"m_{m}"] = 1 if next_month_features["month"] == m else 0

    X_train = df.dropna(subset=feats + [target_kind])[feats].values
    y_train = df.dropna(subset=feats + [target_kind])[target_kind].values
    scaler = StandardScaler().fit(X_train)
    model = get_models()[model_name]
    model.fit(scaler.transform(X_train), y_train)

    X_new = pd.DataFrame([new_row])[feats].values
    pred_target = model.predict(scaler.transform(X_new))[0]

    if target_kind == "trend_dev":
        next_t = df["t"].iloc[-1] + 1
        trend_fn = fit_trend(df["t"].values, df["log_target"].values)
        trend_at_next = trend_fn(np.array([next_t]))[0]
        log_level_pred = trend_at_next + pred_target
    elif target_kind == "mom_logdiff":
        log_level_pred = last_row["log_target"] + pred_target
    elif target_kind == "yoy_logdiff":
        row_12mo_ago = df.iloc[-12]
        log_level_pred = row_12mo_ago["log_target"] + pred_target
    else:
        raise ValueError(target_kind)

    return np.exp(log_level_pred)
