"""
Software Job Openings Forecaster
---------------------------------
Predicts monthly software job openings using a Gradient Boosting model
trained on lagged and seasonal features.

This prototype ships with synthetic-but-realistic historical data so it
runs end to end with zero setup. Swap in real data by replacing
`build_historical_data()` with a CSV loader — see the README for the
exact format expected.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------

def build_historical_data(start="2015-01-01", end="2026-08-01", seed=7):
    """
    Generates monthly synthetic software job-opening counts with:
      - a long-run upward trend
      - yearly seasonality (hiring pushes in Jan/Sep, dips in Jul/Dec)
      - a 2020 pandemic shock
      - the 2022-2023 tech hiring boom-then-correction
      - random noise

    Replace this function with a real loader, e.g.:
        df = pd.read_csv("job_openings.csv", parse_dates=["month"])
    The rest of the pipeline only needs two columns: 'month', 'openings'.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range(start=start, end=end, freq="MS")
    n = len(months)

    month_nums = np.asarray(months.month)

    t = np.arange(n)
    trend = 8000 + 55 * t                      # steady long-run growth
    seasonality = 900 * np.sin(2 * np.pi * (month_nums - 1) / 12 + 0.4)

    values = np.asarray(trend + seasonality, dtype=float)

    for i, m in enumerate(months):
        # 2020 pandemic dip
        if m.year == 2020 and 3 <= m.month <= 8:
            values[i] *= 0.62
        elif m.year == 2020 and m.month > 8:
            values[i] *= 0.85
        # 2021-2022 tech hiring boom
        if 2021 <= m.year <= 2022:
            values[i] *= 1.28
        # 2023 correction / layoffs
        if m.year == 2023:
            values[i] *= 0.80
        # 2024-2026 gradual, AI-driven partial rebound
        if m.year >= 2024:
            recovery = min(0.18, 0.02 * (m.year - 2023))
            values[i] *= (0.88 + recovery)

    noise = rng.normal(0, 260, n)
    values = np.clip(values + noise, 500, None).round().astype(int)

    return pd.DataFrame({"month": months, "openings": values})


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

def make_features(df):
    df = df.copy()
    df["month_num"] = df["month"].dt.month
    df["time_idx"] = np.arange(len(df))
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    for lag in (1, 3, 6, 12):
        df[f"lag_{lag}"] = df["openings"].shift(lag)

    df["roll_mean_3"] = df["openings"].shift(1).rolling(3).mean()
    df["roll_mean_6"] = df["openings"].shift(1).rolling(6).mean()

    return df


FEATURE_COLS = [
    "time_idx", "month_sin", "month_cos",
    "lag_1", "lag_3", "lag_6", "lag_12",
    "roll_mean_3", "roll_mean_6",
]


# ---------------------------------------------------------------------------
# 3. Train + evaluate (holdout = last 12 months)
# ---------------------------------------------------------------------------

def train_and_evaluate(df_feat):
    df_model = df_feat.dropna().reset_index(drop=True)

    holdout = 12
    train = df_model.iloc[:-holdout]
    test = df_model.iloc[-holdout:]

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(train[FEATURE_COLS], train["openings"])

    preds = model.predict(test[FEATURE_COLS])
    mae = mean_absolute_error(test["openings"], preds)
    mape = mean_absolute_percentage_error(test["openings"], preds) * 100

    print(f"Holdout evaluation (last {holdout} months):")
    print(f"  MAE:  {mae:,.0f} openings")
    print(f"  MAPE: {mape:.1f}%")

    # refit on all available data for the actual forecast
    final_model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    final_model.fit(df_model[FEATURE_COLS], df_model["openings"])

    residual_std = np.std(test["openings"].values - preds)

    return final_model, mae, mape, residual_std


# ---------------------------------------------------------------------------
# 4. Forecast forward (recursive: each prediction feeds the next month's lags)
# ---------------------------------------------------------------------------

def forecast_forward(model, df_raw, periods=12, residual_std=0.0):
    history = df_raw.copy()
    forecasts = []

    for step in range(periods):
        feat_hist = make_features(history)
        last_row = feat_hist.iloc[[-1]]

        next_month = history["month"].iloc[-1] + pd.DateOffset(months=1)
        next_time_idx = last_row["time_idx"].values[0] + 1
        next_month_num = next_month.month

        row = pd.DataFrame({
            "time_idx": [next_time_idx],
            "month_sin": [np.sin(2 * np.pi * next_month_num / 12)],
            "month_cos": [np.cos(2 * np.pi * next_month_num / 12)],
            "lag_1": [history["openings"].iloc[-1]],
            "lag_3": [history["openings"].iloc[-3]],
            "lag_6": [history["openings"].iloc[-6]],
            "lag_12": [history["openings"].iloc[-12]],
            "roll_mean_3": [history["openings"].iloc[-3:].mean()],
            "roll_mean_6": [history["openings"].iloc[-6:].mean()],
        })

        pred = float(model.predict(row[FEATURE_COLS])[0])
        forecasts.append({"month": next_month, "openings": pred})

        history = pd.concat(
            [history, pd.DataFrame({"month": [next_month], "openings": [pred]})],
            ignore_index=True,
        )

    fc = pd.DataFrame(forecasts)
    fc["lower_80"] = fc["openings"] - 1.28 * residual_std
    fc["upper_80"] = fc["openings"] + 1.28 * residual_std
    return fc


# ---------------------------------------------------------------------------
# 5. Plot + save
# ---------------------------------------------------------------------------

def plot_and_save(df_hist, df_fc, out_png):
    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(df_hist["month"], df_hist["openings"], color="#2C3868",
            linewidth=1.6, label="Historical openings")
    ax.plot(df_fc["month"], df_fc["openings"], color="#C1402F",
            linewidth=1.8, linestyle="--", label="Forecast")
    ax.fill_between(df_fc["month"], df_fc["lower_80"], df_fc["upper_80"],
                     color="#C1402F", alpha=0.15, label="80% interval")

    ax.set_title("Software Job Openings — Historical vs. 12-Month Forecast")
    ax.set_ylabel("Monthly openings")
    ax.legend(loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    print(f"Saved chart to {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df_raw = build_historical_data()
    df_feat = make_features(df_raw)

    model, mae, mape, residual_std = train_and_evaluate(df_feat)
    df_fc = forecast_forward(model, df_raw, periods=12, residual_std=residual_std)

    print("\nNext 12 months forecast:")
    print(df_fc[["month", "openings", "lower_80", "upper_80"]]
          .assign(month=lambda d: d["month"].dt.strftime("%Y-%m"))
          .round(0)
          .to_string(index=False))

    df_fc.to_csv("/mnt/user-data/outputs/job_openings_forecast.csv", index=False)
    df_raw.to_csv("/mnt/user-data/outputs/job_openings_history.csv", index=False)
    plot_and_save(df_raw, df_fc, "/mnt/user-data/outputs/job_openings_forecast.png")


if __name__ == "__main__":
    main()
