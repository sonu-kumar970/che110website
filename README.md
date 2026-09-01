# Software Job Openings Forecaster

Predicts monthly software job openings 12 months ahead using a Gradient
Boosting model trained on lag and seasonal features.

## What's in this folder
- `job_openings_forecast.py` — the full pipeline: data, features, training, evaluation, forecast, chart
- `job_openings_forecast.png` — historical data + 12-month forecast with an 80% interval band
- `job_openings_forecast.csv` — the forecasted months as raw numbers
- `job_openings_history.csv` — the historical series the model was trained on

## How to run it
```
pip install pandas numpy scikit-learn matplotlib
python3 job_openings_forecast.py
```
It prints a holdout accuracy check (MAE / MAPE on the last 12 known months)
and the 12-month forecast table, then writes the CSV and PNG files above.

## Important: the data is synthetic
There's no live jobs API wired in, so the script generates a realistic
monthly series (trend + seasonality + the 2020 dip + the 2022 hiring boom +
the 2023 correction) so you have something to run immediately. The model
and pipeline are real — only the input data is simulated.

**To use real data**, replace `build_historical_data()` in the script with
a loader for your source, e.g.:
```python
df = pd.read_csv("your_data.csv", parse_dates=["month"])
```
The rest of the pipeline only needs two columns: `month` (first-of-month
date) and `openings` (a count). Good real sources to plug in:
- BLS JOLTS (Job Openings and Labor Turnover Survey) — has an "Information"
  and broader tech-adjacent series, free API
- Indeed Hiring Lab's public job postings index
- LinkedIn Economic Graph (partner access required)
- Your own ATS/job-board export, if you're forecasting for a specific company

## What I'd extend first, in order
1. **Real data source** — this is the highest-leverage change; swap the
   synthetic generator for a real feed (see above) and the model stays the same.
2. **A proper time-series model comparison** — try SARIMAX and Prophet
   alongside the current Gradient Boosting model and keep whichever backtests
   best; different data shapes favor different models.
3. **Exogenous features** — feed in signals that actually drive hiring, e.g.
   NASDAQ tech index, interest rates, or company earnings sentiment, as
   additional model inputs.
4. **A small web dashboard** — wrap this in a Flask/FastAPI endpoint that
   returns the forecast as JSON, and pair it with a chart-based frontend
   (like the ones we've built earlier in this conversation) so it's
   browsable instead of script-only.
5. **Automatic retraining** — schedule the script to re-run monthly as new
   real data comes in, so the forecast stays current instead of going stale.
