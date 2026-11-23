# SPY directional forecasting and strategy backtesting

Can a model using daily price and volume data forecast whether SPY will rise between the next two market opens? This project compares a constant probability, logistic regression, XGBoost, an MLP and an LSTM. A second experiment asks whether the forecasts improve a long/cash strategy after trading costs. Buy & Hold is the investment benchmark.

This executed archive includes the research code, nine notebooks, their inputs and their results. Running the experiments generates the market snapshot, fitted models and result tables. Notebook 09 evaluates the frozen Phase 4 strategy on the 2022–2026 final holdout.

## Run the project

From the extracted project directory, create and activate a virtual environment. Python 3.11 or 3.12 is recommended.

| Platform | Create environment | Activate |
|---|---|---|
| Windows PowerShell | `py -3.11 -m venv .venv` | `.\.venv\Scripts\Activate.ps1` |
| Linux or macOS | `python3 -m venv .venv` | `source .venv/bin/activate` |

Install the package and its research dependencies, then run the experiments in order.

```sh
python -m pip install -e ".[all]"
python scripts/run_research.py
jupyter lab
```

The first run downloads SPY data from Yahoo Finance. Later runs verify and reuse the local snapshot. The development runner (Phases 1–4) trains the models, generates the backtests and runs the tests. The default neural searches alone require 48 MLP fits and 72 LSTM fits, plus two final refits.

To run only the backtests after generating the forecasts, use `python scripts/run_research.py --phase4-only`. It reads the training and validation parquet files and `reports/phase3_2/validation_predictions.csv`. To run only the offline test suite, use `python -m pytest -q`.

## Follow the experiments

Run an individual step with `python -m quant_trading.<module>` from the project root. Open its notebook after the step has produced its files. Notebook 01 reads the prepared market data. Notebooks 02–08 read the generated reports.

| Notebook | Module | What to inspect | Output directory |
|---|---|---|---|
| [01 · Market data and EDA](notebooks/01_data/01_market_data_eda.ipynb) | `pipeline` | Trading calendar, return distribution, 17 features, label timing and chronological splits | `data/` |
| [02 · Baselines](notebooks/02_classical_ml/02_ml_baselines.ipynb) | `phase2_baselines` | Training-fold class prior, scaled L2 logistic regression and calibration | `reports/phase2_1/` |
| [03 · XGBoost](notebooks/02_classical_ml/03_xgboost_forecasting.ipynb) | `phase2_xgboost` | Walk-forward comparison of eight configurations and feature importance | `reports/phase2_2/` |
| [04 · Forecast diagnostics](notebooks/02_classical_ml/04_robustness_diagnostics.ipynb) | `phase2_robustness` | Paired loss differences, block bootstrap intervals and stability across years and regimes | `reports/phase2_3/` |
| [05 · MLP](notebooks/03_neural_models/05_mlp_forecasting.ipynb) | `phase3_mlp` | Feedforward forecasts, seed variability and training loss | `reports/phase3_1/` |
| [06 · LSTM](notebooks/03_neural_models/06_lstm_forecasting.ipynb) | `phase3_lstm` | Sequence lengths of 10, 20 and 40 sessions, plus five-model forecasts on aligned dates | `reports/phase3_2/` |
| [07 · Strategy backtest](notebooks/04_backtesting/07_strategy_backtesting.ipynb) | `phase4_backtest` | Wealth, drawdown, exposure and turnover at a fixed threshold | `reports/phase4_2/` |
| [08 · Strategy robustness](notebooks/04_backtesting/08_strategy_robustness.ipynb) | `phase4_robustness` | Threshold selection, volatility regimes, sensitivity to fees and paired excess returns | `reports/phase4_4/` |
| [09 · Final holdout](notebooks/05_final_holdout/09_final_holdout.ipynb) | Run the notebook directly | Frozen XGBoost against the training prior and Buy & Hold, with net performance and paired intervals | `reports/phase5/` |

Each neural module accepts `--quick` for a reduced training run once its inputs exist. Quick and full runs write to the same report directories. After a quick run, rerun the full model and downstream steps before interpreting metrics.

## Research design

- **Signal and target.** Features use data through the close of session $t$. The label is one when the adjusted opening price rises from session $t+1$ to session $t+2$. Adjusted open is calculated as open times adjusted close divided by close.
- **Time splits.** Training uses 2005–2018, development validation uses 2019–2021, and the final holdout starts in 2022. The requested snapshot ends on 30 October 2026. Rows whose label reaches the next split are purged. Scalers are fitted on each training fold. LSTM windows end on the signal date and can include earlier sessions across a split boundary.
- **Forecast selection.** Expanding-window cross-validation ranks configurations by log loss. Brier score and ROC-AUC provide additional diagnostics. The neural search averages across folds and seeds and uses seed 42 for final refits.
- **Trading rule.** The fixed-rule comparison takes a long position when the probability exceeds 0.55 and holds cash otherwise. By default each change in position costs 5 basis points per side. Cash earns zero, positions start in cash, and the final position is marked to market. The backtest also records fees for the benchmark's initial purchase.
- **Strategy sensitivity.** The final step uses XGBoost by default and compares thresholds of 0.50, 0.55 and 0.60 on development net Sharpe with exposure constrained to 10–95%. A fallback to 0.50 is flagged when no candidate qualifies. Other diagnostics use a training-period volatility cutoff, costs of 0, 5 and 10 basis points, and a paired moving-block bootstrap of mean daily excess returns.

Model and threshold comparisons reuse the development period. Reported intervals are conditional on those choices. Yahoo's retrospective adjustments can change with corporate actions, so the adjusted-open backtest estimates historical returns with hypothetical fees. Actual fills, dividend cash flows and terminal liquidation would require different accounting.

## Where to edit

| Location | Responsibility |
|---|---|
| `src/quant_trading/data.py`, `features.py`, `targets.py`, `splits.py` | Market data, causal feature construction, labels and split boundaries |
| `ml.py`, `xgboost_model.py`, `mlp_model.py`, `lstm_model.py`, `neural_utils.py` in the same package | Estimators, scaling, sequence construction and training |
| `robustness.py`, `backtest.py`, `performance.py`, `strategy_research.py` in the same package | Forecast diagnostics, portfolio accounting and strategy comparisons |
| `pipeline.py` and `phase*.py` in the same package | Read inputs and write the tables, forecasts and metadata used by later steps |
| `notebooks/` | Edit the nine numbered notebooks directly |
| `scripts/run_research.py` | Run the experiments in dependency order on Windows, Linux or macOS |
| `tests/` | Synthetic-data checks for chronology, isolation of the holdout, models and backtest accounting |
| `data/` and `reports/` | Local outputs generated by the experiments and excluded from version control |

## Executed research snapshot

The eight development notebooks contain their saved tables and figures. The experiments used the full default configuration, including 48 MLP and 72 LSTM cross-validation fits at 60 epochs per fit. The automated test suite passed all 32 tests. See `reports/phase3_2/validation_metrics.csv` for the aligned forecast comparison and `reports/phase4_2/performance_metrics.csv` for the fixed-threshold backtests. These results cover development validation in 2019–2021.

The private run bundle includes the cached SPY snapshot and its checksum so the notebooks can be rerun against the same data. The repository's ignore rules exclude generated data and reports from Git.

Runtime packages included `pandas 2.2.3`, `numpy 2.3.5`, `scipy 1.17.0`, `scikit-learn 1.6.1`, `xgboost 2.1.4`, `torch 2.6.0+cpu`, `statsmodels 0.15.0`, `yfinance 0.2.66`, `exchange-calendars 4.13.3`, `pyarrow 24.0.0`.

## Independent final holdout

Over the 1,210 held-out open-to-open returns, XGBoost had a log loss of 0.69514 and ROC-AUC of 0.48333. The constant training prior had a lower log loss of 0.69049. The strategy grew $10,000 to $15,007.97 after fees, against $16,918.98 for Buy & Hold. It held SPY on 85.95% of the return intervals. Maximum drawdown magnitude was 19.23%, versus 26.32% for Buy & Hold.

The mean daily net return difference was −1.08 basis points. The prespecified 95% moving-block bootstrap interval was [−3.59, +1.78] basis points per day. The realized shortfall is clear, while the interval includes zero. The experiment supplies no evidence of economic outperformance or an improvement in probabilistic forecasts. The drawdown reduction occurred with lower average market exposure. Future strategy changes require fresh data for an independent final assessment.

Notebook 09 replays this evaluation from the archived files. After installing the dependencies, open it in Jupyter and run all cells. It checks the input hashes against `reports/phase5/frozen_protocol.json`. The fitted 2005–2018 XGBoost model, the strict 0.50 long/cash threshold selected on development validation, and the 5 basis point fee per change of position were fixed before evaluation. Both portfolios use zero yield on cash. The first entry was 4 January 2022 and the final valuation was 30 October 2026.
