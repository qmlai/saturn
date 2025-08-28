# ml_models

This repository provides a framework to train machine learning classifiers (LightGBM, XGBoost, CatBoost) on spread data, generate classification reports, and evaluate trading performance using backtests. It is designed for time series classification problems where features include lagged values, rolling statistics, and calendar effects (holidays, time-of-day, etc.). 

## Features
- Flexible preprocessing pipeline: add time features (hour, day-of-week, etc.), periodic encodings (sine/cosine of seasonal cycles), lagged features, rolling-window features, holiday calendar integration.  
- Model support: LightGBM, XGBoost, CatBoost.  
- Time series cross-validation using TimeSeriesSplit.  
- Reporting: classification metrics (accuracy, precision, recall, F1, AUC, confusion matrix), ROC curve plotting, and trading backtest reports (PnL, long/short performance).  

The preprocessing pipeline will create a binary classification target (`target`) automatically.

## Installation (Poetry)

1. Make sure you have Poetry installed:

```bash
curl -sSL https://install.python-poetry.org | python3 -
````

2. Install dependencies from pyproject.toml:

```bash
poetry install
```

3. Activate the virtual environment:

```bash
poetry shell
```

## Usage
Run the main script:

```bash
poetry run python main.py --data_path data.csv --model_type lightgbm
```

## Output
- Classification report: Accuracy, Precision, Recall, F1, AUC score, Confusion matrix, Log loss.
- ROC curve plotted with AUC annotation.
- Trading report: aggregated PnL by day/hour, long/short trade breakdown, Sharpe ratio metrics.

## Extending
- Modify DataPreprocessor to add/remove features.
- Adjust hyperparameters in main.py.
- Add new model wrappers in model.py.
- Extend reporting in report.py for custom trading KPIs.

## Roadmap / TODO

- [ ] Hyperparameter tuning with Optuna  
- [ ] Fetch market & system data directly from TSOs APIs  
- [ ] Add support for more models (RandomForest, Neural Nets)  
- [ ] Dockerize for reproducible training  
