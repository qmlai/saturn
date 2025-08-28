from sklearn.model_selection import TimeSeriesSplit
from collections import defaultdict

import argparse
import holidays
import pandas as pd

from saturn import (ClassifierModel, DataPreprocessor, 
                    store_fold_data, make_trading_report, 
                    print_trading_report, plot_cummulative_pnl)

def main(data_path: str,
         use_time_features=True,
         use_lag_features=True,
         use_periodic_features=True,
         use_rolling_features=True,
         model_type="lightgbm",
         model_params=None,
         callback=store_fold_data):
    
    raw_data = pd.read_csv(data_path)
    raw_data["date"] = pd.to_datetime(raw_data["date"], utc=True)
    
    de_holidays = holidays.CountryHoliday('DE', years=range(raw_data["date"].dt.year.min(), 
                                                            raw_data["date"].dt.year.max()+1))

    if use_rolling_features is True:
        rolling_columns = ["solar", "wind", "load"]
        window_sizes    = [3, 6, 12, 24, 48, 96]
    else:
        rolling_columns = None
        window_sizes    = None

    if use_lag_features is True:
        lag_columns = ["solar", "wind", "load"]
        lag_sizes   = [1, 2, 3, 6, 12, 24, 48, 96]
    else:
        lag_columns = None
        lag_sizes   = None
                 
    data_prep = DataPreprocessor(raw_data, holidays=list(de_holidays.keys()), 
                                 use_time_features=use_time_features, 
                                 use_lag_features=use_lag_features, 
                                 use_periodic_features=use_periodic_features, 
                                 use_rolling_features=use_rolling_features,
                                 rolling_columns=rolling_columns,
                                 lag_columns=lag_columns,
                                 window_sizes=window_sizes,
                                 lag_sizes=lag_sizes)
    
    X, y, z = data_prep.preprocess()

    results_log = defaultdict(list)

    ts_split = TimeSeriesSplit(n_splits=365, test_size=96)

    spread_model = ClassifierModel(X, y, z, model_type, model_params, 
                                   ts_split, callback, results_log)
    spread_model.train()

    df_out = pd.DataFrame(results_log)
    
    # Classifier report and plots
    spread_model.make_report()
    spread_model.plot_roc_curve()
    spread_model.plot_feature_importances()

    # Trading report and plots
    report = make_trading_report(df_out)
    print_trading_report(df_out, report)
    plot_cummulative_pnl(df_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_path", type=str, default="data.csv")
    parser.add_argument("--model_type", type=str, choices=["xgboost", "lightgbm", "catboost"], default="lightgbm")
    
    args = parser.parse_args()

    data_path = args.data_path
    model_type = args.model_type

    if model_type == "xgboost":
       model_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "lambda": 0.8,
            "alpha": 0.4,
            "max_depth": 10,
            "max_delta_step": 1,
            "n_jobs": 4
        }
    elif model_type == "lightgbm":
        model_params = {
            "objective": "binary",
            "metric": "auc",
            "lambda_l1": 0.8,
            "lambda_l2": 0.4,
            "max_depth": 10,
            "n_jobs": 4,
            "verbosity": -1
        }
    elif model_type == "catboost":  
        model_params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "l2_leaf_reg": 3,
            "depth": 10,
            "boosting_type": "Plain",
            "leaf_estimation_iterations": 1,
            "thread_count": 4,
            "verbose": False
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    main(data_path=data_path, model_type=model_type, model_params=model_params)


