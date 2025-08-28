import pandas as pd
import numpy as np
import datetime

class DataPreprocessor:
    def __init__(self, df: pd.DataFrame, 
                 holidays: list[datetime.date], 
                 use_time_features: bool = True, 
                 use_periodic_features: bool = True, 
                 use_rolling_features: bool = False, 
                 use_lag_features: bool = False, 
                 date_col: str = "date",
                 spread_col: str = "spread",
                 target_col: str = "target",
                 drop_cols: list[str] = ["date", "spread", "target"],
                 rolling_columns: list[str] | None = None,
                 lag_columns: list[str] | None = None, 
                 window_sizes: list[int] | None = None,
                 lag_sizes: list[int] | None = None,
                 target_type: str = "binary"):
        
        self.data_raw = df
        self.data     = df.copy()

        self.holidays = holidays

        self.use_time_features     = use_time_features
        self.use_periodic_features = use_periodic_features
        self.use_rolling_features  = use_rolling_features 
        self.use_lag_features      = use_lag_features 

        self.date_col   = date_col
        self.spread_col = spread_col
        self.target_col = target_col
        self.drop_cols  = drop_cols

        self.rolling_columns = rolling_columns
        self.lag_columns     = lag_columns

        self.window_sizes = window_sizes
        self.lag_sizes    = lag_sizes
        self.target_type  = target_type

    def reset_data(self):
        self.data = self.data_raw.copy()

    def add_time_features(self):
        self.data["hour"]         = self.data[self.date_col].dt.hour
        self.data["year"]         = self.data[self.date_col].dt.year
        self.data["month"]        = self.data[self.date_col].dt.month
        self.data["quarter"]      = self.data[self.date_col].dt.quarter
        self.data["day_of_week"]  = self.data[self.date_col].dt.dayofweek
        self.data["day_of_month"] = self.data[self.date_col].dt.day
        self.data["day_of_year"]  = self.data[self.date_col].dt.dayofyear
        self.data["week_of_year"] = self.data[self.date_col].dt.isocalendar().week
        self.data["is_weekend"]   = self.data[self.date_col].dt.dayofweek >= 5
        self.data["is_holiday"]   = self.data[self.date_col].dt.date.isin(self.holidays)

        return None
    
    def add_periodic_time_features(self):
        self.data["sin_hour"]    = np.sin(2 * np.pi * self.data["hour"] / 24)
        self.data["cos_hour"]    = np.cos(2 * np.pi * self.data["hour"] / 24)
        self.data["sin_day"]     = np.sin(2 * np.pi * self.data["day_of_year"] / 365)
        self.data["cos_day"]     = np.cos(2 * np.pi * self.data["day_of_year"] / 365)
        self.data["sin_month"]   = np.sin(2 * np.pi * self.data["month"] / 12)
        self.data["cos_month"]   = np.cos(2 * np.pi * self.data["month"] / 12)
        self.data["sin_quarter"] = np.sin(2 * np.pi * self.data["quarter"] / 4)
        self.data["cos_quarter"] = np.cos(2 * np.pi * self.data["quarter"] / 4)

        return None

    def add_rolling_features(self):
        rolling_dict = {}
        for col in self.rolling_columns:
            for window in self.window_sizes:
                rolling_dict[f"{col}_rolling_mean_{window}"]   = self.data[col].rolling(window).mean()
                rolling_dict[f"{col}_rolling_std_{window}"]    = self.data[col].rolling(window).std()
                rolling_dict[f"{col}_rolling_min_{window}"]    = self.data[col].rolling(window).min()
                rolling_dict[f"{col}_rolling_max_{window}"]    = self.data[col].rolling(window).max()
                rolling_dict[f"{col}_rolling_median_{window}"] = self.data[col].rolling(window).median()
            
        rolling_features = pd.DataFrame(rolling_dict, index=self.data.index)
        self.data = pd.concat([self.data, rolling_features], axis=1)

        return None

    def add_lag_features(self):
        lag_dict = {}
        for col in self.lag_columns:
            for lag in self.lag_sizes:
                lag_dict[f"{col}_lag_{lag}"] = self.data[col].shift(lag)
        
        lag_features = pd.DataFrame(lag_dict, index=self.data.index)
        self.data = pd.concat([self.data, lag_features], axis=1)
        
        return None
    
    def add_target_variable(self):
        if self.target_type == "binary":
            self.data[self.target_col] = np.where(self.data[self.spread_col] > 0, 1, 0)
        elif self.target_type == "ternary":
            self.data[self.target_col] = np.where(self.data[self.spread_col] > 0.01, 1, 
                                         np.where(self.data[self.spread_col] < -0.01, -1, 0))
        elif self.target_type == "stack":
            raise ValueError("Not implemented yet.")
        else:
            raise ValueError("Invalid target_type. target_type values can be 'binary', 'ternary', or 'stack'.")

        return None
    
    def preprocess(self):
        if self.use_time_features:
            self.add_time_features()
        if self.use_periodic_features:    
            self.add_periodic_time_features()
        if self.use_rolling_features:
            self.add_rolling_features()
        if self.use_lag_features:
            self.add_lag_features()

        self.add_target_variable()
        
        self.data.dropna(inplace=True)
        
        X = self.data[self.data.columns.difference(self.drop_cols)]
        y = self.data[self.target_col]
        z = self.data[[self.date_col , self.spread_col]]
        
        return X, y, z
