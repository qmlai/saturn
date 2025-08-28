from tqdm import tqdm
from collections import defaultdict
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from collections.abc import Callable

from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                            recall_score, roc_auc_score, confusion_matrix, 
                            log_loss, roc_curve)


class ClassifierModel:
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, z: pd.DataFrame,
                  model: str, model_params: dict, split: TimeSeriesSplit, 
                  fold_callback: Callable, collector: defaultdict):
        self.X = X
        self.y = y
        self.z = z

        self.model      = self.instantiate_model(model, model_params)
        self.model_name = model
        
        self.split      = split

        self.callback  = fold_callback
        self.collector = collector

    def instantiate_model(self, model, model_params):
        if model == "xgboost":
            import xgboost as xgb
            return xgb.XGBClassifier(**model_params)
        elif model == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMClassifier(**model_params)
        elif model == "catboost":   
            import catboost as cb
            return cb.CatBoostClassifier(**model_params)
        else:
            raise ValueError("Unsupported model type. Currently 'xgboost', 'lightgbm' and 'catboost' are supported.")
        
    def train(self, show_progress=True):
        if show_progress is True:
            split = tqdm(
                self.split.split(self.X),
                desc="Training splits",
                total=self.split.n_splits,
            )
        else:
            split = self.split(self.X)

        # Train and evaluation loop for the model
        for train_index, test_index in split:
            # Split the data
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            z_test = self.z.iloc[test_index]

            # Fit the model
            self.model.fit(X_train, y_train)

            # Evaluate the model
            self.callback(self.collector, self.model, X_test, y_test, z_test)

        return None

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)
      
    def make_report(self, print_report=True, save_path=None):
        y_pred = self.collector["predictions"]
        y_true = self.collector["actuals"]
        y_prob = self.collector["probabilities"]
        
        accuracy    = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1          = f1_score(y_true=y_true, y_pred=y_pred)
        precision   = precision_score(y_true=y_true, y_pred=y_pred)
        recall      = recall_score(y_true=y_true, y_pred=y_pred)
        logloss     = log_loss(y_true=y_true, y_pred=y_prob)
        auc         = roc_auc_score(y_true=y_true, y_score=y_prob)
        conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)

        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall, 
            "F1": f1, 
            "logloss": logloss,
            "AUC score": auc
        }

        report = f"Classification Report for {self.model_name} model:\n"
        for name, value in metrics.items():
            report += f"{name}: {value:.2f}\n"

        report += f"confusion_matrix: {conf_matrix}\n" 
        
        if print_report:
            print(report)

        if save_path is None:
            save_path = "classification_report.txt"

        with open(save_path, "w") as f:
            f.write(report)

    def plot_roc_curve(self):
        y_true = self.collector["actuals"]
        y_prob = self.collector["probabilities"]

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)

        fig = go.Figure()

        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, 
            y=tpr,
            mode="lines",
            name=f"ROC curve (AUC = {auc_score:.2f})",
            line=dict(color="darkorange", width=3)
        ))

        # Diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Guess",
            line=dict(color="navy", width=2, dash="dash")
        ))

        # Layout
        fig.update_layout(
            title="Receiver Operating Characteristic (ROC) Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            template="plotly_white",
            width=700,
            height=500,
            legend=dict(
                x=0.6, y=0.05,
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="LightGray",
                borderwidth=1
            )
        )

        fig.show()

    def plot_feature_importances(self, top_n=None, sort=True):
        if not hasattr(self.model, "feature_importances_"):
            raise AttributeError(f"Model type '{self.model_type}' does not expose feature_importances_.")

        importances = self.model.feature_importances_
        names = np.array(self.X.columns)

        if sort:
            idx = np.argsort(importances)[::-1]
            importances = importances[idx]
            names = names[idx]

        if top_n is not None:
            importances = importances[:top_n]
            names = names[:top_n]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=names,
            y=importances,
            marker=dict(color="steelblue"),
            name="Importance"
        ))

        fig.update_layout(
            title="Feature Importances",
            xaxis=dict(title="Features", tickangle=45),
            yaxis=dict(title="Importance"),
            bargap=0.4,
            template="plotly_white"
        )

        fig.show()