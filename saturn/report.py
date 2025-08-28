import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Callback to store fold data, that is then used for trading report
def store_fold_data(results_log, model, X_test, y_test, z_test, *, prob_threshold=0.55):
    # Step 1: Predictions
    pred_class = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Step 2: Trade filter
    long_signal  = np.where(pred_proba > prob_threshold, 1, 0)
    short_signal = np.where(pred_proba < (1 - prob_threshold), -1, 0)
    
    # Step 3: Spread & returns
    spread = z_test["spread"].values
    
    # Step 4: PnL
    long_pnl  = long_signal * spread
    short_pnl = short_signal * spread
    pnl       = long_pnl + short_pnl

    # Store in results_log
    results_log["date"].extend(z_test["date"].values)
    results_log["hour"].extend(X_test["hour"])
    results_log["year"].extend(X_test["year"])
    results_log["month"].extend(X_test["month"])
    results_log["quarter"].extend(X_test["quarter"])
    results_log["day_of_week"].extend(X_test["day_of_week"])
    results_log["day_of_month"].extend(X_test["day_of_month"])
    results_log["day_of_year"].extend(X_test["day_of_year"])
    results_log["week_of_year"].extend(X_test["week_of_year"])
    results_log["is_weekend"].extend(X_test["is_weekend"])
    results_log["is_holiday"].extend(X_test["is_holiday"])
    
    results_log["predictions"].extend(pred_class)
    results_log["actuals"].extend(y_test.values)
    results_log["probabilities"].extend(pred_proba)

    results_log["long_pnl"].extend(long_pnl)   
    results_log["short_pnl"].extend(short_pnl) 
    results_log["pnl"].extend(pnl)

    return None

def make_trading_report(trading_stats: pd.DataFrame) -> dict:
    report = {}

    # Number of trades
    num_long  = np.count_nonzero(trading_stats["long_pnl"])
    num_short = np.count_nonzero(trading_stats["short_pnl"])
    num_total = num_long + num_short

    # Total PnL
    total_long  = trading_stats["long_pnl"].sum()
    total_short = trading_stats["short_pnl"].sum()
    total_pnl   = trading_stats["pnl"].sum()

    # Average PnL
    avg_long  = total_long / num_long if num_long > 0 else 0
    avg_short = total_short / num_short if num_short > 0 else 0
    avg_total = total_pnl / num_total if num_total > 0 else 0

    # Win rate
    win_long  = (trading_stats["long_pnl"] > 0).mean() if num_long > 0 else 0
    win_short = (trading_stats["short_pnl"] > 0).mean() if num_short > 0 else 0
    win_total = (trading_stats["pnl"] > 0).mean() if num_total > 0 else 0

    # Sharpe ratio
    # TODO: this is a simplified version, consider using daily returns
    pnl = trading_stats["pnl"]
    sharpe = pnl.mean() / pnl.std() * np.sqrt(365) if pnl.std() != 0 else np.nan

    # Max drawdown
    cum_pnl = pnl.cumsum()
    max_dd = (cum_pnl.cummax() - cum_pnl).max()

    report.update({
        "num_trades": num_total,
        "num_long_trades": num_long,
        "num_short_trades": num_short,

        "total_pnl": total_pnl,
        "total_long_pnl": total_long,
        "total_short_pnl": total_short,

        "avg_pnl": avg_total,
        "avg_long_pnl": avg_long,
        "avg_short_pnl": avg_short,

        "win_rate": win_total,
        "win_rate_long": win_long,
        "win_rate_short": win_short,

        "sharpe": sharpe,
        "max_drawdown": max_dd,
    })
    
    return report

def print_trading_report(df_out: pd.DataFrame, report: dict):
    # Global summary
    print("="*40)
    print("         Trading Performance Report")
    print("="*40)
    print(f" Total trades:       {report['num_trades']:>6}")
    print(f"   Long trades:      {report['num_long_trades']:>6}")
    print(f"   Short trades:     {report['num_short_trades']:>6}")
    print("-"*40)
    print(f" Total PnL:        {report['total_pnl']:>10.2f}")
    print(f"   Long PnL:       {report['total_long_pnl']:>10.2f}")
    print(f"   Short PnL:      {report['total_short_pnl']:>10.2f}")
    print("-"*40)
    print(f" Avg PnL/trade:    {report['avg_pnl']:>10.2f}")
    print(f"   Long Avg:       {report['avg_long_pnl']:>10.2f}")
    print(f"   Short Avg:      {report['avg_short_pnl']:>10.2f}")
    print("-"*40)
    print(f" Win rate total:   {report['win_rate']*100:>9.2f}%")
    print(f"   Long win rate:  {report['win_rate_long']*100:>9.2f}%")
    print(f"   Short win rate: {report['win_rate_short']*100:>9.2f}%")
    print("-"*40)
    print(f" Sharpe ratio:     {report['sharpe']:>10.2f}")
    print(f" Max drawdown:     {report['max_drawdown']:>10.2f}")
    print("="*40)

    # Grouped summaries
    groupings = {
        "hour": "By Hour of Day",
        "day_of_week": "By Day of Week",
        "day_of_month": "By Day of Month",
        "month": "By Month",
        "quarter": "By Quarter",
        "is_weekend": "Weekend vs Weekday",
        "is_holiday": "Holiday vs Non-Holiday",
    }

    for col, title in groupings.items():
        print("="*40)
        print(f" {title} ")
        print("="*40)
        grouped = df_out.groupby(col)[["long_pnl","short_pnl","pnl"]].sum()
        print(grouped.to_string(float_format=lambda x: f"{x:,.2f}"))
        print("="*40)

def plot_cummulative_pnl(df_out: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_out["date"],
        y=df_out["long_pnl"].cumsum(),
        mode="lines",
        name="Long cumulative PnL",
        line=dict(color="blue", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df_out["date"],
        y=df_out["short_pnl"].cumsum(),
        mode="lines",
        name="Short cumulative PnL",
        line=dict(color="Red", width=2)
    ))

    fig.update_layout(
        title="Cumulative Long and Short PnL Over Time",
        xaxis_title="Date",
        yaxis_title="Cumulative PnL",
        template="plotly_white"
    )

    fig.show()