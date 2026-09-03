import pandas as pd

from src.strategies import (
    moving_average_strategy,
    rsi_strategy,
    macd_strategy
)

from src.ml_model import ml_strategy
from src.backtester import run_backtest
from src.metrics import calculate_metrics


def compare_strategies(
    df,
    initial_capital=100000,
    ml_threshold=0.55
):

    results = []
    portfolio_data = {}

    # Moving Average Strategy

    ma_df = moving_average_strategy(
        df.copy()
    )

    ma_df = run_backtest(
        ma_df,
        initial_capital
    )

    ma_metrics = calculate_metrics(
        ma_df,
        initial_capital
    )

    results.append(
        {
            "Strategy": "Moving Average",
            **ma_metrics
        }
    )

    portfolio_data[
        "Moving Average"
    ] = ma_df["Portfolio_Value"]


    # RSI Strategy

    rsi_df = rsi_strategy(
        df.copy()
    )

    rsi_df = run_backtest(
        rsi_df,
        initial_capital
    )

    rsi_metrics = calculate_metrics(
        rsi_df,
        initial_capital
    )

    results.append(
        {
            "Strategy": "RSI",
            **rsi_metrics
        }
    )

    portfolio_data[
        "RSI"
    ] = rsi_df["Portfolio_Value"]


    # MACD Strategy

    macd_df = macd_strategy(
        df.copy()
    )

    macd_df = run_backtest(
        macd_df,
        initial_capital
    )

    macd_metrics = calculate_metrics(
        macd_df,
        initial_capital
    )

    results.append(
        {
            "Strategy": "MACD",
            **macd_metrics
        }
    )

    portfolio_data[
        "MACD"
    ] = macd_df["Portfolio_Value"]


    # ML Strategy

    ml_df, _, _ = ml_strategy(
        df.copy(),
        probability_threshold=ml_threshold
    )

    ml_df = run_backtest(
        ml_df,
        initial_capital
    )

    ml_metrics = calculate_metrics(
        ml_df,
        initial_capital
    )

    results.append(
        {
            "Strategy": "ML Strategy",
            **ml_metrics
        }
    )

    portfolio_data[
        "ML Strategy"
    ] = ml_df["Portfolio_Value"]


    # Create comparison dataframe

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "Total Return (%)",
        ascending=False
    )

    portfolio_df = pd.DataFrame(
        portfolio_data
    )

    return results_df, portfolio_df