import numpy as np


def calculate_metrics(df, initial_capital=100000):
    """
    Calculate performance metrics.
    """

    final_value = df["Portfolio_Value"].iloc[-1]

    total_return = (
        (final_value - initial_capital)
        / initial_capital
    ) * 100

    buy_hold_final = (
        df["Buy_Hold_Value"].iloc[-1]
    )

    buy_hold_return = (
        (buy_hold_final - initial_capital)
        / initial_capital
    ) * 100

    # Maximum drawdown
    rolling_max = (
        df["Portfolio_Value"]
        .cummax()
    )

    drawdown = (
        df["Portfolio_Value"]
        - rolling_max
    ) / rolling_max

    max_drawdown = drawdown.min() * 100

    # Sharpe Ratio
    returns = df["Strategy_Return"]

    if returns.std() != 0:
        sharpe_ratio = (
            np.sqrt(252)
            * returns.mean()
            / returns.std()
        )
    else:
        sharpe_ratio = 0

    # Count position changes
    trades = (
        df["Position"]
        .diff()
        .fillna(0)
        .ne(0)
        .sum()
    )

    return {
        "Final Portfolio Value": round(
            float(final_value), 2
        ),
        "Total Return (%)": round(
            float(total_return), 2
        ),
        "Buy & Hold Return (%)": round(
            float(buy_hold_return), 2
        ),
        "Max Drawdown (%)": round(
            float(max_drawdown), 2
        ),
        "Sharpe Ratio": round(
            float(sharpe_ratio), 2
        ),
        "Number of Trades": int(trades)
    }