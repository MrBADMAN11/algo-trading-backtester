def run_backtest(df, initial_capital=100000):
    """
    Run the trading strategy backtest.
    """

    df = df.copy()

    # Calculate market returns
    df["Market_Return"] = (
        df["Close"]
        .pct_change()
        .fillna(0)
    )

    # Strategy returns
    df["Strategy_Return"] = (
        df["Position"]
        * df["Market_Return"]
    )

    df["Strategy_Return"] = (
        df["Strategy_Return"]
        .fillna(0)
    )

    # Strategy portfolio value
    df["Portfolio_Value"] = (
        initial_capital
        * (1 + df["Strategy_Return"]).cumprod()
    )

    # Buy and Hold portfolio
    df["Buy_Hold_Value"] = (
        initial_capital
        * (1 + df["Market_Return"]).cumprod()
    )

    return df