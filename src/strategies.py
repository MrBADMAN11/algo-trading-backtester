import numpy as np


def moving_average_strategy(df):
    """
    Moving Average Crossover Strategy.
    """

    df = df.copy()

    df["Signal"] = 0

    # Bullish
    df.loc[
        df["SMA_20"] > df["SMA_50"],
        "Signal"
    ] = 1

    # Bearish
    df.loc[
        df["SMA_20"] < df["SMA_50"],
        "Signal"
    ] = -1

    # Shift one period to avoid look-ahead bias
    df["Position"] = (
        df["Signal"]
        .shift(1)
        .fillna(0)
    )

    return df


def rsi_strategy(df):
    """
    RSI Trading Strategy.
    """

    df = df.copy()

    df["Signal"] = 0

    # Buy when oversold
    df.loc[
        df["RSI"] < 30,
        "Signal"
    ] = 1

    # Sell when overbought
    df.loc[
        df["RSI"] > 70,
        "Signal"
    ] = -1

    # Hold previous position until signal changes
    df["Position"] = (
        df["Signal"]
        .replace(0, np.nan)
        .ffill()
        .fillna(0)
        .shift(1)
        .fillna(0)
    )

    return df


def macd_strategy(df):
    """
    MACD Crossover Strategy.
    """

    df = df.copy()

    df["Signal"] = 0

    # Bullish
    df.loc[
        df["MACD"] > df["MACD_Signal"],
        "Signal"
    ] = 1

    # Bearish
    df.loc[
        df["MACD"] < df["MACD_Signal"],
        "Signal"
    ] = -1

    df["Position"] = (
        df["Signal"]
        .shift(1)
        .fillna(0)
    )

    return df