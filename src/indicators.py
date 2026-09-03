import ta


def add_indicators(df):
    """
    Add technical indicators to the dataframe.
    """

    df = df.copy()

    # Moving averages
    df["SMA_20"] = df["Close"].rolling(
        window=20
    ).mean()

    df["SMA_50"] = df["Close"].rolling(
        window=50
    ).mean()

    # RSI
    rsi = ta.momentum.RSIIndicator(
        close=df["Close"],
        window=14
    )

    df["RSI"] = rsi.rsi()

    # MACD
    macd = ta.trend.MACD(
        close=df["Close"],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )

    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Histogram"] = macd.macd_diff()

    # Daily returns
    df["Daily_Return"] = df["Close"].pct_change()

    # Rolling volatility
    df["Volatility"] = (
        df["Daily_Return"]
        .rolling(window=20)
        .std()
    )

    return df.dropna()