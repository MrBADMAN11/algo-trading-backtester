import yfinance as yf
import pandas as pd


def load_stock_data(ticker, start_date, end_date):
    """
    Download historical stock data from Yahoo Finance.
    """

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError(
            f"No data found for ticker: {ticker}"
        )

    # Handle MultiIndex columns returned by yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.dropna()