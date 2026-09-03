import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import datetime

from src.data_loader import load_stock_data
from src.indicators import add_indicators

from src.strategies import (
    moving_average_strategy,
    rsi_strategy,
    macd_strategy
)

from src.backtester import run_backtest
from src.metrics import calculate_metrics

from src.ml_model import (
    train_ml_model,
    predict_next_day,
    ml_strategy
)

from src.strategy_comparison import compare_strategies

# Page configuration

st.set_page_config(
    page_title="Algo Trading Backtester",
    page_icon="📈",
    layout="wide"
)


# Title

st.title("📈 Algorithmic Trading Backtester by Mitul Thakur")

st.write(
    "Test trading strategies using historical market data "
    "and compare them against a Buy & Hold strategy."
)


# Sidebar

st.sidebar.header("⚙️ Backtest Settings")


# Stock / Asset Selection

stock_options = {
    "Apple Inc.": "AAPL",
    "Microsoft Corporation": "MSFT",
    "NVIDIA Corporation": "NVDA",
    "Amazon": "AMZN",
    "Alphabet (Google)": "GOOGL",
    "Meta Platforms": "META",
    "Tesla": "TSLA",
    "Netflix": "NFLX",
    "AMD": "AMD",
    "Intel": "INTC",

    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "ITC": "ITC.NS",
    "Tata Motors": "TATAMOTORS.NS",

    "S&P 500 ETF": "SPY",
    "NASDAQ 100 ETF": "QQQ",
    "Dow Jones ETF": "DIA",

    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",

    "S&P 500 Index": "^GSPC",
    "NASDAQ Composite": "^IXIC",
    "NIFTY 50": "^NSEI",
    "NIFTY BANK": "^NSEBANK"
}


selected_stock = st.sidebar.selectbox(
    "Select Stock / Asset",
    list(stock_options.keys()),
    index=0
)

ticker = stock_options[selected_stock]

st.sidebar.caption(
    f"Yahoo Finance Ticker: {ticker}"
)


# Strategy Selection

strategy = st.sidebar.selectbox(
    "Trading Strategy",
    [
        "Moving Average Crossover",
        "RSI Strategy",
        "MACD Strategy",
        "ML Strategy"
    ]
)


# ML Threshold

if strategy == "ML Strategy":

    ml_threshold = st.sidebar.slider(
        "ML Confidence Threshold",
        min_value=0.50,
        max_value=0.80,
        value=0.55,
        step=0.01
    )

else:

    ml_threshold = 0.55


# Date Settings

today = datetime.date.today()

default_start = (
    today - datetime.timedelta(days=730)
)

start_date = st.sidebar.date_input(
    "Start Date",
    value=default_start
)

end_date = st.sidebar.date_input(
    "End Date",
    value=today
)


# Initial Capital

initial_capital = st.sidebar.number_input(
    "Initial Capital ($)",
    min_value=1000,
    value=100000,
    step=1000
)


# Run Button

run_button = st.sidebar.button(
    "🚀 Run Backtest",
    use_container_width=True
)


# Main Application

if run_button:

    try:

        # Validate Dates

        if start_date >= end_date:

            st.error(
                "Start date must be before end date."
            )

            st.stop()


        # Load Data

        with st.spinner("Downloading market data..."):

            df = load_stock_data(
                ticker,
                start_date,
                end_date
            )


        if df.empty:

            st.error(
                "No data found. Please check the ticker."
            )

            st.stop()


        # Add Indicators

        df = add_indicators(df)


        # Check Data

        if len(df) < 100:

            st.error(
                "Not enough historical data. "
                "Please select a larger date range."
            )

            st.stop()


        # Apply Trading Strategy

        if strategy == "Moving Average Crossover":

            df = moving_average_strategy(df)

        elif strategy == "RSI Strategy":

            df = rsi_strategy(df)

        elif strategy == "MACD Strategy":

            df = macd_strategy(df)

        elif strategy == "ML Strategy":

            with st.spinner("Training ML trading strategy..."):

                df, strategy_model, strategy_features = ml_strategy(
                    df,
                    probability_threshold=ml_threshold
                )


        # Run Backtest

        df = run_backtest(
            df,
            initial_capital
        )


        # Calculate Metrics

        metrics = calculate_metrics(
            df,
            initial_capital
        )


        # Performance Metrics

        st.subheader("📊 Performance Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Final Portfolio Value",
            f"${metrics['Final Portfolio Value']:,.2f}"
        )

        col2.metric(
            "Strategy Return",
            f"{metrics['Total Return (%)']}%"
        )

        col3.metric(
            "Buy & Hold Return",
            f"{metrics['Buy & Hold Return (%)']}%"
        )

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Maximum Drawdown",
            f"{metrics['Max Drawdown (%)']}%"
        )

        col5.metric(
            "Sharpe Ratio",
            metrics["Sharpe Ratio"]
        )

        col6.metric(
            "Number of Trades",
            metrics["Number of Trades"]
        )


        # Machine Learning Prediction

        st.divider()

        st.subheader("🤖 Machine Learning Prediction")

        st.write(
            "Random Forest model trained on technical "
            "indicators to predict whether the next "
            "trading day's closing price will move "
            "UP or DOWN."
        )

        with st.spinner(
            "Training Machine Learning model..."
        ):

            model, ml_accuracy, features, ml_df = (
                train_ml_model(df)
            )

            prediction, probability_up, probability_down = (
                predict_next_day(
                    model,
                    ml_df,
                    features
                )
            )


        ml_col1, ml_col2, ml_col3 = st.columns(3)

        if prediction == 1:

            ml_col1.metric(
                "Next Day Prediction",
                "📈 UP"
            )

        else:

            ml_col1.metric(
                "Next Day Prediction",
                "📉 DOWN"
            )

        ml_col2.metric(
            "Probability UP",
            f"{probability_up:.2f}%"
        )

        ml_col3.metric(
            "Model Test Accuracy",
            f"{ml_accuracy * 100:.2f}%"
        )

        st.caption(
            f"Probability DOWN: {probability_down:.2f}%"
        )


        # Stock Price Chart

        st.divider()

        st.subheader(
            f"📈 {selected_stock} Stock Price"
        )

        price_fig = go.Figure()

        price_fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                mode="lines",
                name="Close Price"
            )
        )


        # Find Actual Buy and Sell Transitions

        df["Trade_Action"] = 0

        df.loc[
            (df["Signal"] == 1) &
            (df["Signal"].shift(1) != 1),
            "Trade_Action"
        ] = 1

        df.loc[
            (df["Signal"] == -1) &
            (df["Signal"].shift(1) != -1),
            "Trade_Action"
        ] = -1


        buy_signals = df[
            df["Trade_Action"] == 1
        ]

        sell_signals = df[
            df["Trade_Action"] == -1
        ]


        # Buy Markers

        price_fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals["Close"],
                mode="markers",
                name="BUY",
                marker=dict(
                    symbol="triangle-up",
                    size=14
                ),
                hovertemplate=(
                    "<b>BUY</b><br>"
                    "Date: %{x}<br>"
                    "Price: %{y:.2f}"
                    "<extra></extra>"
                )
            )
        )


        # Sell Markers

        price_fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals["Close"],
                mode="markers",
                name="SELL",
                marker=dict(
                    symbol="triangle-down",
                    size=14
                ),
                hovertemplate=(
                    "<b>SELL</b><br>"
                    "Date: %{x}<br>"
                    "Price: %{y:.2f}"
                    "<extra></extra>"
                )
            )
        )


        # Moving Average Lines

        if strategy == "Moving Average Crossover":

            price_fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["SMA_20"],
                    mode="lines",
                    name="SMA 20"
                )
            )

            price_fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["SMA_50"],
                    mode="lines",
                    name="SMA 50"
                )
            )


        price_fig.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified"
        )

        st.plotly_chart(
            price_fig,
            use_container_width=True
        )


        # Portfolio Performance

        st.divider()

        st.subheader(
            "💰 Strategy Performance vs Buy & Hold"
        )

        portfolio_fig = go.Figure()

        portfolio_fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Portfolio_Value"],
                mode="lines",
                name="Strategy Portfolio"
            )
        )

        portfolio_fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Buy_Hold_Value"],
                mode="lines",
                name="Buy & Hold"
            )
        )

        portfolio_fig.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode="x unified"
        )

        st.plotly_chart(
            portfolio_fig,
            use_container_width=True
        )


        # Feature Importance

        st.divider()

        st.subheader(
            "🧠 Machine Learning Feature Importance"
        )

        importance_df = pd.DataFrame(
            {
                "Feature": features,
                "Importance": model.feature_importances_
            }
        ).sort_values(
            "Importance",
            ascending=False
        )

        st.bar_chart(
            importance_df.set_index("Feature")
        )


        # Trade History

        st.divider()

        st.subheader("📋 Trade History")

        trades = df[
            df["Trade_Action"] != 0
        ][[
            "Close",
            "Signal",
            "Trade_Action"
        ]].copy()

        trades["Action"] = trades[
            "Trade_Action"
        ].map(
            {
                1: "BUY",
                -1: "SELL"
            }
        )

        trades = trades.rename(
            columns={
                "Close": "Price"
            }
        )

        trades = trades[
            [
                "Action",
                "Price"
            ]
        ]

        st.dataframe(
            trades,
            use_container_width=True
        )


        # Trading Data

        st.divider()

        st.subheader("📊 Trading Data")

        with st.expander(
            "Click to view recent trading data"
        ):

            st.dataframe(
                df.tail(100),
                use_container_width=True
            )


    except Exception as e:

        st.error(
            f"An error occurred: {str(e)}"
        )

        st.exception(e)


# Initial Screen

else:

    st.info(
        "👈 Configure your settings in the sidebar "
        "and click Run Backtest."
    )
