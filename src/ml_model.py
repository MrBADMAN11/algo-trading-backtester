import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def prepare_ml_data(df):
    """
    Prepare stock data for machine learning.
    """

    df = df.copy()

    df["Tomorrow_Close"] = df["Close"].shift(-1)

    df["Target"] = (
        df["Tomorrow_Close"] > df["Close"]
    ).astype(int)

    df = df.dropna()

    return df


def train_ml_model(df):
    """
    Train a Random Forest model using historical data.
    """

    df = prepare_ml_data(df)

    features = [
        "SMA_20",
        "SMA_50",
        "RSI",
        "MACD",
        "MACD_Signal",
        "Volatility"
    ]

    X = df[features]
    y = df["Target"]

    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return model, accuracy, features, df


def predict_next_day(model, df, features):
    """
    Predict whether the next trading day
    will move UP or DOWN.
    """

    latest_data = df[features].iloc[[-1]]

    prediction = model.predict(
        latest_data
    )[0]

    probabilities = model.predict_proba(
        latest_data
    )[0]

    probability_down = probabilities[0] * 100
    probability_up = probabilities[1] * 100

    return (
        prediction,
        probability_up,
        probability_down
    )


def ml_strategy(df, probability_threshold=0.55):
    """
    Create ML trading signals.

    Uses a chronological train/test split so that
    future data is not used to generate test signals.

    Position:
    1  = Long
    -1 = Short
    0  = Hold previous / no position
    """

    df = prepare_ml_data(df)

    features = [
        "SMA_20",
        "SMA_50",
        "RSI",
        "MACD",
        "MACD_Signal",
        "Volatility"
    ]

    split_index = int(len(df) * 0.8)

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:].copy()

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(
        train_df[features],
        train_df["Target"]
    )

    probabilities = model.predict_proba(
        test_df[features]
    )

    test_df["Probability_Down"] = probabilities[:, 0]
    test_df["Probability_Up"] = probabilities[:, 1]

    test_df["Signal"] = 0

    test_df.loc[
        test_df["Probability_Up"] >= probability_threshold,
        "Signal"
    ] = 1

    test_df.loc[
        test_df["Probability_Down"] >= probability_threshold,
        "Signal"
    ] = -1

    test_df["Position"] = (
        test_df["Signal"]
        .replace(0, np.nan)
        .ffill()
        .fillna(0)
        .shift(1)
        .fillna(0)
    )

    return test_df, model, features