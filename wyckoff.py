# wyckoff.py

import yfinance as yf
import pandas as pd
import numpy as np

def run_backtest(ticker: str,
                 start_date: str = "2020-01-01",
                 end_date: str   = "2025-06-30"
                ) -> tuple[pd.DataFrame, dict]:
    """
    Runs Richard Wyckoff backtest on the given ticker and returns:
      - equity_df: DataFrame with daily cumulative PnL columns
      - metrics: dict of aggregate results
    """
    # 1) Load data
    stock = yf.download(ticker, start=start_date, end=end_date)
    # if yfinance returns a single‑ticker DataFrame, no need for xs; else adjust
    try:
        df = stock.xs(ticker, axis=1, level='Ticker')
    except Exception:
        df = stock.copy()

    # 2) Compute 40‑day range
    window = 40
    df['Range_High'] = df['Close'].rolling(window).max()
    df['Range_Low']  = df['Close'].rolling(window).min()

    # 3) Phase C (Spring) detection
    df['Spring'] = (
        (df['Low']   < df['Range_Low'].shift(1)) &
        (df['Close'] > df['Range_Low'].shift(1)) &
        (df['Volume'] > df['Volume'].rolling(window).mean())
    )

    # 4) Phase D (Breakout) detection
    df['Breakout'] = (
        (df['Close'] > df['Range_High'].shift(1)) &
        (df['Volume'] > df['Volume'].rolling(window).mean())
    )

    # 5) Build signals
    df['Signal'] = 0
    df.loc[df['Spring'],   'Signal'] = 1
    df.loc[df['Breakout'], 'Signal'] = 1

    # exit when price falls back below resistance
    df['Exit'] = (
        (df['Close'] < df['Range_High']) &
        (df['Close'].shift(1) > df['Range_High'])
    )
    df.loc[df['Exit'], 'Signal'] = -1

    # 6) Position (1 if long, 0 if flat)
    df['Position'] = 0
    position = 0
    for i in range(len(df)):
        sig = df['Signal'].iloc[i]
        if sig == 1:
            position = 1
        elif sig == -1:
            position = 0
        # df['Position'].iloc[i] = position
        df.iloc[i, df.columns.get_loc('Position')] = position


    # 7) Returns & equity curves
    df['Buy-and-Hold Return'] = df['Close'].pct_change()
    df['Strategy Return']   = df['Buy-and-Hold Return'] * df['Position']
    df['Cumulative_PnL_Strategy'] = df['Strategy Return'].cumsum()
    df['Cumulative_PnL_BuyHold'] = df['Buy-and-Hold Return'].cumsum()

    equity_df = df[['Cumulative_PnL_Strategy', 'Cumulative_PnL_BuyHold']].dropna()

    # 8) Aggregate metrics
    strat_final = equity_df['Cumulative_PnL_Strategy'].iloc[-1] - 1
    bh_final    = equity_df['Cumulative_PnL_BuyHold'].iloc[-1]   - 1

    metrics = {
        "Total Return (Wyckoff)": f"{strat_final:.2%}",
        "Total Return (Buy/Hold)": f"{bh_final:.2%}",
    }

    return equity_df, metrics


def get_available_tickers() -> list[str]:
    """
    Return a list of tickers to populate the dropdown.
    """
    return ["GS", "AAPL", "GOOGL", "AMZN", "NVDA"]
