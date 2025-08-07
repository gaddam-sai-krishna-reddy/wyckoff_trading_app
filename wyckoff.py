# wyckoff.py

import yfinance as yf
import pandas as pd
import numpy as np
import logging
import os

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

def run_backtest(ticker: str,
                 start_date: str = "2020-01-01",
                 end_date: str   = "2025-06-30"
                ) -> tuple[pd.DataFrame, dict]:
    """
    Runs Richard Wyckoff backtest on the given ticker and returns:
      - equity_df: DataFrame with daily cumulative PnL columns
      - metrics: dict of aggregate results
    """
    logger.info(f"Starting Wyckoff backtest for {ticker} from {start_date} to {end_date}")
    
    try:
        # 1) Load data
        logger.debug(f"Downloading data for {ticker}")
        stock = yf.download(ticker, start=start_date, end=end_date)
        
        if stock.empty:
            logger.error(f"No data found for {ticker}")
            raise ValueError(f"No data available for {ticker}")
        
        logger.info(f"Downloaded {len(stock)} data points for {ticker}")
        
        # if yfinance returns a single‑ticker DataFrame, no need for xs; else adjust
        try:
            df = stock.xs(ticker, axis=1, level='Ticker')
            logger.debug("Using multi-level DataFrame structure")
        except Exception:
            df = stock.copy()
            logger.debug("Using single-level DataFrame structure")

        # 2) Compute 40‑day range
        window = 40
        logger.debug(f"Computing {window}-day rolling range")
        df['Range_High'] = df['Close'].rolling(window).max()
        df['Range_Low']  = df['Close'].rolling(window).min()

        # 3) Phase C (Spring) detection
        logger.debug("Detecting Spring patterns")
        df['Spring'] = (
            (df['Low']   < df['Range_Low'].shift(1)) &
            (df['Close'] > df['Range_Low'].shift(1)) &
            (df['Volume'] > df['Volume'].rolling(window).mean())
        )
        spring_count = df['Spring'].sum()
        logger.info(f"Detected {spring_count} Spring patterns")

        # 4) Phase D (Breakout) detection
        logger.debug("Detecting Breakout patterns")
        df['Breakout'] = (
            (df['Close'] > df['Range_High'].shift(1)) &
            (df['Volume'] > df['Volume'].rolling(window).mean())
        )
        breakout_count = df['Breakout'].sum()
        logger.info(f"Detected {breakout_count} Breakout patterns")

        # 5) Build signals
        logger.debug("Building trading signals")
        df['Signal'] = 0
        df.loc[df['Spring'],   'Signal'] = 1
        df.loc[df['Breakout'], 'Signal'] = 1

        # exit when price falls back below resistance
        df['Exit'] = (
            (df['Close'] < df['Range_High']) &
            (df['Close'].shift(1) > df['Range_High'])
        )
        df.loc[df['Exit'], 'Signal'] = -1
        
        total_signals = (df['Signal'] != 0).sum()
        logger.info(f"Generated {total_signals} total trading signals")

        # 6) Position (1 if long, 0 if flat)
        logger.debug("Computing position vector")
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
        logger.debug("Computing returns and equity curves")
        df['Buy-and-Hold Return'] = df['Close'].pct_change()
        df['Strategy Return']   = df['Buy-and-Hold Return'] * df['Position']
        df['Cumulative_PnL_Strategy'] = df['Strategy Return'].cumsum()
        df['Cumulative_PnL_BuyHold'] = df['Buy-and-Hold Return'].cumsum()

        equity_df = df[['Cumulative_PnL_Strategy', 'Cumulative_PnL_BuyHold']].dropna()
        logger.info(f"Computed equity curves with {len(equity_df)} data points")

        # 8) Aggregate metrics
        strat_final = equity_df['Cumulative_PnL_Strategy'].iloc[-1] - 1
        bh_final    = equity_df['Cumulative_PnL_BuyHold'].iloc[-1]   - 1

        metrics = {
            "Total Return (Wyckoff)": f"{strat_final:.2%}",
            "Total Return (Buy/Hold)": f"{bh_final:.2%}",
        }
        
        logger.info(f"Backtest completed successfully")
        logger.info(f"Wyckoff Strategy Return: {strat_final:.2%}")
        logger.info(f"Buy & Hold Return: {bh_final:.2%}")
        
        return equity_df, metrics
        
    except Exception as e:
        logger.error(f"Error in backtest for {ticker}: {str(e)}")
        raise


def get_available_tickers() -> list[str]:
    """
    Return a list of tickers to populate the dropdown.
    """
    logger.debug("Getting available tickers")
    tickers = ["GS", "AAPL", "GOOGL", "AMZN", "NVDA"]
    logger.info(f"Available tickers: {tickers}")
    return tickers
