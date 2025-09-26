import pandas as pd
import yfinance as yf
from typing import List, Tuple, Optional

def download_ohlcv(tickers: List[str],
                   start: str,
                   end: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (open, high, low, close, volume) with columns=multiindex [Ticker], rows=Date
    All prices in each ticker's trading currency.
    """
    data = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False, group_by='ticker', threads=True)
    # Handle both single and multi ticker shapes
    if isinstance(data.columns, pd.MultiIndex):
        # align to (field, ticker) order
        opens  = pd.concat({t: data[t]["Open"]  for t in tickers if t in data.columns.levels[0]}, axis=1)
        highs  = pd.concat({t: data[t]["High"]  for t in tickers if t in data.columns.levels[0]}, axis=1)
        lows   = pd.concat({t: data[t]["Low"]   for t in tickers if t in data.columns.levels[0]}, axis=1)
        closes = pd.concat({t: data[t]["Close"] for t in tickers if t in data.columns.levels[0]}, axis=1)
        vols   = pd.concat({t: data[t]["Volume"]for t in tickers if t in data.columns.levels[0]}, axis=1)
    else:
        # single ticker: columns are fields
        opens = data[["Open"]].copy()
        highs = data[["High"]].copy()
        lows  = data[["Low"]].copy()
        closes= data[["Close"]].copy()
        vols  = data[["Volume"]].copy()
        opens.columns = highs.columns = lows.columns = closes.columns = vols.columns = [tickers[0]]
    return opens, highs, lows, closes, vols

def download_close(tickers: List[str], start: str, end: Optional[str] = None) -> pd.DataFrame:
    return yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)["Close"]
