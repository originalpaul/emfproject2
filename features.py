import pandas as pd
import numpy as np

def _safe_div(a, b):
    out = a / b
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    # shit keeps exploding otherwise
    out = out.clip(-1e6, 1e6)
    return out

def technical_features(close: pd.DataFrame,
                       volume: pd.DataFrame) -> pd.DataFrame:
    """
    Build simple, fast features per ticker:
      - 5d & 20d momentum
      - 10d SMA / 50d SMA
      - 14d RSI
      - 20d realized volatility
      - 20d volume z-score
    Returns a long-form DataFrame with columns:
      ['date','ticker', <features...>]
    """
    feats = []
    rets = close.pct_change(fill_method=None)

    sma10 = close.rolling(10).mean()
    sma50 = close.rolling(50).mean()
    mom5  = close.pct_change(5, fill_method=None)
    mom20 = close.pct_change(20, fill_method=None)

    # RSI(14) - more robust calculation
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = _safe_div(gain, loss)
    # Add small epsilon to prevent division by zero
    rsi = 100 - (100 / (1 + rs + 1e-8))

    # 20d volatility (daily)
    vol20 = rets.rolling(20).std()

    # volume z-score (20d) - more robust calculation
    vol_mu = volume.rolling(20).mean()
    vol_sd = volume.rolling(20).std()
    # Add small epsilon to prevent division by zero
    vol_z = _safe_div((volume - vol_mu), vol_sd + 1e-8)

    # Additional features for better prediction
    # Price momentum features
    mom1 = close.pct_change(1, fill_method=None)
    mom3 = close.pct_change(3, fill_method=None)
    mom10 = close.pct_change(10, fill_method=None)
    
    # Volatility features
    vol5 = rets.rolling(5).std()
    vol10 = rets.rolling(10).std()
    
    # Price position features
    high_20 = close.rolling(20).max()
    low_20 = close.rolling(20).min()
    price_position = _safe_div((close - low_20), (high_20 - low_20))
    
    # Volume momentum
    vol_mom = volume.pct_change(5, fill_method=None)
    
    # ratios
    sma_ratio = _safe_div(sma10, sma50)

    for t in close.columns:
        df = pd.DataFrame({
            "date": close.index,
            "ticker": t,
            "mom1": mom1[t].values,
            "mom3": mom3[t].values,
            "mom5": mom5[t].values,
            "mom10": mom10[t].values,
            "mom20": mom20[t].values,
            "sma_ratio": sma_ratio[t].values,
            "rsi14": rsi[t].values,
            "vol5": vol5[t].values,
            "vol10": vol10[t].values,
            "vol20": vol20[t].values,
            "volz20": vol_z[t].values,
            "vol_mom": vol_mom[t].values,
            "price_position": price_position[t].values
        })
        feats.append(df)

    feat_df = pd.concat(feats, axis=0).sort_values(["date","ticker"])
    return feat_df

def forward_return(close: pd.DataFrame, horizon_days: int = 5) -> pd.DataFrame:
    """
    Compute future return over next 'horizon_days' using close-to-close pct change.
    Align so that y at date t is return from t+1 ... t+horizon.
    """
    fwd = close.shift(-1).pct_change(horizon_days-1, fill_method=None)  # (C_{t+h}/C_{t+1}) - 1
    # Align index to today's date for label; drop last horizon window where y is NaN
    fwd.index.name = "date"
    y_long = fwd.stack().rename("fwd_ret").reset_index()
    y_long.columns = ["date","ticker","fwd_ret"]
    return y_long
