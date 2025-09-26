import pandas as pd
import numpy as np
from typing import Optional
from config import TOP_K, MAX_WEIGHT, CASH_BUFFER, VOL_TARGET_ANNUAL

def weekly_endpoints(dates: pd.DatetimeIndex, rule: str = "W-FRI") -> pd.DatetimeIndex:
    # ensure we have the shit
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(dates)
    # create a Series with the shit for resampling
    series = pd.Series(index=dates, dtype=float)
    resampled = series.resample(rule).last().dropna()
    # Return the actual dates, not just the index
    return resampled.index

def _cap_weights(w: pd.Series, cap: float) -> pd.Series:
    if w.empty:
        return w
    w = w.clip(upper=cap)
    return w / max(w.sum(), 1e-12)

def backtest_signals(close: pd.DataFrame,
                     bench_close: pd.Series,
                     daily_scores: pd.DataFrame,
                     reb_rule: str = "W-FRI") -> pd.DataFrame:
    """
    daily_scores: long DF ['date','ticker','score'] where score is model's predicted 5d return
    Builds a weekly-rebalanced long-only equal-weight (capped) portfolio, compares to MCHI.
    Returns daily equity curves for strategy and benchmark.
    """
    dates = close.index
    rebal_days = weekly_endpoints(dates, reb_rule)
    
    # Fallback: if no rebalancing days, use monthly
    if len(rebal_days) == 0:
        rebal_days = weekly_endpoints(dates, "ME")  # Use 'ME' instead of deprecated 'M'
        if len(rebal_days) == 0:
            # Last resort: rebalance every 20 days
            rebal_days = dates[::20]
    
    # compute daily returns n shit
    rets = close.pct_change(fill_method=None).fillna(0.0)
    bench_rets = bench_close.pct_change(fill_method=None).fillna(0.0)

    # turn scores into selections each rebalance day
    daily_scores = daily_scores.set_index(["date","ticker"]).sort_index()

    # portfolio weights per rebalance date
    weights_by_day = {}
    print(f"Rebalancing on {len(rebal_days)} days: {rebal_days[:5]}...")  # Debug info
    
    for d in rebal_days:
        if (d, ) not in daily_scores.index.names:
            pass  # just standard handling below i hink
        try:
            day_scores = daily_scores.loc[d]
        except KeyError:
            continue
        # Enhanced stock selection with better scoring
        # Only select stocks with positive scores (momentum strategy)
        positive_scores = day_scores[day_scores["score"] > 0]
        if len(positive_scores) == 0:
            continue  # No positive signals, stay in cash
            
        picks = positive_scores.sort_values("score", ascending=False).head(TOP_K)
        if picks.empty:
            continue
        
        # Use score-weighted allocation with momentum bias
        raw_weights = picks["score"].values
        # Apply exponential scaling to emphasize top picks
        raw_weights = np.exp(raw_weights * 2)  # Scale up the differences
        raw_weights = raw_weights / raw_weights.sum()  # Normalize
        
        w = pd.Series(raw_weights, index=picks.index)
        w = _cap_weights(w, MAX_WEIGHT)
        # keep some cash buffer
        w = w * (1.0 - CASH_BUFFER)
        weights_by_day[d] = w

    # propagate weights forward until next rebalance
    tickers = close.columns
    port_ret = pd.Series(0.0, index=dates)
    current_w = pd.Series(0.0, index=tickers)

    for i, d in enumerate(dates):
        if d in weights_by_day:
            # reset to new weights at close -> next day's return applies
            current_w = pd.Series(0.0, index=tickers)
            w_new = weights_by_day[d]
            current_w.loc[w_new.index] = w_new.values
        
        # strategy daily return - ensure we have valid weights and returns
        if current_w.sum() > 0:  # Only calculate if we have positions
            port_ret[d] = (current_w * rets.loc[d].reindex(tickers).fillna(0.0)).sum()
        else:
            port_ret[d] = 0.0  # No positions = no return

    # (will add later) naive volatility targeting on the fly
    if VOL_TARGET_ANNUAL:
        # estimate rolling 60d vol and scale exposure
        ann = np.sqrt(252)
        rolling_vol = port_ret.rolling(60).std() * ann
        leverage = (VOL_TARGET_ANNUAL / (rolling_vol.replace(0.0, np.nan))).clip(upper=2.0).fillna(1.0)
        port_ret = port_ret * leverage

    strat_equity = (1 + port_ret).cumprod()
    bench_equity = (1 + bench_rets.reindex(dates).fillna(0.0)).cumprod()

    out = pd.DataFrame({
        "strategy": strat_equity,
        "benchmark": bench_equity
    }).dropna()
    return out

def summary_stats(curve: pd.Series, freq: int = 252) -> dict:
    daily_ret = curve.pct_change().dropna()
    ann = freq
    cagr = curve.iloc[-1] ** (ann / len(curve)) - 1
    vol  = daily_ret.std() * np.sqrt(freq)
    sharpe = 0.0 if vol == 0 else (daily_ret.mean() * freq) / vol
    mdd = ((curve / curve.cummax()) - 1).min()
    return {"CAGR": float(cagr), "Vol": float(vol), "Sharpe": float(sharpe), "MaxDD": float(mdd)}
