import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import START_DATE, END_DATE, UNIVERSE, BENCH_TICKER, REB_FREQ, HOLDING_PERIOD_DAYS
from data import download_ohlcv, download_close
from features import technical_features, forward_return
from model import ReturnModel, join_features_labels
from backtest import backtest_signals, summary_stats

def main(start, end):
    print("downloading data...")
    o, h, l, c, v = download_ohlcv(UNIVERSE, start, end)
    bench_close = download_close([BENCH_TICKER], start, end)
    bench_close = bench_close[BENCH_TICKER].dropna()

    # align tickers that actually downloaded
    tickers = [t for t in UNIVERSE if t in c.columns]
    c = c[tickers].dropna(how="all")
    v = v[tickers].reindex_like(c)

    print("building features & labels...")
    feat_df = technical_features(c, v)
    y_df    = forward_return(c, horizon_days=HOLDING_PERIOD_DAYS)
    data_df = join_features_labels(feat_df, y_df).dropna()

    # simple time split: use 70% for training, 30% for testing
    data_df = data_df.sort_values("date")
    split_idx = int(len(data_df) * 0.7)
    train = data_df.iloc[:split_idx]
    test = data_df.iloc[split_idx:]

    X_train, y_train = train.drop(columns=["fwd_ret"]), train["fwd_ret"]
    X_test  , y_test  = test.drop(columns=["fwd_ret"]),  test["fwd_ret"]

    print(f"train size: {len(train):,}  test size: {len(test):,}")

    print("training model...")
    model = ReturnModel()
    model.fit(X_train, y_train)

    print("scoring daily (test period only)...")
    test = test.copy()  # Avoid SettingWithCopyWarning
    
    # Get raw predictions
    raw_scores = model.predict(X_test)
    
    # Apply signal processing to improve predictions
    # 1. Z-score normalization within each date
    test_scores = test.copy()
    test_scores["raw_score"] = raw_scores
    
    # Normalize scores by date to create relative rankings
    test_scores["score"] = test_scores.groupby("date")["raw_score"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    
    # Apply additional signal processing
    # 2. Apply momentum filter (only positive momentum)
    test_scores["score"] = test_scores["score"] * (test_scores["score"] > 0).astype(int)
    
    test["score"] = test_scores["score"]

    # convert scores to daily wide table
    daily_scores = test[["date","ticker","score"]].dropna()
    
    # Debug: show score statistics
    print(f"Score statistics:")
    print(f"  Mean: {daily_scores['score'].mean():.4f}")
    print(f"  Std: {daily_scores['score'].std():.4f}")
    print(f"  Min: {daily_scores['score'].min():.4f}")
    print(f"  Max: {daily_scores['score'].max():.4f}")
    print(f"  Positive scores: {(daily_scores['score'] > 0).sum()}/{len(daily_scores)}")
    
    equity = backtest_signals(c, bench_close, daily_scores, reb_rule=REB_FREQ)

    # align to test period (use the test data dates)
    test_start_date = test["date"].min()
    equity = equity[equity.index >= test_start_date]
    s_stats = summary_stats(equity["strategy"])
    b_stats = summary_stats(equity["benchmark"])
    print("\n== Strategy vs Benchmark (test period) ==")
    for k,v in s_stats.items():
        print(f"Strategy {k}: {v:.3f}")
    for k,v in b_stats.items():
        print(f"Benchmark {k}: {v:.3f}")

    # plot
    ax = equity.plot(title="EMFCGStrat vs MSCI China (MCHI)")
    ax.set_ylabel("Growth of $1")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default=START_DATE)
    parser.add_argument("--end", type=str, default=END_DATE)
    args = parser.parse_args()
    main(args.start, args.end)
