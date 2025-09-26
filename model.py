import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

FEATURE_COLS = ["mom1","mom3","mom5","mom10","mom20","sma_ratio","rsi14","vol5","vol10","vol20","volz20","vol_mom","price_position"]

class ReturnModel:
    """
    Enhanced model using GradientBoostingRegressor to predict next-5d returns.
    Optimized for better performance and reliability.
    """
    def __init__(self, random_state: int = 42):
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=random_state
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = X[FEATURE_COLS].fillna(0.0)
        y = y.fillna(0.0)
        
        # Clean data: remove infinite values and extremely large values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X = X.clip(-1e6, 1e6)  # Clip extreme values
        
        y = y.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = y.clip(-1e6, 1e6)  # Clip extreme values
        
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X[FEATURE_COLS].fillna(0.0)
        
        # Clean data: remove infinite values and extremely large values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X = X.clip(-1e6, 1e6)  # Clip extreme values
        
        return self.model.predict(X)

def join_features_labels(feat_df: pd.DataFrame, y_df: pd.DataFrame) -> pd.DataFrame:
    df = feat_df.merge(y_df, on=["date","ticker"], how="inner").dropna(subset=["fwd_ret"])
    # Fill NaN values with 0 for features (more lenient approach)
    feature_cols = [col for col in df.columns if col not in ["date", "ticker", "fwd_ret"]]
    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df

