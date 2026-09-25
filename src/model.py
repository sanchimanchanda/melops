"""Phase 4: LightGBM Pairwise Classifier & Re-ranker."""

import os
import joblib
import numpy as np
import lightgbm as lgb
from typing import Dict, Any, List, Optional
try:
    from src.feature_engine import FEATURE_NAMES
except ImportError:
    from feature_engine import FEATURE_NAMES

class EntityMatchingModel:
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": 8,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 5,
            "min_child_samples": 50,
            "n_estimators": 500,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
            "is_unbalance": True
        }
        self.model = lgb.LGBMClassifier(**self.params)
        
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        if X_val is not None and y_val is not None:
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
            )
        else:
            self.model.fit(X, y)
            
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
        
    def save(self, filepath: str):
        joblib.dump(self.model, filepath)
        
    def load(self, filepath: str):
        self.model = joblib.load(filepath)
