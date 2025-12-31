"""Machine learning-based trading strategy."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

from .base_strategy import BaseStrategy, Signal
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class MLHybridStrategy(BaseStrategy):
    """
    Machine learning hybrid strategy using ensemble methods.

    Combines multiple ML models to predict price direction and generate signals.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ML hybrid strategy.

        Args:
            config: Strategy configuration with:
                - features: List of feature columns to use
                - lookback: Lookback period for features
                - retrain_interval: Days between model retraining
                - model_type: Type of ML model ('rf', 'xgb', 'ensemble')
        """
        super().__init__("MLHybrid", config)
        self.feature_cols = config.get('features', [
            'returns', 'volume', 'volatility_20', 'rsi', 'macd'
        ])
        self.lookback = config.get('lookback', 60)
        self.retrain_interval = config.get('retrain_interval', 30)
        self.model_type = config.get('model_type', 'ensemble')

        # Initialize models
        self.models = self._initialize_models()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_train_date = None

    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize ML models."""
        models = {}

        if self.model_type in ['rf', 'ensemble']:
            models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42
            )

        if self.model_type in ['xgb', 'ensemble']:
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )

        if self.model_type in ['gb', 'ensemble']:
            models['gradient_boosting'] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )

        return models

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML model.

        Args:
            data: Raw market data

        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame(index=data.index)

        # Price-based features
        if 'returns' in self.feature_cols:
            features['returns'] = data['close'].pct_change()
            features['returns_5'] = data['close'].pct_change(5)
            features['returns_20'] = data['close'].pct_change(20)

        # Volume features
        if 'volume' in self.feature_cols:
            features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
            features['volume_change'] = data['volume'].pct_change()

        # Volatility features
        if 'volatility' in self.feature_cols or 'volatility_20' in self.feature_cols:
            features['volatility'] = data['returns'].rolling(20).std()
            features['volatility_60'] = data['returns'].rolling(60).std()

        # Technical indicators
        if 'rsi' in self.feature_cols and 'rsi' in data.columns:
            features['rsi'] = data['rsi']
            features['rsi_change'] = data['rsi'].diff()

        if 'macd' in self.feature_cols and 'macd' in data.columns:
            features['macd'] = data['macd']
            features['macd_signal'] = data.get('macd_signal', 0)
            features['macd_hist'] = data.get('macd_hist', 0)

        # Moving averages
        features['ma_ratio_20_50'] = data.get('sma_20', data['close'].rolling(20).mean()) / \
                                     data.get('sma_50', data['close'].rolling(50).mean())

        # Price position relative to Bollinger Bands
        if 'bb_middle' in data.columns:
            features['bb_position'] = (data['close'] - data['bb_lower']) / \
                                      (data['bb_upper'] - data['bb_lower'])

        # ATR
        if 'atr_14' in data.columns:
            features['atr_ratio'] = data['atr_14'] / data['close']

        # Lag features
        for col in ['returns', 'volume_ratio', 'rsi']:
            if col in features.columns:
                for lag in [1, 2, 3, 5]:
                    features[f'{col}_lag_{lag}'] = features[col].shift(lag)

        return features.dropna()

    def _create_labels(self, data: pd.DataFrame, horizon: int = 1) -> pd.Series:
        """
        Create labels for supervised learning.

        Args:
            data: Market data
            horizon: Forward-looking periods for label

        Returns:
            Series of labels (1: up, 0: down)
        """
        # Forward returns
        forward_returns = data['close'].pct_change(horizon).shift(-horizon)

        # Binary classification: 1 if positive return, 0 otherwise
        labels = (forward_returns > 0).astype(int)

        return labels

    def train(self, data: pd.DataFrame):
        """
        Train ML models on historical data.

        Args:
            data: Historical market data
        """
        logger.info("Training ML models...")

        # Prepare features and labels
        features_df = self._prepare_features(data)
        labels = self._create_labels(data)

        # Align features and labels
        common_index = features_df.index.intersection(labels.index)
        X = features_df.loc[common_index]
        y = labels.loc[common_index]

        # Remove any remaining NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        if len(X) < 100:
            logger.warning(f"Insufficient data for training: {len(X)} samples")
            return

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train each model
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}...")
                model.fit(X_scaled, y)
                logger.info(f"{name} trained successfully")
            except Exception as e:
                logger.error(f"Error training {name}: {e}")

        self.is_trained = True
        self.last_train_date = data.index[-1]
        logger.info("ML models training complete")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals using ML models.

        Args:
            data: Market data

        Returns:
            Series of signals
        """
        if not self.is_trained:
            logger.warning("Models not trained, training now...")
            self.train(data)

        signals = pd.Series(0, index=data.index)

        # Prepare features
        features_df = self._prepare_features(data)

        if features_df.empty:
            logger.warning("No features available for prediction")
            return signals

        # Scale features
        try:
            X_scaled = self.scaler.transform(features_df)
        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            return signals

        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)
                predictions[name] = pred
            except Exception as e:
                logger.error(f"Error predicting with {name}: {e}")

        if not predictions:
            return signals

        # Ensemble: average predictions
        ensemble_pred = np.mean(list(predictions.values()), axis=0)

        # Convert to signals
        # Buy if ensemble predicts up (> 0.5), sell if predicts down (< 0.5)
        signals_array = np.where(ensemble_pred > 0.5, 1, -1)

        # Align with original index
        signals.loc[features_df.index] = signals_array

        buy_count = (signals == 1).sum()
        sell_count = (signals == -1).sum()
        logger.info(f"ML strategy generated {buy_count} buy and {sell_count} sell signals")

        return signals

    def get_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """
        Get feature importance from trained models.

        Returns:
            Dictionary of DataFrames with feature importances
        """
        importances = {}

        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                imp_df = pd.DataFrame({
                    'feature': self.scaler.feature_names_in_ if hasattr(self.scaler, 'feature_names_in_') else range(len(model.feature_importances_)),
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                importances[name] = imp_df

        return importances
