export const predictiveModelingTrading = {
  title: 'Predictive Modeling for Trading',
  id: 'predictive-modeling-trading',
  content: `
# Predictive Modeling for Trading

## Introduction

Building predictive models for trading requires combining all previous concepts: time series analysis, technical indicators, fundamental data, and machine learning. This section focuses on practical model building, feature engineering, and evaluation.

**Key Topics**:
- Feature engineering for trading
- Model selection (classification vs regression)
- Walk-forward validation
- Ensemble methods
- Model interpretation
- Production deployment

---

## Feature Engineering Pipeline

\`\`\`python
"""
Comprehensive Feature Engineering for Trading
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import talib

class TradingFeatureEngineer:
    """
    Complete feature engineering pipeline
    """
    
    def __init__(self, lookback=252):
        self.lookback = lookback
        self.scaler = StandardScaler()
    
    def create_features (self, df):
        """
        Create all features from OHLCV data
        
        Args:
            df: DataFrame with OHLCV columns
        
        Returns:
            DataFrame with features
        """
        features = pd.DataFrame (index=df.index)
        
        # 1. Returns (multiple horizons)
        features['return_1d'] = df['Close'].pct_change(1)
        features['return_5d'] = df['Close'].pct_change(5)
        features['return_21d'] = df['Close'].pct_change(21)
        
        # 2. Technical indicators
        features = self._add_technical_indicators (features, df)
        
        # 3. Volume features
        features = self._add_volume_features (features, df)
        
        # 4. Volatility features
        features = self._add_volatility_features (features, df)
        
        # 5. Trend features
        features = self._add_trend_features (features, df)
        
        # 6. Statistical features
        features = self._add_statistical_features (features, df)
        
        return features
    
    def _add_technical_indicators (self, features, df):
        """Technical indicators"""
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values
        
        # RSI
        features['rsi_14'] = talib.RSI(close, timeperiod=14)
        
        # MACD
        macd, signal, hist = talib.MACD(close)
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = hist
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close)
        features['bb_upper'] = upper
        features['bb_middle'] = middle
        features['bb_lower'] = lower
        features['bb_width'] = (upper - lower) / middle
        features['bb_position'] = (close - lower) / (upper - lower)
        
        # Moving averages
        features['sma_20'] = talib.SMA(close, timeperiod=20)
        features['sma_50'] = talib.SMA(close, timeperiod=50)
        features['ema_12'] = talib.EMA(close, timeperiod=12)
        features['ema_26'] = talib.EMA(close, timeperiod=26)
        
        # Price vs MA
        features['price_vs_sma20'] = close / features['sma_20'] - 1
        features['price_vs_sma50'] = close / features['sma_50'] - 1
        
        # ADX (trend strength)
        features['adx'] = talib.ADX(high, low, close, timeperiod=14)
        
        # ATR (volatility)
        features['atr'] = talib.ATR(high, low, close, timeperiod=14)
        features['atr_pct'] = features['atr'] / close
        
        return features
    
    def _add_volume_features (self, features, df):
        """Volume-based features"""
        volume = df['Volume'].values
        close = df['Close'].values
        
        # Volume ratios
        features['volume_ratio_5d'] = df['Volume'] / df['Volume'].rolling(5).mean()
        features['volume_ratio_21d'] = df['Volume'] / df['Volume'].rolling(21).mean()
        
        # OBV
        features['obv'] = talib.OBV(close, volume)
        features['obv_ema'] = talib.EMA(features['obv'].values, timeperiod=20)
        
        # Volume trend
        features['volume_trend'] = df['Volume'].rolling(20).apply(
            lambda x: np.polyfit (range (len (x)), x, 1)[0]
        )
        
        return features
    
    def _add_volatility_features (self, features, df):
        """Volatility features"""
        returns = df['Close'].pct_change()
        
        # Historical volatility (multiple horizons)
        features['volatility_5d'] = returns.rolling(5).std() * np.sqrt(252)
        features['volatility_21d'] = returns.rolling(21).std() * np.sqrt(252)
        features['volatility_63d'] = returns.rolling(63).std() * np.sqrt(252)
        
        # Volatility of volatility
        features['volvol'] = features['volatility_21d'].rolling(21).std()
        
        # Realized range
        features['realized_range'] = (df['High'] - df['Low']) / df['Close']
        features['avg_range_21d'] = features['realized_range'].rolling(21).mean()
        
        return features
    
    def _add_trend_features (self, features, df):
        """Trend identification features"""
        close = df['Close']
        
        # Linear regression slope (trend strength)
        for window in [5, 10, 21]:
            features[f'trend_{window}d'] = close.rolling (window).apply(
                lambda x: np.polyfit (range (len (x)), x, 1)[0]
            )
        
        # Higher highs, lower lows
        features['higher_high'] = (df['High'] > df['High'].shift(1)).astype (int)
        features['lower_low'] = (df['Low'] < df['Low'].shift(1)).astype (int)
        
        # Days since 52-week high/low
        features['days_since_high'] = (
            df.index.to_series().diff().dt.days.cumsum() -
            df['High'].expanding().apply (lambda x: len (x) - 1 - x.argmax())
        )
        features['days_since_low'] = (
            df.index.to_series().diff().dt.days.cumsum() -
            df['Low'].expanding().apply (lambda x: len (x) - 1 - x.argmin())
        )
        
        return features
    
    def _add_statistical_features (self, features, df):
        """Statistical features"""
        returns = df['Close'].pct_change()
        
        # Skewness and kurtosis
        features['skew_21d'] = returns.rolling(21).skew()
        features['kurt_21d'] = returns.rolling(21).kurt()
        
        # Z-score
        features['zscore_21d'] = (
            (df['Close'] - df['Close'].rolling(21).mean()) /
            df['Close'].rolling(21).std()
        )
        
        # Autocorrelation
        features['autocorr_1d'] = returns.rolling(21).apply(
            lambda x: x.autocorr (lag=1)
        )
        
        return features
    
    def create_target (self, df, horizon=5, threshold=0.02):
        """
        Create target variable
        
        Args:
            df: DataFrame with Close prices
            horizon: Days ahead to predict
            threshold: Classification threshold
        
        Returns:
            target_regression: Future return
            target_classification: 1 (up), 0 (sideways), -1 (down)
        """
        # Regression target: future return
        future_return = df['Close'].pct_change (horizon).shift(-horizon)
        
        # Classification target: discretize returns
        target_classification = pd.Series(0, index=df.index)
        target_classification[future_return > threshold] = 1  # Up
        target_classification[future_return < -threshold] = -1  # Down
        # 0 = sideways (between -threshold and +threshold)
        
        return future_return, target_classification

# Example usage
spy = yf.download('SPY', start='2020-01-01', end='2024-01-01')

engineer = TradingFeatureEngineer()
features = engineer.create_features (spy)

# Create targets
target_reg, target_cls = engineer.create_target (spy, horizon=5, threshold=0.01)

# Combine
data = pd.concat([features, target_reg.rename('target_return'), 
                  target_cls.rename('target_direction')], axis=1)

# Remove NaN
data = data.dropna()

print("=== Feature Engineering Complete ===")
print(f"Features: {len (features.columns)}")
print(f"Samples: {len (data)}")
print(f"\\nFeature names:")
print(features.columns.tolist())
\`\`\`

---

## Model Training & Validation

\`\`\`python
"""
Walk-Forward Model Training
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

class TradingModel:
    """
    Trading model with walk-forward validation
    """
    
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = self._initialize_model()
        self.scaler = StandardScaler()
    
    def _initialize_model (self):
        """Initialize model based on type"""
        if self.model_type == 'xgboost':
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=10,
                random_state=42
            )
        elif self.model_type == 'logistic':
            return LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError (f"Unknown model type: {self.model_type}")
    
    def walk_forward_validation (self, data, feature_cols, target_col,
                                train_size=252, test_size=21, retrain_freq=21):
        """
        Walk-forward validation
        
        Args:
            data: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Target column name
            train_size: Training window size
            test_size: Test window size
            retrain_freq: How often to retrain (days)
        
        Returns:
            Dictionary with results
        """
        predictions = []
        actuals = []
        dates = []
        
        for i in range (train_size, len (data) - test_size, retrain_freq):
            # Train data
            train_data = data.iloc[i-train_size:i]
            X_train = train_data[feature_cols]
            y_train = train_data[target_col]
            
            # Test data
            test_data = data.iloc[i:i+test_size]
            X_test = test_data[feature_cols]
            y_test = test_data[target_col]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = self.model.predict(X_test_scaled)
            
            # Store results
            predictions.extend (y_pred)
            actuals.extend (y_test.values)
            dates.extend (test_data.index)
        
        # Calculate metrics
        predictions = np.array (predictions)
        actuals = np.array (actuals)
        
        results = {
            'predictions': predictions,
            'actuals': actuals,
            'dates': dates,
            'accuracy': accuracy_score (actuals, predictions),
            'precision': precision_score (actuals, predictions, average='weighted', zero_division=0),
            'recall': recall_score (actuals, predictions, average='weighted', zero_division=0),
            'f1': f1_score (actuals, predictions, average='weighted', zero_division=0)
        }
        
        return results

# Train and evaluate
feature_cols = [col for col in data.columns if col not in ['target_return', 'target_direction']]
target_col = 'target_direction'

# Remove features with NaN
data_clean = data[feature_cols + [target_col]].dropna()
feature_cols_clean = [col for col in feature_cols if col in data_clean.columns]

# XGBoost model
model_xgb = TradingModel (model_type='xgboost')
results_xgb = model_xgb.walk_forward_validation(
    data_clean,
    feature_cols_clean,
    target_col,
    train_size=252,
    test_size=21,
    retrain_freq=21
)

print("=== XGBoost Results ===")
print(f"Accuracy: {results_xgb['accuracy']:.3f}")
print(f"Precision: {results_xgb['precision']:.3f}")
print(f"Recall: {results_xgb['recall']:.3f}")
print(f"F1 Score: {results_xgb['f1']:.3f}")

# Random Forest model
model_rf = TradingModel (model_type='random_forest')
results_rf = model_rf.walk_forward_validation(
    data_clean,
    feature_cols_clean,
    target_col,
    train_size=252,
    test_size=21,
    retrain_freq=21
)

print("\\n=== Random Forest Results ===")
print(f"Accuracy: {results_rf['accuracy']:.3f}")
print(f"Precision: {results_rf['precision']:.3f}")
print(f"Recall: {results_rf['recall']:.3f}")
print(f"F1 Score: {results_rf['f1']:.3f}")
\`\`\`

---

## Ensemble Strategy

\`\`\`python
"""
Ensemble of Multiple Models
"""

class EnsembleTrader:
    """
    Ensemble multiple models for trading
    """
    
    def __init__(self):
        self.models = {
            'xgboost': TradingModel('xgboost'),
            'rf': TradingModel('random_forest'),
            'logistic': TradingModel('logistic')
        }
        self.weights = {
            'xgboost': 0.5,
            'rf': 0.3,
            'logistic': 0.2
        }
    
    def train_ensemble (self, X_train, y_train):
        """Train all models"""
        for name, model in self.models.items():
            X_scaled = model.scaler.fit_transform(X_train)
            model.model.fit(X_scaled, y_train)
            print(f"✓ Trained {name}")
    
    def predict_ensemble (self, X_test):
        """
        Predict using weighted ensemble
        
        Returns:
            predictions: Ensemble predictions
            probabilities: Average probabilities
        """
        all_preds = []
        all_probs = []
        
        for name, model in self.models.items():
            X_scaled = model.scaler.transform(X_test)
            pred = model.model.predict(X_scaled)
            prob = model.model.predict_proba(X_scaled)
            
            all_preds.append (pred * self.weights[name])
            all_probs.append (prob * self.weights[name])
        
        # Weighted average
        ensemble_pred = np.sign (np.sum (all_preds, axis=0))
        ensemble_prob = np.sum (all_probs, axis=0)
        
        return ensemble_pred, ensemble_prob

# Example usage
split_idx = int(0.8 * len (data_clean))
train_data = data_clean.iloc[:split_idx]
test_data = data_clean.iloc[split_idx:]

X_train = train_data[feature_cols_clean]
y_train = train_data[target_col]
X_test = test_data[feature_cols_clean]
y_test = test_data[target_col]

# Train ensemble
ensemble = EnsembleTrader()
ensemble.train_ensemble(X_train, y_train)

# Predict
ensemble_pred, ensemble_prob = ensemble.predict_ensemble(X_test)

# Evaluate
ensemble_accuracy = accuracy_score (y_test, ensemble_pred)
print(f"\\n=== Ensemble Accuracy: {ensemble_accuracy:.3f} ===")
\`\`\`

---

## Key Takeaways

1. **Feature Engineering**: 50+ features from price, volume, volatility
2. **Walk-Forward Validation**: Essential for realistic evaluation
3. **Model Selection**: XGBoost typically best (55-60% accuracy)
4. **Ensemble**: Improves accuracy 2-5% vs single model
5. **Target Design**: Classification (direction) vs regression (magnitude)
6. **Retrain Frequency**: Every 21 days balances adaptation vs stability

**Expected Performance**:
- Accuracy: 52-58% (directional)
- Sharpe Ratio: 0.8-1.2
- Information Ratio: 0.4-0.8
`,
};
