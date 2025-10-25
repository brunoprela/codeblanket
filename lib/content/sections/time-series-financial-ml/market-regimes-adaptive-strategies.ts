export const marketRegimesAdaptiveStrategies = {
  title: 'Market Regimes & Adaptive Strategies',
  id: 'market-regimes-adaptive-strategies',
  content: `
# Market Regimes & Adaptive Strategies

## Introduction

Markets don't behave the same way all the time. They cycle through different regimes with distinct statistical properties:

- **Bull Markets (Trending Up)**: Momentum works, buy dips
- **Bear Markets (Trending Down)**: Short-selling, defensive
- **Ranging Markets (Sideways)**: Mean reversion excels
- **High Volatility**: Risk-off, reduce exposure
- **Low Volatility**: Carry trades, leverage-friendly
- **Crisis**: Correlations → 1, diversification fails

**Why Regime Detection Matters**:
- Same strategy doesn't work everywhere
- Momentum in ranging markets = losses
- Mean reversion in trending markets = missed gains
- Adaptive strategies can improve Sharpe by 20-40%

**Approaches**:
1. **Hidden Markov Models (HMM)**: Statistical regime detection
2. **Volatility Clustering**: High/low vol regimes
3. **Trend Detection**: ADX, moving averages
4. **Factor Models**: Multiple indicators
5. **Machine Learning**: Random Forest, LSTM for regime classification

---

## Hidden Markov Model (HMM) Regime Detection

\`\`\`python
"""
Complete HMM implementation for regime detection
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
import yfinance as yf
import matplotlib.pyplot as plt

class HMMRegimeDetector:
    """
    Hidden Markov Model for Market Regime Detection
    
    Identifies latent market states based on returns and volatility
    """
    
    def __init__(self, n_regimes: int = 3):
        """
        Args:
            n_regimes: Number of hidden states (typically 2-4)
                2: Bull/Bear
                3: Bull/Neutral/Bear
                4: Bull/Range/Bear/Crisis
        """
        self.n_regimes = n_regimes
        self.model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=1000,
            random_state=42
        )
        self.fitted = False
    
    def prepare_features (self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare multi-dimensional features for HMM
        
        Features:
        - Returns
        - Volatility
        - Volume
        """
        features = pd.DataFrame (index=data.index)
        
        # Returns
        features['returns'] = data['Close'].pct_change()
        
        # Volatility (20-day rolling)
        features['volatility'] = features['returns'].rolling(20).std()
        
        # Volume (normalized)
        features['volume'] = (data['Volume'] - data['Volume'].rolling(20).mean()) / data['Volume'].rolling(20).std()
        
        features = features.dropna()
        
        return features.values
    
    def fit (self, data: pd.DataFrame):
        """Fit HMM to historical data"""
        X = self.prepare_features (data)
        
        self.model.fit(X)
        self.fitted = True
        
        return self
    
    def predict_regime (self, data: pd.DataFrame) -> pd.Series:
        """
        Predict regime for each time period
        
        Returns:
            Series of regime labels (0, 1, 2, ...)
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = self.prepare_features (data)
        regimes = self.model.predict(X)
        
        # Create series with original index (minus NaNs)
        features_df = pd.DataFrame(X, index=data.index[len (data) - len(X):])
        regime_series = pd.Series (regimes, index=features_df.index)
        
        return regime_series
    
    def regime_probabilities (self, data: pd.DataFrame) -> pd.DataFrame:
        """Get probability of each regime over time"""
        X = self.prepare_features (data)
        probs = self.model.predict_proba(X)
        
        prob_df = pd.DataFrame(
            probs,
            columns=[f'Regime_{i}_prob' for i in range (self.n_regimes)],
            index=data.index[len (data) - len(X):]
        )
        
        return prob_df
    
    def regime_statistics (self, data: pd.DataFrame, regimes: pd.Series) -> dict:
        """Calculate statistics for each regime"""
        returns = data['Close'].pct_change()
        
        stats = {}
        for regime_id in range (self.n_regimes):
            regime_mask = regimes == regime_id
            regime_returns = returns[regime_mask]
            
            if len (regime_returns) > 0:
                stats[regime_id] = {
                    'mean_return': regime_returns.mean(),
                    'volatility': regime_returns.std(),
                    'sharpe': (regime_returns.mean() / regime_returns.std() * np.sqrt(252)) if regime_returns.std() > 0 else 0,
                    'skewness': regime_returns.skew(),
                    'kurtosis': regime_returns.kurt(),
                    'max_drawdown': (regime_returns.cumsum() - regime_returns.cumsum().cummax()).min(),
                    'count': len (regime_returns),
                    'duration_pct': len (regime_returns) / len (returns)
                }
        
        return stats
    
    def label_regimes (self, stats: dict) -> dict:
        """
        Assign human-readable labels based on statistics
        
        Heuristic:
        - High mean, low vol = Bull
        - Low mean, high vol = Bear
        - Low mean, low vol = Range
        - High vol = Crisis
        """
        labeled = {}
        
        # Sort by mean return
        sorted_regimes = sorted (stats.items(), key=lambda x: x[1]['mean_return'], reverse=True)
        
        if self.n_regimes == 2:
            labeled[sorted_regimes[0][0]] = 'Bull'
            labeled[sorted_regimes[1][0]] = 'Bear'
        
        elif self.n_regimes == 3:
            labeled[sorted_regimes[0][0]] = 'Bull'
            labeled[sorted_regimes[1][0]] = 'Neutral'
            labeled[sorted_regimes[2][0]] = 'Bear'
        
        elif self.n_regimes == 4:
            # Check for crisis regime (high vol, negative return)
            crisis_regime = max (stats.items(), key=lambda x: x[1]['volatility'])
            if crisis_regime[1]['mean_return'] < 0:
                labeled[crisis_regime[0]] = 'Crisis'
                remaining = [r for r in sorted_regimes if r[0] != crisis_regime[0]]
                labeled[remaining[0][0]] = 'Bull'
                labeled[remaining[1][0]] = 'Neutral'
                labeled[remaining[2][0]] = 'Bear'
            else:
                labeled[sorted_regimes[0][0]] = 'Bull'
                labeled[sorted_regimes[1][0]] = 'Neutral'
                labeled[sorted_regimes[2][0]] = 'Range'
                labeled[sorted_regimes[3][0]] = 'Bear'
        
        return labeled
    
    def plot_regimes (self, data: pd.DataFrame, regimes: pd.Series, labels: dict = None):
        """Visualize regimes over time"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Plot price with regime colors
        for regime_id in range (self.n_regimes):
            regime_mask = regimes == regime_id
            label = labels[regime_id] if labels else f'Regime {regime_id}'
            
            ax1.scatter(
                data.index[regime_mask],
                data['Close'][regime_mask],
                label=label,
                s=10,
                alpha=0.6
            )
        
        ax1.set_ylabel('Price')
        ax1.set_title('Price with Market Regimes')
        ax1.legend()
        ax1.grid (alpha=0.3)
        
        # Plot regime over time
        ax2.plot (regimes.index, regimes, drawstyle='steps-post', linewidth=2)
        ax2.set_ylabel('Regime')
        ax2.set_xlabel('Date')
        ax2.set_title('Regime Sequence')
        ax2.grid (alpha=0.3)
        
        if labels:
            # Set y-axis labels
            ax2.set_yticks (list (labels.keys()))
            ax2.set_yticklabels (list (labels.values()))
        
        plt.tight_layout()
        plt.show()


# ============================================================================
# EXAMPLE: HMM REGIME DETECTION
# ============================================================================

# Download data
data = yf.download('SPY', start='2015-01-01', end='2024-01-01')

# Detect regimes
detector = HMMRegimeDetector (n_regimes=3)
detector.fit (data)

regimes = detector.predict_regime (data)
stats = detector.regime_statistics (data, regimes)
labels = detector.label_regimes (stats)

print("="*70)
print("HMM REGIME DETECTION RESULTS")
print("="*70)

for regime_id, label in labels.items():
    stat = stats[regime_id]
    print(f"\\n{label} (Regime {regime_id}):")
    print(f"  Mean Return: {stat['mean_return']:.4f} ({stat['mean_return']*252:.2%} annualized)")
    print(f"  Volatility: {stat['volatility']:.4f} ({stat['volatility']*np.sqrt(252):.2%} annualized)")
    print(f"  Sharpe Ratio: {stat['sharpe']:.2f}")
    print(f"  Duration: {stat['duration_pct']:.1%} of time")
    print(f"  Max Drawdown: {stat['max_drawdown']:.2%}")

# Plot
detector.plot_regimes (data, regimes, labels)
\`\`\`

---

## Simple Regime Detection Methods

\`\`\`python
"""
Simpler, faster regime detection methods
"""

class SimpleRegimeDetector:
    """
    Fast regime detection using technical indicators
    """
    
    @staticmethod
    def volatility_regime (returns: pd.Series, window: int = 21,
                         low_threshold: float = 0.33,
                         high_threshold: float = 0.67) -> pd.Series:
        """
        Classify by volatility levels
        
        Low Vol: Quiet markets, carry trades work
        Medium Vol: Normal markets
        High Vol: Stressed markets, reduce risk
        """
        vol = returns.rolling (window).std() * np.sqrt(252)
        
        regimes = pd.Series('medium', index=returns.index)
        regimes[vol < vol.quantile (low_threshold)] = 'low_vol'
        regimes[vol > vol.quantile (high_threshold)] = 'high_vol'
        
        return regimes
    
    @staticmethod
    def trend_regime (prices: pd.Series, short_window: int = 50,
                    long_window: int = 200) -> pd.Series:
        """
        Trending vs ranging using moving averages
        
        Trending: Price above/below MA with clear direction
        Ranging: Price oscillating around MA
        """
        sma_short = prices.rolling (short_window).mean()
        sma_long = prices.rolling (long_window).mean()
        
        regimes = pd.Series('ranging', index=prices.index)
        
        # Uptrend: Short MA > Long MA and price > both
        uptrend = (sma_short > sma_long) & (prices > sma_short)
        regimes[uptrend] = 'uptrend'
        
        # Downtrend: Short MA < Long MA and price < both
        downtrend = (sma_short < sma_long) & (prices < sma_short)
        regimes[downtrend] = 'downtrend'
        
        return regimes
    
    @staticmethod
    def adx_regime (data: pd.DataFrame, adx_threshold: int = 25) -> pd.Series:
        """
        Use ADX (Average Directional Index) for trend strength
        
        ADX > 25: Strong trend
        ADX < 25: Weak trend / ranging
        """
        # Calculate ADX
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr1 = high - low
        tr2 = abs (high - close.shift(1))
        tr3 = abs (low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max (axis=1)
        
        atr = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        
        dx = 100 * abs (plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean()
        
        regimes = pd.Series('ranging', index=data.index)
        regimes[adx > adx_threshold] = 'trending'
        
        return regimes
    
    @staticmethod
    def correlation_regime (returns_matrix: pd.DataFrame,
                          window: int = 60) -> pd.Series:
        """
        Detect correlation regimes
        
        High correlation: Crisis, diversification doesn't work
        Low correlation: Normal, diversification works
        """
        # Calculate rolling average correlation
        rolling_corr = returns_matrix.rolling (window).corr()
        
        # Average pairwise correlation for each date
        avg_corr = rolling_corr.groupby (level=0).apply(
            lambda x: x.values[np.triu_indices_from (x.values, k=1)].mean()
        )
        
        regimes = pd.Series('normal', index=avg_corr.index)
        regimes[avg_corr > avg_corr.quantile(0.75)] = 'crisis'  # High correlation
        regimes[avg_corr < avg_corr.quantile(0.25)] = 'dispersed'  # Low correlation
        
        return regimes


# ============================================================================
# EXAMPLE: SIMPLE REGIME DETECTION
# ============================================================================

returns = data['Close'].pct_change().dropna()

# Volatility regime
vol_regime = SimpleRegimeDetector.volatility_regime (returns)

# Trend regime
trend_regime = SimpleRegimeDetector.trend_regime (data['Close'])

# ADX regime
adx_regime = SimpleRegimeDetector.adx_regime (data)

print("\\n" + "="*70)
print("SIMPLE REGIME DETECTION")
print("="*70)

print("\\nVolatility Regime Distribution:")
print(vol_regime.value_counts (normalize=True))

print("\\nTrend Regime Distribution:")
print(trend_regime.value_counts (normalize=True))

print("\\nADX Regime Distribution:")
print(adx_regime.value_counts (normalize=True))
\`\`\`

---

## Adaptive Strategy Implementation

\`\`\`python
"""
Complete adaptive strategy that switches based on regime
"""

class AdaptiveStrategy:
    """
    Multi-strategy system that adapts to market regimes
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_regime = None
        
        # Strategy components
        self.strategies = {
            'Bull': self.momentum_strategy,
            'Bear': self.defensive_strategy,
            'Neutral': self.mean_reversion_strategy,
            'low_vol': self.carry_strategy,
            'high_vol': self.risk_off_strategy,
            'uptrend': self.trend_following,
            'downtrend': self.short_strategy,
            'ranging': self.range_trading,
        }
    
    def momentum_strategy (self, data: pd.DataFrame) -> pd.Series:
        """
        Bull market strategy: Momentum
        
        Buy winners, ride trends
        """
        lookback = 21
        returns = data['Close'].pct_change (lookback)
        
        signals = pd.Series(0, index=data.index)
        signals[returns > returns.quantile(0.7)] = 1  # Buy top 30%
        
        return signals
    
    def defensive_strategy (self, data: pd.DataFrame) -> pd.Series:
        """
        Bear market strategy: Defensive / Short
        
        Reduce exposure or short
        """
        signals = pd.Series(-0.5, index=data.index)  # 50% short or hedge
        return signals
    
    def mean_reversion_strategy (self, data: pd.DataFrame) -> pd.Series:
        """
        Neutral / Ranging market: Mean reversion
        
        Buy oversold, sell overbought
        """
        window = 20
        sma = data['Close'].rolling (window).mean()
        std = data['Close'].rolling (window).std()
        z_score = (data['Close'] - sma) / std
        
        signals = pd.Series(0, index=data.index)
        signals[z_score < -2] = 1  # Oversold - buy
        signals[z_score > 2] = -1  # Overbought - sell
        
        return signals
    
    def carry_strategy (self, data: pd.DataFrame) -> pd.Series:
        """
        Low volatility: Carry / leveraged
        
        Low risk environment - can use more leverage
        """
        signals = pd.Series(1.5, index=data.index)  # 150% exposure
        return signals
    
    def risk_off_strategy (self, data: pd.DataFrame) -> pd.Series:
        """
        High volatility: Risk-off
        
        Reduce exposure significantly
        """
        signals = pd.Series(0.3, index=data.index)  # Only 30% exposure
        return signals
    
    def trend_following (self, data: pd.DataFrame) -> pd.Series:
        """Uptrend: Follow the trend"""
        signals = pd.Series(1, index=data.index)
        return signals
    
    def short_strategy (self, data: pd.DataFrame) -> pd.Series:
        """Downtrend: Short"""
        signals = pd.Series(-1, index=data.index)
        return signals
    
    def range_trading (self, data: pd.DataFrame) -> pd.Series:
        """Range-bound: Buy support, sell resistance"""
        window = 20
        high = data['High'].rolling (window).max()
        low = data['Low'].rolling (window).min()
        
        signals = pd.Series(0, index=data.index)
        signals[data['Close'] <= low * 1.01] = 1  # Near support - buy
        signals[data['Close'] >= high * 0.99] = -1  # Near resistance - sell
        
        return signals
    
    def generate_adaptive_signals (self, data: pd.DataFrame,
                                  regimes: pd.Series) -> pd.Series:
        """
        Generate signals by adapting to detected regime
        
        Args:
            data: OHLCV data
            regimes: Regime labels for each period
        
        Returns:
            Position signals (-1, 0, 1 or fractional for sizing)
        """
        signals = pd.Series(0, index=data.index)
        
        for date in regimes.index:
            if date not in data.index:
                continue
            
            regime = regimes.loc[date]
            
            if regime in self.strategies:
                # Get strategy for this regime
                strategy_func = self.strategies[regime]
                
                # Generate signal for current data point
                current_data = data.loc[:date]
                regime_signal = strategy_func (current_data)
                
                if date in regime_signal.index:
                    signals.loc[date] = regime_signal.loc[date]
        
        return signals
    
    def backtest (self, data: pd.DataFrame, signals: pd.Series) -> dict:
        """
        Simple backtest of adaptive strategy
        """
        returns = data['Close'].pct_change()
        strategy_returns = signals.shift(1) * returns
        
        equity_curve = (1 + strategy_returns).cumprod()
        
        # Performance metrics
        total_return = equity_curve.iloc[-1] - 1
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        
        # Max drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_dd = drawdown.min()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'final_equity': equity_curve.iloc[-1] * self.initial_capital
        }


# ============================================================================
# EXAMPLE: ADAPTIVE STRATEGY BACKTEST
# ============================================================================

# Detect regimes (using HMM from earlier)
detector = HMMRegimeDetector (n_regimes=3)
detector.fit (data)
regimes = detector.predict_regime (data)
labels_dict = detector.label_regimes (detector.regime_statistics (data, regimes))

# Map regime IDs to labels
regime_labels = regimes.map (labels_dict)

# Create adaptive strategy
adaptive = AdaptiveStrategy (initial_capital=100000)
adaptive_signals = adaptive.generate_adaptive_signals (data, regime_labels)

# Backtest
adaptive_performance = adaptive.backtest (data, adaptive_signals)

# Compare to buy-and-hold
returns = data['Close'].pct_change()
bh_equity = (1 + returns).cumprod()
bh_return = bh_equity.iloc[-1] - 1
bh_sharpe = returns.mean() / returns.std() * np.sqrt(252)
bh_dd = ((bh_equity - bh_equity.cummax()) / bh_equity.cummax()).min()

print("\\n" + "="*70)
print("ADAPTIVE STRATEGY VS BUY-AND-HOLD")
print("="*70)

print("\\nAdaptive Strategy:")
print(f"  Total Return: {adaptive_performance['total_return']:.2%}")
print(f"  Sharpe Ratio: {adaptive_performance['sharpe_ratio']:.2f}")
print(f"  Max Drawdown: {adaptive_performance['max_drawdown']:.2%}")
print(f"  Final Equity: $\${adaptive_performance['final_equity']:,.0f}")

print("\\nBuy-and-Hold:")
print(f"  Total Return: {bh_return:.2%}")
print(f"  Sharpe Ratio: {bh_sharpe:.2f}")
print(f"  Max Drawdown: {bh_dd:.2%}")

print("\\nImprovement:")
print(f"  Return: {(adaptive_performance['total_return'] - bh_return):.2%}")
print(f"  Sharpe: {(adaptive_performance['sharpe_ratio'] - bh_sharpe):+.2f}")
print(f"  Max DD: {(adaptive_performance['max_drawdown'] - bh_dd):.2%}")

print("=" * 70)
\`\`\`

---

## Key Takeaways

**Regime Detection Methods**:
1. **HMM**: Most sophisticated, finds hidden states
2. **Volatility**: Simple, effective for risk management
3. **Trend (ADX, MA)**: Good for strategy switching
4. **Correlation**: Detects crisis regimes
5. **Combine**: Use multiple for robustness

**Common Regimes**:
- **Bull**: Momentum, trend-following, leverage
- **Bear**: Defensive, short, hedging
- **Ranging**: Mean reversion, sell premium
- **High Vol**: Risk-off, reduce size
- **Crisis**: Cash, flight to quality

**Adaptive Strategy Benefits**:
- **Higher Sharpe**: 20-40% improvement typical
- **Lower Drawdowns**: Avoid worst periods
- **Robustness**: Works across market conditions
- **Risk Management**: Automatic de-risking

**Implementation Tips**:
1. **Don't over-fit**: Use simple, robust regimes
2. **Lag aware**: Regimes detected with delay
3. **Transition smoothly**: Don't whipsaw
4. **Combine signals**: Multiple regime indicators
5. **Backtest thoroughly**: Across all regimes

**Remember**: No regime lasts forever. The ability to adapt is more valuable than any single strategy.
`,
};
