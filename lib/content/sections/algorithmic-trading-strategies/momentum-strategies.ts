export const momentumStrategies = {
  title: 'Momentum Strategies',
  slug: 'momentum-strategies',
  description:
    'Master momentum trading: cross-sectional momentum, time-series momentum, and factor-based approaches',
  content: `
# Momentum Strategies

## Introduction: Trading Persistence

Momentum - the tendency of assets that have performed well (poorly) to continue performing well (poorly) - is one of the most robust anomalies in finance. Eugene Fama called it "the premier anomaly." AQR, Two Sigma, and other quant funds have built empires on momentum.

**What you'll learn:**
- Cross-sectional momentum (relative strength)
- Time-series momentum (trend following)
- Factor momentum and multi-factor models
- Risk management for momentum strategies
- Why momentum works (behavioral finance)

**Why momentum matters:**
- Works across all asset classes
- 80+ years of evidence (since 1930s)
- Sharpe ratios: 0.8-1.2
- Complements mean reversion

**Key Research:**
- Jegadeesh & Titman (1993): First academic paper
- AQR (2014): Momentum works everywhere
- Carhart (1997): Four-factor model (adds momentum)

---

## Cross-Sectional Momentum

### Concept: Relative Strength

Buy top performers, short bottom performers within a universe.

**Classic Approach:**1. Universe: S&P 500
2. Ranking period: 12 months (exclude last month)
3. Holding period: 1 month
4. Rebalance monthly
5. Long top 10%, short bottom 10%

\`\`\`python
import pandas as pd
import numpy as np
from typing import List, Tuple

class CrossSectionalMomentum:
    """
    Cross-sectional momentum strategy
    
    Buy winners, sell losers relative to universe
    """
    
    def __init__(self,
                 universe: List[str],
                 lookback_months: int = 12,
                 holding_months: int = 1,
                 long_pct: float = 0.10,
                 short_pct: float = 0.10):
        """
        Initialize momentum strategy
        
        Args:
            universe: List of symbols
            lookback_months: Ranking period (12 months standard)
            holding_months: Rebalance frequency
            long_pct: Top percentile to long (10% = top decile)
            short_pct: Bottom percentile to short (10% = bottom decile)
        """
        self.universe = universe
        self.lookback = lookback_months
        self.holding = holding_months
        self.long_pct = long_pct
        self.short_pct = short_pct
    
    def calculate_momentum(self,
                          prices: pd.DataFrame,
                          skip_last_month: bool = True) -> pd.DataFrame:
        """
        Calculate momentum scores
        
        Momentum = Cumulative return over lookback period
        
        Args:
            prices: DataFrame of prices
            skip_last_month: Skip last month (reduce reversal risk)
            
        Returns:
            DataFrame of momentum scores
        """
        # Calculate returns
        returns = prices.pct_change()
        
        if skip_last_month:
            # Standard: Use months t-12 to t-2 (skip t-1)
            momentum = (1 + returns).rolling(window=self.lookback).apply(
                lambda x: x[:-21].prod() - 1  # Skip last ~21 days
            )
        else:
            momentum = (1 + returns).rolling(window=self.lookback).apply(
                lambda x: x.prod() - 1
            )
        
        return momentum
    
    def rank_securities(self, momentum_scores: pd.Series) -> pd.DataFrame:
        """
        Rank securities by momentum
        
        Returns:
            DataFrame with ranks and percentiles
        """
        df = pd.DataFrame()
        df['score'] = momentum_scores
        df['rank'] = momentum_scores.rank(ascending=False)
        df['percentile'] = df['rank'] / len(momentum_scores)
        
        return df
    
    def generate_portfolio(self,
                          momentum_scores: pd.DataFrame) -> pd.Series:
        """
        Generate long/short portfolio
        
        Returns:
            Series with positions (-1, 0, +1)
        """
        positions = pd.Series(0, index=momentum_scores.index)
        
        for date in momentum_scores.index:
            scores = momentum_scores.loc[date].dropna()
            
            # Rank
            ranked = scores.rank(ascending=False, pct=True)
            
            # Long top percentile
            positions.loc[date, ranked <= self.long_pct] = 1
            
            # Short bottom percentile
            positions.loc[date, ranked >= (1 - self.short_pct)] = -1
        
        return positions
    
    def backtest(self,
                prices: pd.DataFrame,
                returns: pd.DataFrame,
                initial_capital: float = 1_000_000) -> dict:
        """
        Backtest momentum strategy
        
        Returns:
            Performance metrics
        """
        # Calculate momentum
        momentum = self.calculate_momentum(prices)
        
        # Generate positions
        positions = self.generate_portfolio(momentum)
        
        # Calculate portfolio returns
        portfolio_returns = (positions.shift(self.holding) * returns).sum(axis=1)
        
        # Performance metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'volatility': volatility
        }

# Example usage
sp500_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', ...]  # Full S&P 500

strategy = CrossSectionalMomentum(
    universe=sp500_stocks,
    lookback_months=12,
    holding_months=1,
    long_pct=0.10,
    short_pct=0.10
)

print("\\n=== Cross-Sectional Momentum Strategy ===")
print(f"Universe: {len(sp500_stocks)} stocks")
print(f"Long top {strategy.long_pct*100:.0f}%, Short bottom {strategy.short_pct*100:.0f}%")
print(f"Lookback: {strategy.lookback} months")
print(f"Rebalance: {strategy.holding} month(s)")
\`\`\`

---

## Time-Series Momentum

### Concept: Absolute Performance

Trade each asset based on its own historical performance (not relative to others).

**Approach:**
- If asset up over past 12 months → long
- If asset down over past 12 months → short
- Each asset independent decision

\`\`\`python
class TimeSeriesMomentum:
    """
    Time-series momentum (absolute momentum)
    
    Each asset traded independently based on own history
    """
    
    def __init__(self,
                 lookback_months: int = 12,
                 threshold: float = 0.0):
        """
        Initialize time-series momentum
        
        Args:
            lookback_months: Historical period
            threshold: Return threshold for signal (0 = zero)
        """
        self.lookback = lookback_months
        self.threshold = threshold
    
    def generate_signals(self,
                        returns: pd.DataFrame) -> pd.DataFrame:
        """
        Generate momentum signals
        
        Signal = sign(cumulative_return - threshold)
        
        Returns:
            DataFrame of signals (-1, 0, +1)
        """
        # Calculate cumulative returns
        cum_returns = (1 + returns).rolling(window=self.lookback).apply(
            lambda x: x.prod() - 1
        )
        
        # Generate signals
        signals = pd.DataFrame(0, index=returns.index, columns=returns.columns)
        signals[cum_returns > self.threshold] = 1
        signals[cum_returns < -self.threshold] = -1
        
        return signals
    
    def calculate_portfolio_returns(self,
                                    signals: pd.DataFrame,
                                    returns: pd.DataFrame) -> pd.Series:
        """
        Calculate portfolio returns
        
        Equal-weighted across assets
        """
        # Weight = signal / number_of_signals
        num_signals = signals.abs().sum(axis=1)
        weights = signals.div(num_signals, axis=0).fillna(0)
        
        # Portfolio returns
        portfolio_returns = (weights.shift(1) * returns).sum(axis=1)
        
        return portfolio_returns

# Example: Apply to futures
futures = ['ES', 'NQ', 'GC', 'CL', 'ZB', 'ZN']  # Equity, gold, oil, bonds

ts_momentum = TimeSeriesMomentum(lookback_months=12)

print("\\n=== Time-Series Momentum ===")
print("Trade each asset independently")
print("Long if up over 12 months, short if down")
\`\`\`

---

## Factor Momentum

### Multi-Factor Approach

Combine momentum with other factors for enhanced performance.

**Common Factors:**1. Value (book-to-market)
2. Size (market cap)
3. Momentum (12-month return)
4. Quality (ROE, margins)
5. Low volatility

\`\`\`python
class FactorMomentum:
    """
    Multi-factor momentum strategy
    
    Combine momentum with value, quality, etc.
    """
    
    def __init__(self):
        self.factors = ['momentum', 'value', 'quality', 'size']
        self.weights = {'momentum': 0.40, 'value': 0.30, 'quality': 0.20, 'size': 0.10}
    
    def calculate_momentum_factor(self, prices: pd.DataFrame) -> pd.DataFrame:
        """12-month momentum (skip last month)"""
        returns = prices.pct_change()
        momentum = (1 + returns).rolling(252).apply(lambda x: x[:-21].prod() - 1)
        return momentum
    
    def calculate_value_factor(self, prices: pd.DataFrame, book_values: pd.DataFrame) -> pd.DataFrame:
        """Book-to-market ratio"""
        return book_values / prices
    
    def calculate_quality_factor(self, roe: pd.DataFrame, margins: pd.DataFrame) -> pd.DataFrame:
        """Quality score from ROE and margins"""
        return (roe + margins) / 2
    
    def calculate_size_factor(self, market_caps: pd.DataFrame) -> pd.DataFrame:
        """Inverse market cap (favor small)"""
        return 1 / market_caps
    
    def calculate_composite_score(self,
                                  factor_scores: dict) -> pd.DataFrame:
        """
        Combine factors into composite score
        
        Score = Σ (weight_i × z-score_i)
        """
        composite = pd.DataFrame(0, index=factor_scores['momentum'].index,
                                columns=factor_scores['momentum'].columns)
        
        for factor, weight in self.weights.items():
            # Z-score each factor
            scores = factor_scores[factor]
            z_scores = (scores - scores.mean()) / scores.std()
            
            # Add to composite
            composite += weight * z_scores
        
        return composite
    
    def generate_portfolio(self,
                          composite_scores: pd.DataFrame,
                          top_pct: float = 0.20) -> pd.DataFrame:
        """
        Generate portfolio from composite scores
        
        Long top 20% (high composite score)
        """
        positions = pd.DataFrame(0, index=composite_scores.index,
                                columns=composite_scores.columns)
        
        for date in composite_scores.index:
            scores = composite_scores.loc[date]
            ranked = scores.rank(ascending=False, pct=True)
            
            positions.loc[date, ranked <= top_pct] = 1
        
        return positions
\`\`\`

---

## Why Momentum Works

### Behavioral Explanations

**1. Under-reaction to News**
- Investors slow to incorporate information
- Prices drift after earnings announcements
- Momentum captures this drift

**2. Herding**
- Investors follow trends
- "Buy what's going up"
- Self-reinforcing

**3. Disposition Effect**
- Sell winners too early
- Hold losers too long
- Creates momentum

### Risk-Based Explanations

**1. Compensation for Crash Risk**
- Momentum strategies crash during reversals
- 2008-2009: -50% drawdowns
- Investors demand premium

**2. Exposure to Macroeconomic Risk**
- Momentum loads on business cycle
- Performs well in expansions
- Crashes in recessions

---

## Risk Management

### Momentum Crashes

**Problem**: Momentum strategies crash during sharp reversals

**2009 Example:**
- Winners became losers overnight
- Momentum strategies: -50% in Q1 2009
- Recovery took 2 years

**Solutions:**

\`\`\`python
class MomentumRiskManager:
    """
    Protect against momentum crashes
    """
    
    def detect_crash_risk(self, returns: pd.DataFrame) -> bool:
        """
        Detect high crash risk
        
        Indicators:
        1. High market volatility (VIX > 30)
        2. Large recent losses (down > 10%)
        3. High correlation (everything moving together)
        """
        # Market volatility
        vol = returns.std() * np.sqrt(252)
        high_vol = vol > 0.30
        
        # Recent losses
        recent_return = returns.iloc[-20:].sum()
        large_loss = recent_return < -0.10
        
        # Correlation
        corr_matrix = returns.iloc[-60:].corr()
        avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        high_corr = avg_corr > 0.70
        
        crash_risk = high_vol and large_loss and high_corr
        
        return crash_risk
    
    def apply_volatility_scaling(self,
                                positions: pd.DataFrame,
                                target_vol: float = 0.15) -> pd.DataFrame:
        """
        Scale positions by volatility
        
        Lower positions when volatility high
        """
        # Calculate rolling volatility
        returns = positions.pct_change()
        realized_vol = returns.rolling(60).std() * np.sqrt(252)
        
        # Scaling factor
        scale = target_vol / realized_vol
        scale = scale.clip(0.25, 2.0)  # Limit to [0.25, 2.0]
        
        # Apply scaling
        scaled_positions = positions * scale
        
        return scaled_positions
\`\`\`

---

## Summary

**Cross-Sectional Momentum:**
- Long top 10%, short bottom 10%
- Rebalance monthly
- Sharpe: 0.8-1.2

**Time-Series Momentum:**
- Trade each asset independently
- Long if up, short if down
- Works across asset classes

**Factor Momentum:**
- Combine with value, quality
- Enhanced returns
- Better diversification

**Risk Management:**
- Volatility scaling
- Crash detection
- Diversification

**Next Section**: Market Making Strategies
`,
};
