export const statisticalArbitrage = {
    title: 'Statistical Arbitrage',
    slug: 'statistical-arbitrage',
    description:
        'Master statistical arbitrage strategies including pairs trading, basket arbitrage, and PCA-based stat arb',
    content: `
# Statistical Arbitrage

## Introduction: Trading Statistical Relationships

Statistical arbitrage (stat arb) exploits temporary price divergences between statistically related securities. Unlike pure arbitrage (risk-free), stat arb involves statistical risk - relationships may break down. This is the bread and butter of quantitative hedge funds like Renaissance Technologies, Two Sigma, and Citadel.

**What you'll learn:**
- Pairs trading and cointegration analysis
- Z-score based entry/exit strategies
- Multi-asset stat arb using PCA
- Kalman filters for dynamic hedge ratios
- Risk management for stat arb portfolios

**Why this matters for engineers:**
- Stat arb combines statistics, programming, and trading
- Scalable (can trade hundreds of pairs simultaneously)
- Market-neutral (hedged against market direction)
- Quantifiable edge (statistical relationships)

**Performance Characteristics:**
- **Win Rate**: 55-65% (higher than trend following)
- **Sharpe Ratio**: 2.0-4.0 (excellent risk-adjusted returns)
- **Capacity**: $100M-$1B per strategy (limited scalability)
- **Holding Period**: Hours to weeks

**Key Firms Using Stat Arb:**
- Renaissance Technologies: $60B+ AUM
- Two Sigma: $60B+ AUM
- Citadel: $50B+ AUM
- DE Shaw: $60B+ AUM

---

## Statistical Arbitrage Fundamentals

### What Is Statistical Arbitrage?

**Definition**: Trading strategy that exploits mean-reverting statistical relationships between securities.

**Core Concept**:
1. Identify securities with historical statistical relationship
2. Wait for relationship to diverge (price spread widens)
3. Trade the spread (long undervalued, short overvalued)
4. Profit when spread reverts to historical mean

**Types of Statistical Relationships:**
- **Correlation**: Assets move together (0.7+ correlation)
- **Cointegration**: Spreads are stationary (stronger than correlation)
- **Beta Relationship**: Asset follows index predictably
- **Factor Exposure**: Assets share common risk factors

\`\`\`python
from dataclasses import dataclass
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.api import OLS
import warnings
warnings.filterwarnings('ignore')

@dataclass
class StatArbPair:
    """
    Represents a statistical arbitrage pair
    """
    asset_1: str
    asset_2: str
    hedge_ratio: float
    half_life: float
    cointegration_pvalue: float
    correlation: float
    spread_mean: float
    spread_std: float
    
    def is_viable(self) -> bool:
        """
        Determine if pair is suitable for trading
        
        Criteria:
        - Cointegration p-value < 0.05
        - Half-life < 60 days (reverts within 2 months)
        - Correlation > 0.7 (moves together)
        """
        return (
            self.cointegration_pvalue < 0.05 and
            self.half_life < 60 and
            self.correlation > 0.7
        )
    
    def quality_score(self) -> float:
        """
        Calculate pair quality score (0-100)
        
        Higher score = better pair
        """
        score = 0
        
        # Cointegration strength (40 points)
        if self.cointegration_pvalue < 0.01:
            score += 40
        elif self.cointegration_pvalue < 0.05:
            score += 20
        
        # Half-life (30 points)
        if self.half_life < 20:
            score += 30
        elif self.half_life < 40:
            score += 20
        elif self.half_life < 60:
            score += 10
        
        # Correlation (30 points)
        if self.correlation > 0.9:
            score += 30
        elif self.correlation > 0.8:
            score += 20
        elif self.correlation > 0.7:
            score += 10
        
        return score

class StatisticalArbitrageAnalyzer:
    """
    Analyze and identify statistical arbitrage opportunities
    """
    
    def __init__(self, lookback_period: int = 252):
        """
        Initialize analyzer
        
        Args:
            lookback_period: Period for calculating statistics (252 = 1 year)
        """
        self.lookback_period = lookback_period
    
    def test_cointegration(self, 
                          prices_1: pd.Series,
                          prices_2: pd.Series) -> Tuple[float, float, float]:
        """
        Test if two price series are cointegrated
        
        Cointegration: Linear combination of non-stationary series is stationary
        I.e., spread = price_1 - β*price_2 is mean-reverting
        
        Args:
            prices_1: First price series
            prices_2: Second price series
            
        Returns:
            (t_statistic, p_value, hedge_ratio)
        """
        # Engle-Granger cointegration test
        # Step 1: Run regression price_1 = α + β*price_2
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        X = prices_2.values.reshape(-1, 1)
        y = prices_1.values
        model.fit(X, y)
        
        hedge_ratio = model.coef_[0]
        
        # Step 2: Calculate spread
        spread = prices_1 - hedge_ratio * prices_2
        
        # Step 3: Test if spread is stationary (ADF test)
        adf_result = adfuller(spread, maxlag=1)
        t_statistic = adf_result[0]
        p_value = adf_result[1]
        
        return t_statistic, p_value, hedge_ratio
    
    def calculate_hedge_ratio(self,
                             prices_1: pd.Series,
                             prices_2: pd.Series,
                             method: str = 'ols') -> float:
        """
        Calculate optimal hedge ratio
        
        Methods:
        - 'ols': Ordinary Least Squares regression
        - 'tls': Total Least Squares (orthogonal regression)
        - 'rolling': Rolling regression (time-varying)
        
        Args:
            prices_1: First price series
            prices_2: Second price series
            method: Calculation method
            
        Returns:
            Hedge ratio (β)
        """
        if method == 'ols':
            # Standard OLS: price_1 = α + β*price_2
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            X = prices_2.values.reshape(-1, 1)
            y = prices_1.values
            model.fit(X, y)
            return model.coef_[0]
        
        elif method == 'tls':
            # Total Least Squares (minimize perpendicular distance)
            # More robust when both series have noise
            from scipy.linalg import svd
            X = np.column_stack([prices_1, prices_2])
            X_centered = X - X.mean(axis=0)
            U, s, Vt = svd(X_centered)
            
            # Hedge ratio from eigenvector
            hedge_ratio = -Vt[0, 0] / Vt[0, 1]
            return hedge_ratio
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate half-life of mean reversion for spread
        
        Args:
            spread: Spread time series
            
        Returns:
            Half-life in periods
        """
        # Ornstein-Uhlenbeck process
        lag = spread.shift(1).dropna()
        delta = spread.diff().dropna()
        
        # Align series
        lag = lag[delta.index]
        
        # Regression: Δspread = λ*spread + ε
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(lag.values.reshape(-1, 1), delta.values)
        
        lambda_param = model.coef_[0]
        
        if lambda_param >= 0:
            return np.inf  # No mean reversion
        
        half_life = -np.log(2) / lambda_param
        return half_life
    
    def find_pairs(self,
                   prices: pd.DataFrame,
                   min_correlation: float = 0.7,
                   max_pairs: int = 50) -> List[StatArbPair]:
        """
        Screen universe for cointegrated pairs
        
        Args:
            prices: DataFrame with asset prices (columns = assets)
            min_correlation: Minimum correlation threshold
            max_pairs: Maximum pairs to return
            
        Returns:
            List of viable pairs sorted by quality
        """
        symbols = prices.columns
        pairs = []
        
        # Test all combinations
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                # Quick correlation filter
                corr = prices[sym1].corr(prices[sym2])
                if corr < min_correlation:
                    continue
                
                # Test cointegration
                try:
                    t_stat, p_val, hedge_ratio = self.test_cointegration(
                        prices[sym1], prices[sym2]
                    )
                    
                    if p_val > 0.05:
                        continue  # Not cointegrated
                    
                    # Calculate spread characteristics
                    spread = prices[sym1] - hedge_ratio * prices[sym2]
                    half_life = self.calculate_half_life(spread)
                    
                    pair = StatArbPair(
                        asset_1=sym1,
                        asset_2=sym2,
                        hedge_ratio=hedge_ratio,
                        half_life=half_life,
                        cointegration_pvalue=p_val,
                        correlation=corr,
                        spread_mean=spread.mean(),
                        spread_std=spread.std()
                    )
                    
                    if pair.is_viable():
                        pairs.append(pair)
                
                except Exception as e:
                    continue
        
        # Sort by quality and return top pairs
        pairs.sort(key=lambda p: p.quality_score(), reverse=True)
        return pairs[:max_pairs]

# Example usage
if __name__ == "__main__":
    # Simulate correlated price series
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    
    # Create cointegrated pair
    common_factor = np.random.randn(len(dates)).cumsum()
    prices_A = pd.Series(100 + common_factor + np.random.randn(len(dates)), index=dates)
    prices_B = pd.Series(50 + 0.5*common_factor + np.random.randn(len(dates)), index=dates)
    
    # Analyze pair
    analyzer = StatisticalArbitrageAnalyzer()
    t_stat, p_val, hedge_ratio = analyzer.test_cointegration(prices_A, prices_B)
    
    print("\\n=== Cointegration Analysis ===")
    print(f"Hedge Ratio: {hedge_ratio:.4f}")
    print(f"Cointegration p-value: {p_val:.4f}")
    print(f"Cointegrated: {p_val < 0.05}")
    
    # Calculate half-life
    spread = prices_A - hedge_ratio * prices_B
    half_life = analyzer.calculate_half_life(spread)
    print(f"Half-Life: {half_life:.1f} days")
\`\`\`

---

## Z-Score Pairs Trading Strategy

### Classic Pairs Trading Approach

**Entry Rules:**
- Long spread when z-score < -2 (undervalued)
- Short spread when z-score > +2 (overvalued)

**Exit Rules:**
- Exit at z-score = 0 (mean)
- Stop loss at z-score = ±3 (extreme divergence)

\`\`\`python
class PairsTradingStrategy:
    """
    Z-score based pairs trading strategy
    
    Classic approach:
    1. Calculate spread = price_A - β*price_B
    2. Calculate z-score = (spread - mean) / std
    3. Trade when |z-score| > 2
    """
    
    def __init__(self,
                 pair: StatArbPair,
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5,
                 stop_loss_threshold: float = 3.0):
        """
        Initialize pairs trading strategy
        
        Args:
            pair: Statistical arbitrage pair
            entry_threshold: Z-score for entry (±2.0)
            exit_threshold: Z-score for exit (±0.5)
            stop_loss_threshold: Z-score for stop loss (±3.0)
        """
        self.pair = pair
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.lookback = 60  # Days for rolling statistics
    
    def calculate_spread(self,
                        prices_1: pd.Series,
                        prices_2: pd.Series) -> pd.Series:
        """Calculate price spread"""
        return prices_1 - self.pair.hedge_ratio * prices_2
    
    def calculate_zscore(self, spread: pd.Series) -> pd.Series:
        """
        Calculate rolling z-score
        
        Z-score = (spread - rolling_mean) / rolling_std
        """
        rolling_mean = spread.rolling(window=self.lookback).mean()
        rolling_std = spread.rolling(window=self.lookback).std()
        
        zscore = (spread - rolling_mean) / rolling_std
        return zscore
    
    def generate_signals(self,
                        prices_1: pd.Series,
                        prices_2: pd.Series) -> pd.DataFrame:
        """
        Generate trading signals
        
        Returns:
            DataFrame with spread, z-score, and signals
        """
        df = pd.DataFrame(index=prices_1.index)
        df['price_1'] = prices_1
        df['price_2'] = prices_2
        df['spread'] = self.calculate_spread(prices_1, prices_2)
        df['zscore'] = self.calculate_zscore(df['spread'])
        
        # Initialize signals
        df['signal'] = 0
        df['position'] = 0
        
        position = 0  # 0 = flat, 1 = long spread, -1 = short spread
        
        for i in range(self.lookback, len(df)):
            z = df['zscore'].iloc[i]
            
            # Entry signals
            if position == 0:
                if z < -self.entry_threshold:
                    # Long spread: buy asset_1, sell asset_2
                    df.loc[df.index[i], 'signal'] = 1
                    position = 1
                
                elif z > self.entry_threshold:
                    # Short spread: sell asset_1, buy asset_2
                    df.loc[df.index[i], 'signal'] = -1
                    position = -1
            
            # Exit signals
            elif position == 1:  # Long spread
                if z > -self.exit_threshold or z < -self.stop_loss_threshold:
                    df.loc[df.index[i], 'signal'] = 0
                    position = 0
            
            elif position == -1:  # Short spread
                if z < self.exit_threshold or z > self.stop_loss_threshold:
                    df.loc[df.index[i], 'signal'] = 0
                    position = 0
            
            df.loc[df.index[i], 'position'] = position
        
        return df
    
    def calculate_position_sizes(self,
                                capital: float,
                                price_1: float,
                                price_2: float,
                                risk_per_trade: float = 0.02) -> dict:
        """
        Calculate position sizes for both legs
        
        Goal: Dollar-neutral (equal long and short)
        
        Args:
            capital: Total capital
            price_1: Current price of asset 1
            price_2: Current price of asset 2
            risk_per_trade: Risk per trade (2% default)
            
        Returns:
            Position sizes for each asset
        """
        # Risk amount
        risk_amount = capital * risk_per_trade
        
        # For dollar-neutral spread:
        # Long $X of asset_1, Short $X of asset_2
        
        # Calculate shares for equal dollar exposure
        dollars_per_leg = capital * 0.10  # 10% of capital per leg
        
        shares_asset_1 = int(dollars_per_leg / price_1)
        shares_asset_2 = int(dollars_per_leg / price_2 * self.pair.hedge_ratio)
        
        return {
            'asset_1_shares': shares_asset_1,
            'asset_2_shares': shares_asset_2,
            'asset_1_value': shares_asset_1 * price_1,
            'asset_2_value': shares_asset_2 * price_2,
            'net_exposure': shares_asset_1 * price_1 - shares_asset_2 * price_2
        }
    
    def backtest(self,
                prices_1: pd.Series,
                prices_2: pd.Series,
                initial_capital: float = 100000) -> dict:
        """
        Backtest pairs trading strategy
        
        Args:
            prices_1: Price series for asset 1
            prices_2: Price series for asset 2
            initial_capital: Starting capital
            
        Returns:
            Performance metrics
        """
        df = self.generate_signals(prices_1, prices_2)
        
        # Calculate returns for each leg
        returns_1 = prices_1.pct_change()
        returns_2 = prices_2.pct_change()
        
        # Spread returns (long asset_1, short asset_2)
        spread_returns = returns_1 - self.pair.hedge_ratio * returns_2
        
        # Strategy returns
        strategy_returns = df['position'].shift(1) * spread_returns
        
        # Transaction costs (10 bps per trade, both legs)
        trades = df['position'].diff().abs()
        transaction_costs = trades * 0.001  # 10 bps
        strategy_returns_net = strategy_returns - transaction_costs
        
        # Performance metrics
        cumulative = (1 + strategy_returns_net).cumprod()
        total_return = cumulative.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(returns_1)) - 1
        
        volatility = strategy_returns_net.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        winning_trades = (strategy_returns_net > 0).sum()
        losing_trades = (strategy_returns_net < 0).sum()
        win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': trades.sum(),
            'avg_holding_period': len(df) / trades.sum() if trades.sum() > 0 else 0
        }
\`\`\`

---

## Kalman Filter for Dynamic Hedge Ratios

### Why Kalman Filters?

Standard pairs trading assumes constant hedge ratio. In reality, relationships drift over time. Kalman filters provide dynamic, time-varying hedge ratios.

\`\`\`python
from pykalman import KalmanFilter

class KalmanPairsTrading:
    """
    Pairs trading with Kalman Filter for dynamic hedge ratio
    
    Advantage: Adapts to changing relationships
    Disadvantage: More complex, can overfit
    """
    
    def __init__(self):
        self.kalman_filter = None
        self.state_means = None
        self.state_covs = None
    
    def fit_kalman_filter(self,
                         prices_1: pd.Series,
                         prices_2: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Fit Kalman filter to estimate time-varying hedge ratio
        
        State space model:
        - State: hedge_ratio (β)
        - Observation: price_1 = β * price_2 + noise
        
        Args:
            prices_1: First price series
            prices_2: Second price series
            
        Returns:
            (hedge_ratios, spread_stds) time series
        """
        # Observation matrix
        obs_mat = np.expand_dims(prices_2.values, axis=1)
        
        # Initialize Kalman filter
        kf = KalmanFilter(
            n_dim_obs=1,  # 1D observation (price_1)
            n_dim_state=1,  # 1D state (hedge_ratio)
            initial_state_mean=0,
            initial_state_covariance=1,
            transition_matrices=np.eye(1),  # Random walk for hedge ratio
            observation_matrices=obs_mat,
            observation_covariance=1,
            transition_covariance=0.01  # Small changes in hedge ratio
        )
        
        # Fit and predict
        state_means, state_covs = kf.filter(prices_1.values)
        
        # Extract hedge ratios and uncertainties
        hedge_ratios = pd.Series(state_means.flatten(), index=prices_1.index)
        hedge_ratio_stds = pd.Series(np.sqrt(state_covs.flatten()), index=prices_1.index)
        
        self.kalman_filter = kf
        self.state_means = state_means
        self.state_covs = state_covs
        
        return hedge_ratios, hedge_ratio_stds
    
    def calculate_dynamic_spread(self,
                                prices_1: pd.Series,
                                prices_2: pd.Series,
                                hedge_ratios: pd.Series) -> pd.Series:
        """
        Calculate spread using time-varying hedge ratios
        
        Args:
            prices_1: First price series
            prices_2: Second price series
            hedge_ratios: Time-varying hedge ratios
            
        Returns:
            Dynamic spread
        """
        spread = prices_1 - hedge_ratios * prices_2
        return spread
\`\`\`

---

## Multi-Asset Statistical Arbitrage

### PCA-Based Stat Arb

Principal Component Analysis identifies common factors across assets. Trade divergences from factor-implied prices.

\`\`\`python
from sklearn.decomposition import PCA

class PCAStatArb:
    """
    Multi-asset statistical arbitrage using PCA
    
    Approach:
    1. Extract principal components (common factors)
    2. Calculate residuals (idiosyncratic risk)
    3. Trade when residuals are extreme (2+ std devs)
    """
    
    def __init__(self, n_components: int = 5):
        """
        Initialize PCA stat arb
        
        Args:
            n_components: Number of principal components
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.fitted = False
    
    def fit(self, returns: pd.DataFrame):
        """
        Fit PCA model to return data
        
        Args:
            returns: DataFrame of returns (assets as columns)
        """
        self.pca.fit(returns.fillna(0))
        self.fitted = True
        
        # Store factor loadings
        self.factor_loadings = pd.DataFrame(
            self.pca.components_.T,
            index=returns.columns,
            columns=[f'PC{i+1}' for i in range(self.n_components)]
        )
    
    def calculate_residuals(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate residuals (unexplained by factors)
        
        Residual = Actual Return - Factor-Implied Return
        
        Args:
            returns: DataFrame of returns
            
        Returns:
            DataFrame of residuals
        """
        if not self.fitted:
            raise ValueError("Must fit model first")
        
        # Transform to factor space
        factor_returns = self.pca.transform(returns.fillna(0))
        
        # Reconstruct returns from factors
        reconstructed = self.pca.inverse_transform(factor_returns)
        reconstructed_df = pd.DataFrame(
            reconstructed,
            index=returns.index,
            columns=returns.columns
        )
        
        # Calculate residuals
        residuals = returns - reconstructed_df
        
        return residuals
    
    def generate_signals(self,
                        returns: pd.DataFrame,
                        threshold: float = 2.0) -> pd.DataFrame:
        """
        Generate trading signals based on residuals
        
        Signal:
        - Long if residual < -2σ (underperformed factors)
        - Short if residual > +2σ (outperformed factors)
        
        Args:
            returns: DataFrame of returns
            threshold: Z-score threshold
            
        Returns:
            DataFrame of signals (-1, 0, 1)
        """
        residuals = self.calculate_residuals(returns)
        
        # Standardize residuals
        residual_mean = residuals.rolling(window=60).mean()
        residual_std = residuals.rolling(window=60).std()
        z_scores = (residuals - residual_mean) / residual_std
        
        # Generate signals
        signals = pd.DataFrame(0, index=returns.index, columns=returns.columns)
        signals[z_scores < -threshold] = 1  # Long
        signals[z_scores > threshold] = -1  # Short
        
        return signals
\`\`\`

---

## Risk Management for Stat Arb

### Portfolio-Level Risk

\`\`\`python
class StatArbRiskManager:
    """
    Risk management for statistical arbitrage portfolio
    """
    
    def __init__(self, capital: float):
        self.capital = capital
        self.max_pairs = 50  # Diversification limit
        self.max_position_size = 0.05  # 5% per pair
        self.max_portfolio_gross = 2.0  # 200% gross exposure
        self.max_portfolio_net = 0.10  # 10% net exposure
    
    def calculate_pair_correlation_matrix(self,
                                         pairs: List[StatArbPair],
                                         returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation between pair returns
        
        Important: Pairs may be correlated even if dollar-neutral
        """
        pair_returns = pd.DataFrame()
        
        for pair in pairs:
            spread_returns = (
                returns[pair.asset_1] - 
                pair.hedge_ratio * returns[pair.asset_2]
            )
            pair_returns[f"{pair.asset_1}_{pair.asset_2}"] = spread_returns
        
        correlation_matrix = pair_returns.corr()
        return correlation_matrix
    
    def optimize_pair_weights(self,
                             pairs: List[StatArbPair],
                             correlation_matrix: pd.DataFrame) -> dict:
        """
        Optimize position sizes considering correlation
        
        Goal: Maximize diversification, minimize correlation
        """
        # Equal weight as baseline
        equal_weight = 1.0 / len(pairs)
        
        # Adjust for quality and correlation
        weights = {}
        for i, pair in enumerate(pairs):
            quality_score = pair.quality_score() / 100
            
            # Reduce weight if highly correlated with other pairs
            avg_correlation = correlation_matrix.iloc[i].abs().mean()
            correlation_penalty = 1 - (avg_correlation - 0.5)
            
            weights[f"{pair.asset_1}_{pair.asset_2}"] = (
                equal_weight * quality_score * correlation_penalty
            )
        
        # Normalize
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
\`\`\`

---

## Real-World Examples

### Renaissance Technologies - Medallion Fund

**Performance**: 66% annual return (before fees) since 1988

**Approach**:
- Short-term stat arb (hold < 1 day)
- Thousands of positions simultaneously
- High turnover (100+ trades per day)
- Machine learning + traditional stat arb

**Key Success Factors**:
1. PhD-heavy team (math, physics, CS)
2. Proprietary data and signals
3. Ultra-low latency execution
4. Capacity limited to $10B

### Two Sigma

**AUM**: $60B+

**Approach**:
- Multi-asset stat arb
- Machine learning driven
- Alternative data integration
- Longer holding periods (days to weeks) vs Renaissance

**Technology**:
- Cloud-based infrastructure
- Massive compute for backtesting
- Real-time risk management

---

## Summary and Key Takeaways

**Statistical Arbitrage Works When:**
- Strong statistical relationships exist
- Relationships are cointegrated (not just correlated)
- Half-life < 60 days (mean reversion occurs)
- Market-neutral (hedged against market moves)

**Stat Arb Fails When:**
- Relationships break permanently (structural change)
- Crowding (everyone trades same pairs)
- Execution costs too high (high turnover)
- Correlation≠cointegration (correlations can diverge)

**Best Practices:**
1. **Test Cointegration**: Use ADF test, require p < 0.05
2. **Calculate Half-Life**: Ensure < 60 days
3. **Dynamic Hedge Ratios**: Use Kalman filter for time-varying β
4. **Diversify**: 20-50 pairs minimum
5. **Monitor Relationships**: Relationships can break

**Comparison to Other Strategies:**

| Metric | Stat Arb | Trend Following | Mean Reversion |
|--------|----------|----------------|----------------|
| Win Rate | 55-65% | 35-45% | 55-70% |
| Sharpe Ratio | 2-4 | 1-1.5 | 1-2 |
| Market Directional | No | Yes | Somewhat |
| Capacity | $100M-$1B | $10B+ | $100M-$500M |
| Holding Period | Days-Weeks | Weeks-Months | Hours-Days |

**Next Section**: Pairs Trading (deep dive into specific pair selection and execution)
`,
};

