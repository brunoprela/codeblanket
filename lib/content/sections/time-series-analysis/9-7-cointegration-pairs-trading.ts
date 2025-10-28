export const cointegrationPairsTrading = {
  title: 'Cointegration and Pairs Trading',
  slug: 'cointegration-pairs-trading',
  description:
    'Master cointegration theory and build profitable pairs trading strategies',
  content: `
# Cointegration and Pairs Trading

## Introduction: Beyond Correlation

**Cointegration** is a statistical property where two non-stationary series maintain a stable long-run relationship.

**Why cointegration matters in finance:**
- Pairs trading (market-neutral strategy)
- Statistical arbitrage opportunities
- Portfolio construction and risk management
- Understanding economic relationships (e.g., spot vs futures)
- Mean reversion trading

**What you'll learn:**
- Difference between correlation and cointegration
- Engle-Granger two-step test
- Johansen test for multiple cointegration
- Building pairs trading strategies
- Risk management and position sizing
- Real-world implementation

**Key insight:** Correlation measures co-movement, cointegration measures long-run equilibrium!

---

## Cointegration vs Correlation

### The Key Difference

**Correlation:**
- Measures linear relationship between returns
- Can be high even if no long-run relationship
- Can change quickly

**Cointegration:**
- Measures stable long-run relationship between LEVELS
- Implies predictable mean reversion
- More stable for trading

\`\`\`python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import matplotlib.pyplot as plt

def demonstrate_cointegration_vs_correlation():
    """
    Show difference between correlation and cointegration.
    """
    np.random.seed(42)
    n = 500
    
    # Example 1: High correlation, NO cointegration (two random walks)
    rw1 = np.cumsum(np.random.randn(n))
    rw2 = np.cumsum(np.random.randn(n))
    
    corr_no_coint = np.corrcoef(rw1, rw2)[0,1]
    
    # Example 2: Cointegrated pair
    # X is random walk, Y = 2*X + stationary error
    X = np.cumsum(np.random.randn(n)) + 100
    Y = 2 * X + np.random.randn(n) * 5
    
    corr_coint = np.corrcoef(X, Y)[0,1]
    
    # Test cointegration
    _, pvalue_no_coint, _ = coint(rw1, rw2)
    _, pvalue_coint, _ = coint(X, Y)
    
    print("=== Cointegration vs Correlation ===\\n")
    print("Example 1: Two random walks")
    print(f"  Correlation: {corr_no_coint:.4f}")
    print(f"  Cointegration p-value: {pvalue_no_coint:.4f}")
    print(f"  Cointegrated: {'NO' if pvalue_no_coint > 0.05 else 'YES'}\\n")
    
    print("Example 2: Cointegrated pair")
    print(f"  Correlation: {corr_coint:.4f}")
    print(f"  Cointegration p-value: {pvalue_coint:.4f}")
    print(f"  Cointegrated: {'YES' if pvalue_coint < 0.05 else 'NO'}\\n")
    
    # The spread
    spread = Y - 2*X  # Should be stationary
    adf_result = adfuller(spread)
    
    print("Spread (Y - 2*X) stationarity:")
    print(f"  ADF p-value: {adf_result[1]:.4f}")
    print(f"  Stationary: {'YES' if adf_result[1] < 0.05 else 'NO'}")
    
    return X, Y, spread

X, Y, spread = demonstrate_cointegration_vs_correlation()
\`\`\`

---

## Engle-Granger Two-Step Test

### Step 1: Estimate Cointegrating Relationship

Run regression: $Y_t = \\alpha + \\beta X_t + \\epsilon_t$

### Step 2: Test Residuals for Stationarity

If $\\epsilon_t$ is stationary (I(0)), then X and Y are cointegrated.

\`\`\`python
class EngleGrangerTest:
    """
    Engle-Granger cointegration test implementation.
    """
    
    def __init__(self):
        self.beta = None
        self.alpha = None
        self.residuals = None
        self.pvalue = None
        
    def test(self, y: pd.Series, x: pd.Series) -> dict:
        """
        Perform Engle-Granger cointegration test.
        
        Args:
            y: Dependent variable (I(1))
            x: Independent variable (I(1))
            
        Returns:
            Test results
        """
        # Step 1: OLS regression
        from sklearn.linear_model import LinearRegression
        
        X_arr = x.values.reshape(-1, 1)
        y_arr = y.values
        
        model = LinearRegression()
        model.fit(X_arr, y_arr)
        
        self.beta = model.coef_[0]
        self.alpha = model.intercept_
        
        # Residuals (spread)
        self.residuals = y_arr - (self.alpha + self.beta * x.values)
        
        # Step 2: ADF test on residuals
        adf_result = adfuller(self.residuals, maxlag=1, regression='c')
        
        self.pvalue = adf_result[1]
        
        # Critical values for cointegration test (different from standard ADF!)
        # Approximate MacKinnon (1991) critical values for n=100
        critical_values_coint = {
            '1%': -3.90,
            '5%': -3.34,
            '10%': -3.04
        }
        
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'adf_statistic': adf_result[0],
            'pvalue': self.pvalue,
            'critical_values': critical_values_coint,
            'cointegrated_5pct': adf_result[0] < critical_values_coint['5%'],
            'spread_mean': np.mean(self.residuals),
            'spread_std': np.std(self.residuals),
            'interpretation': f"""
Engle-Granger Cointegration Test:

Cointegrating equation: Y = {self.alpha:.4f} + {self.beta:.4f} * X + ε

ADF statistic: {adf_result[0]:.4f}
P-value: {self.pvalue:.4f}

Result: {'COINTEGRATED (reject H0)' if adf_result[0] < critical_values_coint['5%'] else 'NOT COINTEGRATED'}

Spread: μ={np.mean(self.residuals):.4f}, σ={np.std(self.residuals):.4f}
            """
        }

# Example
print("\\n=== Engle-Granger Test Example ===\\n")

eg_test = EngleGrangerTest()
result = eg_test.test(pd.Series(Y), pd.Series(X))
print(result['interpretation'])
\`\`\`

---

## Johansen Test: Multiple Cointegration

For N > 2 variables, use Johansen test to find:
1. Number of cointegrating relationships
2. Cointegrating vectors

\`\`\`python
def johansen_test(data: pd.DataFrame, 
                 det_order: int = 0,
                 k_ar_diff: int = 1) -> dict:
    """
    Johansen cointegration test for multiple time series.
    
    Args:
        data: DataFrame with I(1) variables
        det_order: -1 (no constant), 0 (constant in coint eq), 
                   1 (constant + trend)
        k_ar_diff: Lags of differences in VAR
        
    Returns:
        Test results
    """
    result = coint_johansen(data.values, det_order=det_order, k_ar_diff=k_ar_diff)
    
    # Trace statistic tests
    n_coint = 0
    for i, (trace_stat, crit_val) in enumerate(zip(result.lr1, result.cvt)):
        if trace_stat > crit_val[1]:  # 5% significance
            n_coint = i + 1
    
    return {
        'n_cointegrating_relationships': n_coint,
        'trace_statistics': result.lr1,
        'critical_values_5pct': result.cvt[:, 1],
        'eigenvalues': result.eig,
        'eigenvectors': result.evec,
        'interpretation': f"""
Johansen Cointegration Test:

Number of cointegrating relationships: {n_coint}

Trace statistics: {result.lr1[:3]}
Critical values (5%): {result.cvt[:3, 1]}

{'At least one cointegrating relationship found!' if n_coint > 0 else 'No cointegration detected'}
        """
    }

# Example with 3 variables
print("\\n=== Johansen Test Example ===\\n")

# Create 3-variable cointegrated system
Z1 = X
Z2 = Y
Z3 = 1.5*X - 0.5*Y + np.random.randn(len(X)) * 3

data_3var = pd.DataFrame({'Z1': Z1, 'Z2': Z2, 'Z3': Z3})
johansen_result = johansen_test(data_3var)
print(johansen_result['interpretation'])
\`\`\`

---

## Pairs Trading Strategy

### Strategy Framework

1. **Pair Selection**: Find cointegrated pairs
2. **Spread Construction**: Calculate spread = Y - β*X
3. **Signal Generation**: Trade when spread deviates from mean
4. **Entry**: Buy (sell) when spread z-score < -2 (> +2)
5. **Exit**: Close when spread reverts to mean

\`\`\`python
class PairsTradingStrategy:
    """
    Complete pairs trading strategy implementation.
    """
    
    def __init__(self, entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5):
        """
        Initialize strategy.
        
        Args:
            entry_threshold: Z-score threshold for entry (e.g., ±2)
            exit_threshold: Z-score threshold for exit (e.g., ±0.5)
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.hedge_ratio = None
        self.spread_mean = None
        self.spread_std = None
        
    def fit(self, y_train: pd.Series, x_train: pd.Series):
        """
        Estimate hedge ratio and spread statistics from training data.
        
        Args:
            y_train: Price series for asset Y
            x_train: Price series for asset X
        """
        from sklearn.linear_model import LinearRegression
        
        # Estimate hedge ratio
        X_arr = x_train.values.reshape(-1, 1)
        y_arr = y_train.values
        
        model = LinearRegression()
        model.fit(X_arr, y_arr)
        
        self.hedge_ratio = model.coef_[0]
        
        # Calculate spread
        spread = y_arr - self.hedge_ratio * x_train.values
        
        # Spread statistics
        self.spread_mean = np.mean(spread)
        self.spread_std = np.std(spread)
        
    def generate_signals(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        """
        Generate trading signals.
        
        Args:
            y: Price series for asset Y
            x: Price series for asset X
            
        Returns:
            DataFrame with signals and positions
        """
        # Calculate spread
        spread = y.values - self.hedge_ratio * x.values
        
        # Z-score
        z_score = (spread - self.spread_mean) / self.spread_std
        
        # Initialize positions
        position = np.zeros(len(y))
        
        # State: 0 = flat, 1 = long spread, -1 = short spread
        state = 0
        
        for i in range(1, len(z_score)):
            # Entry signals
            if state == 0:
                if z_score[i] < -self.entry_threshold:
                    # Buy spread: Buy Y, Sell X
                    state = 1
                    position[i] = 1
                elif z_score[i] > self.entry_threshold:
                    # Sell spread: Sell Y, Buy X
                    state = -1
                    position[i] = -1
            
            # Exit signals
            elif state == 1:  # Currently long
                if z_score[i] > -self.exit_threshold:
                    state = 0
                    position[i] = 0
                else:
                    position[i] = 1
            
            elif state == -1:  # Currently short
                if z_score[i] < self.exit_threshold:
                    state = 0
                    position[i] = 0
                else:
                    position[i] = -1
        
        results = pd.DataFrame({
            'Y_price': y.values,
            'X_price': x.values,
            'spread': spread,
            'z_score': z_score,
            'position': position
        }, index=y.index)
        
        return results
    
    def backtest(self, signals: pd.DataFrame,
                y_prices: pd.Series,
                x_prices: pd.Series) -> dict:
        """
        Backtest strategy and calculate performance.
        
        Args:
            signals: DataFrame from generate_signals()
            y_prices: Prices for Y
            x_prices: Prices for X
            
        Returns:
            Performance metrics
        """
        # Calculate returns
        y_returns = y_prices.pct_change()
        x_returns = x_prices.pct_change()
        
        # Portfolio returns (dollar-neutral)
        # Long spread: +$1 in Y, -$hedge_ratio in X
        # Short spread: -$1 in Y, +$hedge_ratio in X
        
        portfolio_returns = (
            signals['position'].shift(1) * y_returns -
            signals['position'].shift(1) * self.hedge_ratio * x_returns
        )
        
        # Cumulative returns
        cum_returns = (1 + portfolio_returns).cumprod()
        
        # Performance metrics
        total_return = cum_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Max drawdown
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = (portfolio_returns[portfolio_returns != 0] > 0).sum()
        total_trades = (portfolio_returns != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_trades': total_trades,
            'cum_returns': cum_returns,
            'portfolio_returns': portfolio_returns
        }

# Example backtest
print("\\n=== Pairs Trading Backtest ===\\n")

# Split data
split = int(len(X) * 0.5)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

X_train_series = pd.Series(X_train)
Y_train_series = pd.Series(Y_train)
X_test_series = pd.Series(X_test, index=range(len(X_test)))
Y_test_series = pd.Series(Y_test, index=range(len(Y_test)))

# Initialize and fit strategy
strategy = PairsTradingStrategy(entry_threshold=2.0, exit_threshold=0.5)
strategy.fit(Y_train_series, X_train_series)

print(f"Hedge ratio: {strategy.hedge_ratio:.4f}")
print(f"Spread mean: {strategy.spread_mean:.4f}")
print(f"Spread std: {strategy.spread_std:.4f}\\n")

# Generate signals on test data
signals = strategy.generate_signals(Y_test_series, X_test_series)

# Backtest
performance = strategy.backtest(signals, Y_test_series, X_test_series)

print("Performance Metrics:")
print(f"  Total Return: {performance['total_return']*100:.2f}%")
print(f"  Annual Return: {performance['annual_return']*100:.2f}%")
print(f"  Annual Volatility: {performance['annual_volatility']*100:.2f}%")
print(f"  Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
print(f"  Max Drawdown: {performance['max_drawdown']*100:.2f}%")
print(f"  Win Rate: {performance['win_rate']*100:.1f}%")
print(f"  Number of Trades: {performance['n_trades']}")
\`\`\`

---

## Risk Management

### Position Sizing

\`\`\`python
def calculate_position_size(spread_zscore: float,
                           portfolio_value: float,
                           max_risk_per_trade: float = 0.02,
                           spread_volatility: float = 0.10) -> dict:
    """
    Kelly criterion-based position sizing.
    
    Args:
        spread_zscore: Current z-score of spread
        portfolio_value: Total portfolio value
        max_risk_per_trade: Maximum risk per trade (e.g., 2%)
        spread_volatility: Volatility of spread
        
    Returns:
        Position sizes
    """
    # Kelly fraction
    win_prob = 0.55  # Estimate from backtests
    avg_win = 0.02   # 2% per winning trade
    avg_loss = 0.015  # 1.5% per losing trade
    
    kelly_fraction = (win_prob / avg_loss) - ((1 - win_prob) / avg_win)
    kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
    
    # Position size
    position_value = portfolio_value * kelly_fraction * 0.5  # Half-Kelly
    
    # Adjust for z-score (higher z-score = higher conviction)
    conviction_adj = min(abs(spread_zscore) / 3, 1.0)
    adjusted_position = position_value * conviction_adj
    
    # Risk limit
    max_position = portfolio_value * max_risk_per_trade / spread_volatility
    final_position = min(adjusted_position, max_position)
    
    return {
        'kelly_fraction': kelly_fraction,
        'position_value': final_position,
        'conviction_adjustment': conviction_adj,
        'risk_percentage': (final_position * spread_volatility / portfolio_value) * 100
    }

# Example
position = calculate_position_size(
    spread_zscore=-2.5,
    portfolio_value=1_000_000,
    max_risk_per_trade=0.02
)

print("\\n=== Position Sizing ===")
print(f"Kelly fraction: {position['kelly_fraction']:.2%}")
print(f"Position value: \\$\{position['position_value']:,.0f}")
print(f"Risk: {position['risk_percentage']:.2f}% of portfolio")
\`\`\`

---

## Summary

**Key Takeaways:**1. **Cointegration ≠ Correlation**: Long-run equilibrium vs co-movement
2. **Engle-Granger**: Two-step test for pair cointegration
3. **Johansen**: Multiple cointegration relationships
4. **Pairs Trading**: Market-neutral, mean-reversion strategy
5. **Risk Management**: Position sizing, stop-losses, portfolio limits

**Next:** Vector Autoregression for multivariate modeling!
`,
};
