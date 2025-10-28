export const riskManagementPositionSizing = {
  title: 'Risk Management & Position Sizing',
  id: 'risk-management-position-sizing',
  content: `
# Risk Management & Position Sizing

## Introduction

Risk management is **the most important** aspect of trading—more important than strategy selection, market analysis, or timing. You can have a mediocre strategy with excellent risk management and succeed. But even the best strategy with poor risk management will fail.

**The Statistics Are Brutal**:
- 90% of retail traders lose money within their first year
- 95% of day traders lose money over 3 years  
- The average lifespan of a prop trader is 4 months
- #1 cause of failure: **Poor risk management**, not bad strategies

**The Mathematics**:
- Lose 50% → Need +100% to recover
- Lose 75% → Need +300% to recover
- Lose 90% → Need +900% to recover

One bad day can wipe out months of gains. Risk management prevents catastrophic losses.

By the end of this section, you'll master:
- Position sizing methods (Fixed fraction, Kelly criterion, volatility-based)
- Stop loss strategies (Fixed, ATR-based, trailing, time-based)
- Portfolio-level risk management (correlation, heat, exposure limits)
- Value at Risk (VaR) and Conditional VaR (CVaR)
- Drawdown management and capital preservation
- Dynamic risk adjustment based on performance

---

## Position Sizing Fundamentals

### The 2% Rule

The **2% rule** is the foundation: Never risk more than 2% of capital on a single trade.

**Why 2%?**
- 50 consecutive losses to blow account (astronomically unlikely)
- Allows for long drawdown periods
- Preserves capital during losing streaks
- Standard in prop trading firms

\`\`\`python
"""
Position Sizing Methods
"""

import numpy as np
import pandas as pd

class PositionSizer:
    """
    Complete position sizing toolkit
    """
    
    def __init__(self, capital=100000):
        self.capital = capital
    
    def fixed_fraction (self, risk_per_trade=0.02):
        """
        Fixed Fraction Method
        
        Risk fixed % of capital per trade
        Most common professional approach
        
        Args:
            risk_per_trade: Fraction of capital to risk (default 2%)
        
        Returns:
            Dollar amount to risk
        """
        return self.capital * risk_per_trade
    
    def calculate_shares (self, entry_price, stop_loss_price, risk_amount):
        """
        Calculate number of shares based on stop loss
        
        Args:
            entry_price: Entry price per share
            stop_loss_price: Stop loss price per share
            risk_amount: Total $ amount to risk
        
        Returns:
            Number of shares to buy
        """
        risk_per_share = abs (entry_price - stop_loss_price)
        if risk_per_share == 0:
            return 0
        
        shares = risk_amount / risk_per_share
        return int (shares)
    
    def kelly_criterion (self, win_rate, avg_win, avg_loss):
        """
        Kelly Criterion - Optimal Growth
        
        Formula: f* = (p*W - q*L) / (W*L)
        Where:
            p = win rate
            q = 1 - p (loss rate)
            W = avg win
            L = avg loss (positive)
        
        Simplified: f* = (p*W - q) / L when W/L = win/loss ratio
        
        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average winning trade ($)
            avg_loss: Average losing trade ($, positive)
        
        Returns:
            Optimal fraction of capital to bet
        """
        if avg_loss == 0:
            return 0
        
        p = win_rate
        q = 1 - win_rate
        W = avg_win
        L = avg_loss
        
        # Kelly formula
        kelly = (p * W - q * L) / (W * L)
        
        # Alternative formula: f = (p*b - q) / b where b = W/L
        # b = W / L
        # kelly_alt = (p * b - q) / b
        
        # Cap at 25% to prevent extreme positions
        kelly = max(0, min (kelly, 0.25))
        
        return kelly
    
    def fractional_kelly (self, win_rate, avg_win, avg_loss, fraction=0.5):
        """
        Fractional Kelly - More Conservative
        
        Use 1/4 to 1/2 of Kelly to reduce volatility
        
        Args:
            fraction: Fraction of Kelly to use (0.25 to 0.5 recommended)
        
        Returns:
            Conservative position size
        """
        full_kelly = self.kelly_criterion (win_rate, avg_win, avg_loss)
        return full_kelly * fraction
    
    def volatility_based_sizing (self, target_volatility=0.15, asset_volatility=0.25):
        """
        Volatility Parity Sizing
        
        Adjust position size to normalize volatility contribution
        
        Args:
            target_volatility: Desired portfolio volatility (e.g., 15%)
            asset_volatility: Asset\'s historical volatility (e.g., 25%)
        
        Returns:
            Position size as fraction of capital
        """
        if asset_volatility == 0:
            return 0
        
        # Position size inversely proportional to volatility
        size = target_volatility / asset_volatility
        
        # Cap at 100%
        return min (size, 1.0)
    
    def atr_based_sizing (self, atr, price, target_risk_pct=0.02, atr_multiplier=2):
        """
        ATR-Based Position Sizing
        
        Use Average True Range for dynamic sizing
        
        Args:
            atr: Average True Range value
            price: Current price
            target_risk_pct: Target risk as % of capital
            atr_multiplier: Stop distance in ATR units (typically 2-3)
        
        Returns:
            Number of shares
        """
        risk_amount = self.capital * target_risk_pct
        stop_distance = atr * atr_multiplier
        
        if stop_distance == 0:
            return 0
        
        shares = risk_amount / stop_distance
        return int (shares)
    
    def compare_methods (self, entry_price=100, stop_price=95, 
                       win_rate=0.55, avg_win=500, avg_loss=300,
                       asset_vol=0.25, atr=2.5):
        """
        Compare all position sizing methods
        
        Returns:
            Dictionary with results from each method
        """
        results = {}
        
        # Fixed fraction (2%)
        risk_2pct = self.fixed_fraction(0.02)
        shares_fixed = self.calculate_shares (entry_price, stop_price, risk_2pct)
        results['fixed_2pct'] = {
            'shares': shares_fixed,
            'dollar_amount': shares_fixed * entry_price,
            'pct_capital': (shares_fixed * entry_price) / self.capital,
            'dollar_risk': risk_2pct
        }
        
        # Kelly criterion
        kelly = self.kelly_criterion (win_rate, avg_win, avg_loss)
        kelly_risk = self.capital * kelly
        shares_kelly = self.calculate_shares (entry_price, stop_price, kelly_risk)
        results['full_kelly'] = {
            'shares': shares_kelly,
            'dollar_amount': shares_kelly * entry_price,
            'pct_capital': kelly,
            'dollar_risk': kelly_risk
        }
        
        # Half Kelly (recommended)
        half_kelly = self.fractional_kelly (win_rate, avg_win, avg_loss, 0.5)
        hk_risk = self.capital * half_kelly
        shares_hk = self.calculate_shares (entry_price, stop_price, hk_risk)
        results['half_kelly'] = {
            'shares': shares_hk,
            'dollar_amount': shares_hk * entry_price,
            'pct_capital': half_kelly,
            'dollar_risk': hk_risk
        }
        
        # Volatility-based
        vol_size = self.volatility_based_sizing(0.15, asset_vol)
        shares_vol = int (self.capital * vol_size / entry_price)
        results['volatility_based'] = {
            'shares': shares_vol,
            'dollar_amount': shares_vol * entry_price,
            'pct_capital': vol_size,
            'dollar_risk': shares_vol * abs (entry_price - stop_price)
        }
        
        # ATR-based
        shares_atr = self.atr_based_sizing (atr, entry_price, 0.02, 2)
        results['atr_based'] = {
            'shares': shares_atr,
            'dollar_amount': shares_atr * entry_price,
            'pct_capital': (shares_atr * entry_price) / self.capital,
            'dollar_risk': shares_atr * atr * 2
        }
        
        return results


# Example Usage
sizer = PositionSizer (capital=100000)

# Compare sizing methods
results = sizer.compare_methods(
    entry_price=100,
    stop_price=95,  # 5% stop loss
    win_rate=0.55,  # 55% win rate
    avg_win=500,
    avg_loss=300,
    asset_vol=0.25,  # 25% volatility
    atr=2.5
)

print("=== Position Sizing Comparison ===")
print(f"{'Method':<20} {'Shares':<8} {'Dollar Amt':<12} {'% Capital':<12} {'$ Risk':<10}")
print("-" * 72)

for method, data in results.items():
    print(f"{method:<20} {data['shares']:<8} \\$\{data['dollar_amount']:<11,.0f} "
          f"{data['pct_capital']:<11.1%} \${data['dollar_risk']:<9,.0f}")

# Kelly details
print(f"\\n=== Kelly Criterion Analysis ===")
kelly_full = sizer.kelly_criterion(0.55, 500, 300)
print(f"Full Kelly: {kelly_full:.1%} of capital")
print(f"Half Kelly: {kelly_full * 0.5:.1%} of capital (RECOMMENDED)")
print(f"Quarter Kelly: {kelly_full * 0.25:.1%} of capital (CONSERVATIVE)")
\`\`\`

---

## Stop Loss Strategies

### Types of Stop Losses

\`\`\`python
"""
Complete Stop Loss Toolkit
"""

class StopLossManager:
    """
    Professional stop loss management
    """
    
    def __init__(self):
        self.stops = {}
    
    def fixed_percentage_stop (self, entry_price, stop_pct=0.05, direction=1):
        """
        Fixed Percentage Stop
        
        Simple but doesn't adapt to volatility
        
        Args:
            entry_price: Entry price
            stop_pct: Stop loss percentage (e.g., 0.05 = 5%)
            direction: 1 for long, -1 for short
        
        Returns:
            Stop loss price
        """
        if direction == 1:  # Long
            return entry_price * (1 - stop_pct)
        else:  # Short
            return entry_price * (1 + stop_pct)
    
    def atr_stop (self, entry_price, atr, multiplier=2, direction=1):
        """
        ATR-Based Stop Loss
        
        Adapts to market volatility
        Industry standard for professional traders
        
        Args:
            entry_price: Entry price
            atr: Average True Range
            multiplier: ATR multiplier (typically 2-3)
                - 2x ATR: Tighter, more signals
                - 3x ATR: Wider, fewer false stops
            direction: 1 for long, -1 for short
        
        Returns:
            Stop loss price
        """
        stop_distance = atr * multiplier
        
        if direction == 1:  # Long
            return entry_price - stop_distance
        else:  # Short
            return entry_price + stop_distance
    
    def trailing_stop (self, current_price, highest_price, trail_pct=0.10, direction=1):
        """
        Trailing Stop Loss
        
        Follows price up (long) or down (short)
        Locks in profits as trade moves favorably
        
        Args:
            current_price: Current market price
            highest_price: Highest price since entry (for long)
            trail_pct: Trailing percentage (e.g., 0.10 = 10%)
            direction: 1 for long, -1 for short
        
        Returns:
            Trailing stop price
        """
        if direction == 1:  # Long
            return highest_price * (1 - trail_pct)
        else:  # Short
            # For short, trail from lowest price
            return highest_price * (1 + trail_pct)
    
    def chandelier_stop (self, high_prices, atr, period=22, multiplier=3):
        """
        Chandelier Stop
        
        Hangs from highest high
        Professional volatility-based trailing stop
        
        Args:
            high_prices: Series of high prices
            atr: Average True Range
            period: Lookback period for highest high
            multiplier: ATR multiplier
        
        Returns:
            Chandelier stop level
        """
        highest_high = high_prices.rolling (period).max()
        stop = highest_high - (atr * multiplier)
        return stop
    
    def time_stop (self, entry_date, current_date, max_holding_days=30):
        """
        Time-Based Stop
        
        Exit after maximum holding period
        Useful for mean-reversion strategies
        
        Args:
            entry_date: Trade entry date
            current_date: Current date
            max_holding_days: Maximum days to hold
        
        Returns:
            True if should exit, False otherwise
        """
        holding_period = (current_date - entry_date).days
        return holding_period >= max_holding_days
    
    def mental_stop (self, entry_price, current_price, max_pain_pct=0.10):
        """
        Mental Stop / Maximum Adverse Excursion
        
        Exit when unrealized loss exceeds threshold
        Used in addition to hard stop as psychological limit
        
        Args:
            entry_price: Entry price
            current_price: Current price
            max_pain_pct: Maximum acceptable unrealized loss
        
        Returns:
            True if should exit, False otherwise
        """
        unrealized_loss = (current_price - entry_price) / entry_price
        return unrealized_loss < -max_pain_pct


# Example: Compare stop loss methods
import yfinance as yf

data = yf.download('AAPL', start='2023-01-01', end='2024-01-01')

# Calculate ATR
high = data['High']
low = data['Low']
close = data['Close']

tr1 = high - low
tr2 = abs (high - close.shift())
tr3 = abs (low - close.shift())
tr = pd.concat([tr1, tr2, tr3], axis=1).max (axis=1)
atr = tr.rolling(14).mean()

stop_manager = StopLossManager()

# Example trade
entry_price = 180.0
current_atr = atr.iloc[-1]

print("=== Stop Loss Comparison ===")
print(f"Entry Price: \\$\{entry_price:.2f}")
print(f"Current ATR: \\$\{current_atr:.2f}\\n")

# Fixed stop
fixed_stop = stop_manager.fixed_percentage_stop (entry_price, 0.05)
print(f"Fixed 5% Stop: \${fixed_stop:.2f} (\\$\{entry_price - fixed_stop:.2f} risk)")

# ATR stops
atr_2x = stop_manager.atr_stop (entry_price, current_atr, 2)
atr_3x = stop_manager.atr_stop (entry_price, current_atr, 3)
print(f"ATR 2x Stop: \${atr_2x:.2f} (\\$\{entry_price - atr_2x:.2f} risk)")
print(f"ATR 3x Stop: \${atr_3x:.2f} (\\$\{entry_price - atr_3x:.2f} risk)")

# Trailing stop
highest = 190.0  # Assume price ran to 190
trail_stop = stop_manager.trailing_stop(185, highest, 0.10)
print(f"\\nTrailing 10% Stop (from high \${highest:.2f}): \\$\{trail_stop:.2f}")
\`\`\`

---

## Portfolio-Level Risk Management

### Portfolio Heat & Exposure

\`\`\`python
"""
Portfolio Risk Management
"""

class PortfolioRiskManager:
    """
    Manage risk across entire portfolio
    """
    
    def __init__(self, capital=100000, max_portfolio_heat=0.20, max_positions=10):
        """
        Args:
            capital: Total capital
            max_portfolio_heat: Maximum total risk as % of capital (e.g., 0.20 = 20%)
            max_positions: Maximum number of concurrent positions
        """
        self.capital = capital
        self.max_portfolio_heat = max_portfolio_heat
        self.max_positions = max_positions
        self.positions = {}
    
    def calculate_portfolio_heat (self):
        """
        Calculate total portfolio risk
        
        Portfolio Heat = Sum of all position risks
        
        Returns:
            Total risk as % of capital
        """
        total_risk = sum (pos['risk_amount'] for pos in self.positions.values())
        return total_risk / self.capital
    
    def can_add_position (self, risk_amount):
        """
        Check if new position can be added
        
        Args:
            risk_amount: Risk amount for new position
        
        Returns:
            (can_add, reason)
        """
        # Check position count
        if len (self.positions) >= self.max_positions:
            return False, f"Max positions reached ({self.max_positions})"
        
        # Check portfolio heat
        current_heat = self.calculate_portfolio_heat()
        new_heat = (current_heat * self.capital + risk_amount) / self.capital
        
        if new_heat > self.max_portfolio_heat:
            return False, f"Portfolio heat would exceed {self.max_portfolio_heat:.0%} (currently {current_heat:.1%})"
        
        return True, "OK"
    
    def correlation_adjusted_sizing (self, base_size, correlation_matrix, existing_tickers, new_ticker):
        """
        Adjust position size based on correlation with existing positions
        
        High correlation → Reduce size
        Low/negative correlation → Keep size
        
        Args:
            base_size: Base position size (before adjustment)
            correlation_matrix: Correlation matrix of returns
            existing_tickers: Tickers of existing positions
            new_ticker: Ticker of new position
        
        Returns:
            Adjusted position size
        """
        if not existing_tickers:
            return base_size
        
        # Calculate average correlation with existing positions
        correlations = []
        for ticker in existing_tickers:
            if ticker in correlation_matrix.index and new_ticker in correlation_matrix.columns:
                corr = correlation_matrix.loc[ticker, new_ticker]
                correlations.append (corr)
        
        if not correlations:
            return base_size
        
        avg_correlation = np.mean (correlations)
        
        # Reduce size if highly correlated
        # avg_corr = 0 → no adjustment
        # avg_corr = 0.5 → reduce by 25%
        # avg_corr = 1.0 → reduce by 50%
        adjustment_factor = 1 - (avg_correlation * 0.5)
        adjustment_factor = max(0.5, adjustment_factor)  # Minimum 50% of base size
        
        adjusted_size = base_size * adjustment_factor
        
        return adjusted_size
    
    def sector_exposure_check (self, positions_by_sector, max_sector_exposure=0.40):
        """
        Check sector concentration
        
        Args:
            positions_by_sector: Dict of {sector: total_value}
            max_sector_exposure: Maximum % in one sector
        
        Returns:
            Dict of sectors exceeding limit
        """
        total_value = sum (positions_by_sector.values())
        violations = {}
        
        for sector, value in positions_by_sector.items():
            exposure = value / total_value
            if exposure > max_sector_exposure:
                violations[sector] = {
                    'exposure': exposure,
                    'limit': max_sector_exposure,
                    'excess': exposure - max_sector_exposure
                }
        
        return violations
    
    def calculate_var (self, returns, confidence=0.95, method='historical'):
        """
        Value at Risk (VaR)
        
        Maximum expected loss at given confidence level
        
        Args:
            returns: Historical returns
            confidence: Confidence level (e.g., 0.95 = 95%)
            method: 'historical', 'parametric', or 'monte_carlo'
        
        Returns:
            VaR value (negative number)
        """
        if method == 'historical':
            # Historical VaR: percentile of historical returns
            var = np.percentile (returns, (1 - confidence) * 100)
        
        elif method == 'parametric':
            # Parametric VaR: assume normal distribution
            mean = returns.mean()
            std = returns.std()
            z_score = -1.645  # For 95% confidence (one-tailed)
            if confidence == 0.99:
                z_score = -2.326
            var = mean + z_score * std
        
        elif method == 'monte_carlo':
            # Monte Carlo VaR: simulate future returns
            mean = returns.mean()
            std = returns.std()
            simulated = np.random.normal (mean, std, 10000)
            var = np.percentile (simulated, (1 - confidence) * 100)
        
        return var
    
    def calculate_cvar (self, returns, confidence=0.95):
        """
        Conditional VaR (Expected Shortfall)
        
        Average loss in worst (1-confidence)% of cases
        Better than VaR because it captures tail severity
        
        Args:
            returns: Historical returns
            confidence: Confidence level
        
        Returns:
            CVaR value (negative number)
        """
        var = self.calculate_var (returns, confidence, 'historical')
        # Average of returns worse than VaR
        cvar = returns[returns <= var].mean()
        return cvar
    
    def dynamic_leverage_adjustment (self, current_sharpe, target_leverage=1.0):
        """
        Dynamically adjust leverage based on strategy performance
        
        High Sharpe → Can increase leverage
        Low Sharpe → Reduce leverage
        
        Args:
            current_sharpe: Current strategy Sharpe ratio
            target_leverage: Base leverage (1.0 = no leverage)
        
        Returns:
            Adjusted leverage multiplier
        """
        if current_sharpe >= 1.5:
            # Excellent performance, can use more leverage
            return target_leverage * 1.5
        elif current_sharpe >= 1.0:
            # Good performance
            return target_leverage * 1.2
        elif current_sharpe >= 0.5:
            # Acceptable performance
            return target_leverage
        else:
            # Poor performance, reduce exposure
            return target_leverage * 0.5


# Example: Portfolio risk management
prm = PortfolioRiskManager (capital=100000, max_portfolio_heat=0.20, max_positions=10)

# Add positions
prm.positions['AAPL'] = {'risk_amount': 2000, 'size': 10000}
prm.positions['MSFT'] = {'risk_amount': 2000, 'size': 10000}
prm.positions['GOOGL'] = {'risk_amount': 2000, 'size': 10000}

print("=== Portfolio Risk Dashboard ===")
print(f"Current Positions: {len (prm.positions)}")
print(f"Portfolio Heat: {prm.calculate_portfolio_heat():.1%}")
print(f"Max Portfolio Heat: {prm.max_portfolio_heat:.0%}")
print(f"Available Heat: {(prm.max_portfolio_heat - prm.calculate_portfolio_heat()):.1%}")

# Try to add new position
can_add, reason = prm.can_add_position (risk_amount=2000)
print(f"\\nCan add new position (risk $2000): {can_add}")
print(f"Reason: {reason}")

# VaR/CVaR example
returns = np.random.normal(0.001, 0.02, 252)  # Simulate daily returns
var_95 = prm.calculate_var (returns, 0.95, 'historical')
cvar_95 = prm.calculate_cvar (returns, 0.95)

print(f"\\n=== Risk Metrics ===")
print(f"VaR (95%): {var_95:.2%} - Worst expected daily loss with 95% confidence")
print(f"CVaR (95%): {cvar_95:.2%} - Average loss in worst 5% of days")
print(f"Dollar VaR: \\$\{var_95 * 100000:,.0f}")
print(f"Dollar CVaR: \\$\{cvar_95 * 100000:,.0f}")
\`\`\`

---

## Drawdown Management

\`\`\`python
"""
Drawdown-Based Risk Management
"""

class DrawdownManager:
    """
    Manage trading based on drawdowns
    """
    
    def __init__(self, max_drawdown_threshold=0.15):
        """
        Args:
            max_drawdown_threshold: Maximum acceptable drawdown (e.g., 0.15 = 15%)
        """
        self.max_drawdown_threshold = max_drawdown_threshold
        self.equity_curve = []
        self.peak_equity = 0
    
    def update_equity (self, current_equity):
        """
        Update equity and calculate drawdown
        
        Returns:
            (current_drawdown, should_halt_trading)
        """
        self.equity_curve.append (current_equity)
        
        # Update peak
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Calculate drawdown from peak
        drawdown = (current_equity - self.peak_equity) / self.peak_equity
        
        # Check if should halt trading
        should_halt = drawdown < -self.max_drawdown_threshold
        
        return drawdown, should_halt
    
    def calculate_max_drawdown (self, equity_curve=None):
        """
        Calculate maximum drawdown from equity curve
        
        Returns:
            Maximum drawdown (negative value)
        """
        if equity_curve is None:
            equity_curve = self.equity_curve
        
        equity = pd.Series (equity_curve)
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        
        return drawdown.min()
    
    def drawdown_duration (self, equity_curve=None):
        """
        Calculate drawdown duration (days underwater)
        
        Returns:
            Maximum consecutive days in drawdown
        """
        if equity_curve is None:
            equity_curve = self.equity_curve
        
        equity = pd.Series (equity_curve)
        running_max = equity.cummax()
        is_underwater = equity < running_max
        
        # Count consecutive underwater periods
        max_duration = 0
        current_duration = 0
        
        for underwater in is_underwater:
            if underwater:
                current_duration += 1
                max_duration = max (max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    def recovery_factor (self, equity_curve=None):
        """
        Recovery Factor = Net Profit / Max Drawdown
        
        Higher is better (faster recovery from drawdowns)
        
        Returns:
            Recovery factor
        """
        if equity_curve is None:
            equity_curve = self.equity_curve
        
        net_profit = equity_curve[-1] - equity_curve[0]
        max_dd = abs (self.calculate_max_drawdown (equity_curve))
        
        if max_dd == 0:
            return np.inf
        
        return net_profit / (equity_curve[0] * max_dd)
    
    def reduce_position_size_on_drawdown (self, base_size, current_drawdown):
        """
        Reduce position sizes during drawdowns
        
        Psychology: Reduce risk when in drawdown to preserve capital
        
        Args:
            base_size: Normal position size
            current_drawdown: Current drawdown (negative)
        
        Returns:
            Adjusted position size
        """
        # No adjustment if no drawdown
        if current_drawdown >= 0:
            return base_size
        
        # Linear reduction based on drawdown
        # 0% DD → 100% size
        # 5% DD → 75% size
        # 10% DD → 50% size
        # 15% DD → 25% size (near halt threshold)
        
        dd_pct = abs (current_drawdown)
        reduction_factor = max(0.25, 1 - (dd_pct / self.max_drawdown_threshold) * 0.75)
        
        return base_size * reduction_factor


# Example: Drawdown management
dd_manager = DrawdownManager (max_drawdown_threshold=0.15)

# Simulate equity curve
equity_curve = [100000]
for _ in range(252):
    # Random return
    ret = np.random.normal(0.001, 0.02)
    new_equity = equity_curve[-1] * (1 + ret)
    equity_curve.append (new_equity)

# Analyze drawdown
max_dd = dd_manager.calculate_max_drawdown (equity_curve)
dd_duration = dd_manager.drawdown_duration (equity_curve)
recovery = dd_manager.recovery_factor (equity_curve)

print("=== Drawdown Analysis ===")
print(f"Maximum Drawdown: {max_dd:.2%}")
print(f"Drawdown Duration: {dd_duration} days")
print(f"Recovery Factor: {recovery:.2f}")
print(f"Final Equity: \\$\{equity_curve[-1]:,.0f}")

# Position size adjustment example
current_dd = -0.08  # 8% drawdown
base_size = 10000
adjusted_size = dd_manager.reduce_position_size_on_drawdown (base_size, current_dd)

print(f"\\n=== Position Sizing in Drawdown ===")
print(f"Current Drawdown: {current_dd:.1%}")
print(f"Base Position Size: \\$\{base_size:,.0f}")
print(f"Adjusted Size: \\$\{adjusted_size:,.0f} ({adjusted_size/base_size:.0%} of base)")
\`\`\`

---

## Key Takeaways

**Position Sizing Hierarchy**:
1. **Fixed 2%**: Simple, safe, professional standard
2. **Half Kelly**: Optimal growth with reduced volatility
3. **ATR-Based**: Adapts to market volatility
4. **Volatility Parity**: Normalizes risk across assets

**Stop Loss Best Practices**:
- **Use ATR stops** (2-3x ATR) for volatility adaptation
- **Never move stops against you** (only trail in your favor)
- **Set stop before entry** (mental stops fail in emotional moments)
- **Accept stop outs** (2% loss is better than 20% loss)

**Portfolio Risk Limits**:
- **Per Trade**: 2% maximum
- **Portfolio Heat**: 20% total risk maximum
- **Max Positions**: 10 concurrent positions
- **Sector Exposure**: < 40% in any sector
- **Correlation**: Reduce size if avg correlation > 0.5

**Drawdown Management**:
- **Halt Trading**: If DD > 15%, stop and reassess
- **Reduce Size**: Scale down in drawdowns
- **Max Duration**: 6 months underwater = problem
- **Recovery Factor**: Target > 3.0

**Expected Performance With Good Risk Management**:
- Sharpe Ratio: 0.8-1.5
- Max Drawdown: 10-20%
- Win Rate: 45-55%
- Profit Factor: > 1.5
- Recovery Factor: > 3.0
- **Most Important**: Survive first year!

**Remember**: The best traders focus on **not losing** rather than winning big. Capital preservation allows you to trade another day.
`,
};
