export const riskManagementFundamentals = {
  id: 'risk-management-fundamentals',
  title: 'Risk Management Fundamentals',
  content: `
# Risk Management Fundamentals

## Introduction

Risk management is the cornerstone of sustainable trading and investment operations. While returns capture headlines, it's risk management that determines long-term survival and success in financial markets. As the saying goes in the industry: "It's not about how much you make, but how much you don't lose."

This section establishes the foundational principles of risk management that underpin all modern financial institutions - from hedge funds managing billions to robo-advisors serving retail investors.

## Why Risk Management Matters

### Historical Context

The financial industry is littered with spectacular failures due to inadequate risk management:

- **Long-Term Capital Management (1998)**: A hedge fund run by Nobel laureates collapsed, requiring a $3.6B bailout
- **Lehman Brothers (2008)**: Excessive leverage and risk concentration led to bankruptcy
- **JPMorgan's London Whale (2012)**: Poor risk controls resulted in $6.2B trading loss
- **Archegos Capital (2021)**: Extreme leverage through total return swaps imploded, causing $10B+ losses

Each disaster shared common themes: inadequate risk measurement, poor governance, excessive leverage, and failure to stress test.

### The Paradox of Risk Management

Good risk management is often invisible:
- When risks are properly managed, nothing dramatic happens
- The absence of blow-ups doesn't mean risk management is unnecessary
- Risk managers prevent disasters that never occur

This creates organizational challenges - risk management is often viewed as a cost center until something goes wrong.

## Core Risk Management Principles

### 1. Risk Identification

You can't manage what you can't identify. Comprehensive risk identification involves:

**Market Risk**: Price movements in traded instruments
- Equity prices
- Interest rates
- FX rates
- Commodity prices
- Volatility

**Credit Risk**: Counterparty default or deterioration
- Corporate bonds
- Derivatives counterparties
- Loan portfolios
- Settlement risk

**Liquidity Risk**: Inability to exit positions without material impact
- Bid-ask spreads widening
- Market depth disappearing
- Funding liquidity (can't roll debt)

**Operational Risk**: Process, system, or human failures
- Trading errors
- Technology failures
- Fraud
- Model risk
- Key person risk

**Regulatory Risk**: Legal and compliance violations
- Trading restrictions
- Reporting failures
- Sanctions violations

### 2. Risk Measurement

Once identified, risks must be quantified:

**Statistical Measures**:
\`\`\`
Volatility (σ): Standard deviation of returns
Correlation (ρ): Co-movement between assets
Beta (β): Systematic risk relative to market
\`\`\`

**Value at Risk (VaR)**:
"What is the maximum loss over horizon H at confidence level α?"

Example: 1-day 99% VaR of $1M means:
- 99% confidence that losses won't exceed $1M in one day
- 1% chance of losing more than $1M

**Conditional VaR (CVaR / Expected Shortfall)**:
"Given that VaR is exceeded, what is the expected loss?"

CVaR addresses VaR's weakness of not describing tail losses.

**Stress Testing**:
"What happens in extreme scenarios?"
- Historical scenarios (2008 crisis, COVID-19)
- Hypothetical scenarios (oil at $200)
- Reverse stress tests (what breaks us?)

### 3. Risk Limits

Measurement without limits is pointless. Risk limits constrain exposure:

**Position Limits**: Maximum exposure to any single position
\`\`\`
Single stock: ≤ 5% of portfolio
Sector: ≤ 20% of portfolio
\`\`\`

**Loss Limits**: Maximum acceptable loss
\`\`\`
Daily stop-loss: $500K
Monthly stop-loss: $2M
Drawdown limit: 10% from peak
\`\`\`

**Risk Factor Limits**: Sensitivity to market factors
\`\`\`
Portfolio beta: 0.8 - 1.2
Duration: 5 - 7 years
Delta: -1000 to +1000 contracts
\`\`\`

**Leverage Limits**: Maximum debt-to-equity
\`\`\`
Gross leverage: ≤ 3:1
Net leverage: ≤ 1.5:1
\`\`\`

### 4. Risk Monitoring

Continuous monitoring ensures limits are respected:

**Real-Time Monitoring**:
- Pre-trade risk checks (will trade violate limits?)
- Intraday position tracking
- Live P&L calculation
- Automatic alerts when approaching limits

**End-of-Day Reconciliation**:
- Position reconciliation
- P&L attribution
- Risk metrics recalculation
- Exception reporting

**Regular Reporting**:
- Daily risk reports to traders
- Weekly reports to management
- Monthly board presentations
- Regulatory reporting

### 5. Risk Control and Mitigation

When risks exceed acceptable levels, action is required:

**Hedging**: Offset exposure with opposite positions
\`\`\`python
# Example: Hedging equity portfolio with index futures
portfolio_value = 10_000_000
portfolio_beta = 1.2
spy_futures_price = 4500
contract_multiplier = 50

# Number of contracts to hedge beta
hedge_ratio = (portfolio_value * portfolio_beta) / (spy_futures_price * contract_multiplier)
contracts_to_short = -int(hedge_ratio)
print(f"Short {-contracts_to_short} SPY futures contracts")
\`\`\`

**Diversification**: Spread risk across uncorrelated exposures
\`\`\`python
import numpy as np

# Portfolio variance with diversification
def portfolio_variance(weights, cov_matrix):
    return weights @ cov_matrix @ weights

# Equal-weighted portfolio of N assets
N = 10
weights = np.ones(N) / N

# Benefit of diversification
individual_variance = 0.20**2  # 20% vol per stock
avg_correlation = 0.30
portfolio_vol = np.sqrt(individual_variance * (1/N + (1 - 1/N) * avg_correlation))
print(f"Individual vol: 20%, Portfolio vol: {portfolio_vol*100:.1f}%")
\`\`\`

**Position Sizing**: Risk-based allocation
\`\`\`python
def kelly_criterion(win_rate, avg_win, avg_loss):
    """
    Optimal position size based on edge
    """
    win_loss_ratio = avg_win / avg_loss
    kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
    # Use half-Kelly for safety
    return kelly_fraction * 0.5

# Example: 55% win rate, 2:1 win/loss ratio
optimal_size = kelly_criterion(0.55, 2, 1)
print(f"Optimal position size: {optimal_size*100:.1f}% of capital")
\`\`\`

**Stop Losses**: Automatic exit at predetermined levels
\`\`\`python
class PositionWithStopLoss:
    def __init__(self, symbol, entry_price, shares, stop_loss_pct=0.02):
        self.symbol = symbol
        self.entry_price = entry_price
        self.shares = shares
        self.stop_price = entry_price * (1 - stop_loss_pct)
        
    def check_stop_loss(self, current_price):
        if current_price <= self.stop_price:
            loss = (self.stop_price - self.entry_price) * self.shares
            return True, loss
        return False, 0
\`\`\`

## Risk Management Framework

### Three Lines of Defense Model

Financial institutions typically organize risk management in three layers:

**First Line: Business Units**
- Traders and portfolio managers
- Own their risks
- Responsible for staying within limits
- Daily risk reporting

**Second Line: Risk Management**
- Independent risk function
- Sets risk policies and limits
- Monitors and reports risks
- Challenges business decisions
- Does NOT make trading decisions

**Third Line: Internal Audit**
- Independent assurance
- Audits risk processes
- Tests control effectiveness
- Reports to board

This separation ensures:
- Independence of risk oversight
- Accountability at business level
- Multiple layers of review

### Risk Committee Structure

Typical governance structure:

\`\`\`
Board of Directors
    ↓
Risk Committee (Board level)
    ↓
Chief Risk Officer (CRO)
    ↓
├─ Market Risk
├─ Credit Risk  
├─ Operational Risk
└─ Liquidity Risk
\`\`\`

**Risk Committee Responsibilities**:
- Approve risk appetite
- Set risk limits
- Review risk reports
- Approve new products/strategies
- Escalation point for limit breaches

### Risk Appetite Framework

Risk appetite defines the organization's tolerance for risk:

**Components**:

1. **Risk Capacity**: Maximum risk the firm can bear
   - Capital available
   - Regulatory requirements
   - Investor constraints

2. **Risk Appetite**: Risk willing to take for returns
   - Target Sharpe ratio: 1.5
   - Maximum drawdown: 15%
   - Maximum VaR: $10M

3. **Risk Tolerance**: Acceptable deviation from appetite
   - Soft limits (warnings)
   - Hard limits (no exceptions)
   - Escalation procedures

**Example Risk Appetite Statement**:
\`\`\`
Risk Appetite for Equity Long/Short Fund:

Objective: Generate 12% annual returns with 10% volatility

Risk Limits:
- Maximum drawdown: 15% (hard limit)
- Daily 99% VaR: $5M (hard limit)
- Gross leverage: 3:1 maximum (hard limit)
- Net exposure: -20% to +50% (soft limit)
- Single position: 5% maximum (hard limit)
- Sector concentration: 30% maximum (soft limit)

Monitoring:
- Real-time position monitoring
- Daily risk reports
- Weekly risk committee review
- Monthly board reporting
\`\`\`

## Risk Metrics Deep Dive

### Volatility

The most basic risk measure - dispersion of returns:

\`\`\`python
import numpy as np
import pandas as pd

def calculate_volatility(returns, annualization_factor=252):
    """
    Calculate annualized volatility
    
    Args:
        returns: Series of returns
        annualization_factor: 252 for daily, 12 for monthly
    """
    return returns.std() * np.sqrt(annualization_factor)

def rolling_volatility(returns, window=20, annualization_factor=252):
    """
    Rolling volatility for time-varying risk
    """
    return returns.rolling(window).std() * np.sqrt(annualization_factor)

# Example
returns = pd.Series(np.random.normal(0.0005, 0.02, 252))
current_vol = calculate_volatility(returns)
print(f"Annualized volatility: {current_vol*100:.1f}%")

# Volatility clustering - high vol periods follow high vol
rolling_vol = rolling_volatility(returns)
print(f"Current 20-day vol: {rolling_vol.iloc[-1]*100:.1f}%")
\`\`\`

**Key Points**:
- Volatility clusters (GARCH effects)
- Not constant over time
- Higher in downturns
- Asset-specific and time-varying

### Beta

Systematic risk - sensitivity to market:

\`\`\`python
def calculate_beta(asset_returns, market_returns):
    """
    Beta = Cov(asset, market) / Var(market)
    """
    covariance = np.cov(asset_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    return covariance / market_variance

def rolling_beta(asset_returns, market_returns, window=60):
    """
    Time-varying beta
    """
    betas = []
    for i in range(window, len(asset_returns)):
        asset_window = asset_returns[i-window:i]
        market_window = market_returns[i-window:i]
        beta = calculate_beta(asset_window, market_window)
        betas.append(beta)
    return betas

# Portfolio beta
def portfolio_beta(weights, individual_betas):
    """
    Portfolio beta = weighted average of individual betas
    """
    return np.dot(weights, individual_betas)
\`\`\`

**Interpretation**:
- β = 1: Moves with market
- β > 1: More volatile than market (tech stocks)
- β < 1: Less volatile (utilities)
- β < 0: Inverse relationship (gold, treasuries)

### Correlation

Co-movement between assets:

\`\`\`python
def correlation_matrix(returns_df):
    """
    Calculate correlation matrix for portfolio
    """
    return returns_df.corr()

def rolling_correlation(asset1_returns, asset2_returns, window=60):
    """
    Time-varying correlation
    """
    return pd.Series(asset1_returns).rolling(window).corr(pd.Series(asset2_returns))

# Correlation breakdown in crisis
def crisis_correlation_analysis(returns_df, crisis_dates):
    """
    Analyze correlation during normal vs crisis periods
    """
    normal_period = returns_df[~returns_df.index.isin(crisis_dates)]
    crisis_period = returns_df[returns_df.index.isin(crisis_dates)]
    
    print("Normal period correlations:")
    print(normal_period.corr().mean())
    print("\\nCrisis period correlations:")
    print(crisis_period.corr().mean())
\`\`\`

**Critical Risk Management Point**:
Correlations increase during market stress - diversification fails when you need it most!

### Maximum Drawdown

Maximum peak-to-trough decline:

\`\`\`python
def calculate_max_drawdown(returns):
    """
    Maximum drawdown - worst peak-to-trough decline
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def drawdown_duration(returns):
    """
    How long to recover from drawdown
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    
    # Find drawdown periods
    in_drawdown = drawdown < 0
    drawdown_periods = []
    start = None
    
    for i, dd in enumerate(in_drawdown):
        if dd and start is None:
            start = i
        elif not dd and start is not None:
            drawdown_periods.append(i - start)
            start = None
    
    return max(drawdown_periods) if drawdown_periods else 0

# Example
returns = pd.Series(np.random.normal(0.001, 0.02, 252))
max_dd = calculate_max_drawdown(returns)
print(f"Maximum drawdown: {max_dd*100:.1f}%")
\`\`\`

**Why Maximum Drawdown Matters**:
- Psychological impact on investors
- Redemption risk in funds
- Regulatory capital requirements
- Recovery time (20% loss requires 25% gain)

## Risk-Adjusted Performance Metrics

Returns without context are meaningless. Risk-adjusted metrics enable fair comparison:

### Sharpe Ratio

Most popular risk-adjusted metric:

\`\`\`python
def sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
    """
    Sharpe Ratio = (Return - RFR) / Volatility
    """
    excess_returns = returns - risk_free_rate/periods_per_year
    return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()

# Example comparison
strategy_a_returns = pd.Series(np.random.normal(0.0008, 0.015, 252))
strategy_b_returns = pd.Series(np.random.normal(0.0006, 0.010, 252))

sr_a = sharpe_ratio(strategy_a_returns)
sr_b = sharpe_ratio(strategy_b_returns)

print(f"Strategy A Sharpe: {sr_a:.2f}")
print(f"Strategy B Sharpe: {sr_b:.2f}")
\`\`\`

**Sharpe Ratio Interpretation**:
- < 0: Losing money
- 0-1: Not great
- 1-2: Good
- 2-3: Very good
- > 3: Excellent (or suspicious)

**Limitations**:
- Assumes normal distribution
- Treats upside and downside volatility equally
- Can be manipulated (smoothing, option selling)

### Sortino Ratio

Focuses on downside risk:

\`\`\`python
def sortino_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
    """
    Sortino Ratio = (Return - RFR) / Downside Deviation
    """
    excess_returns = returns - risk_free_rate/periods_per_year
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std

# Better for asymmetric strategies
\`\`\`

### Calmar Ratio

Return relative to maximum drawdown:

\`\`\`python
def calmar_ratio(returns, periods_per_year=252):
    """
    Calmar Ratio = Annual Return / Maximum Drawdown
    """
    annual_return = returns.mean() * periods_per_year
    max_dd = abs(calculate_max_drawdown(returns))
    return annual_return / max_dd
\`\`\`

Popular in hedge funds - measures return per unit of worst-case loss.

## Real-World Risk Management Systems

### Example: Hedge Fund Risk System

\`\`\`python
class RiskManagementSystem:
    def __init__(self, portfolio, limits):
        self.portfolio = portfolio
        self.limits = limits
        self.alerts = []
        
    def run_daily_risk_checks(self):
        """
        Comprehensive daily risk assessment
        """
        checks = {
            'position_limits': self.check_position_limits(),
            'loss_limits': self.check_loss_limits(),
            'var_limit': self.check_var_limit(),
            'leverage_limit': self.check_leverage(),
            'concentration': self.check_concentration()
        }
        
        for check_name, (passed, details) in checks.items():
            if not passed:
                self.alerts.append({
                    'check': check_name,
                    'details': details,
                    'timestamp': pd.Timestamp.now()
                })
        
        return checks, self.alerts
    
    def check_position_limits(self):
        """
        Individual position size limits
        """
        portfolio_value = self.portfolio.total_value()
        violations = []
        
        for symbol, position in self.portfolio.positions.items():
            position_pct = position.value / portfolio_value
            if position_pct > self.limits['max_position_size']:
                violations.append({
                    'symbol': symbol,
                    'size': position_pct,
                    'limit': self.limits['max_position_size']
                })
        
        return len(violations) == 0, violations
    
    def check_loss_limits(self):
        """
        Daily and cumulative loss limits
        """
        daily_pnl = self.portfolio.daily_pnl()
        monthly_pnl = self.portfolio.monthly_pnl()
        
        violations = []
        if daily_pnl < -self.limits['daily_loss_limit']:
            violations.append({
                'type': 'daily',
                'pnl': daily_pnl,
                'limit': -self.limits['daily_loss_limit']
            })
        
        if monthly_pnl < -self.limits['monthly_loss_limit']:
            violations.append({
                'type': 'monthly',
                'pnl': monthly_pnl,
                'limit': -self.limits['monthly_loss_limit']
            })
        
        return len(violations) == 0, violations
    
    def check_var_limit(self):
        """
        Value at Risk limit
        """
        var_99 = self.portfolio.calculate_var(confidence=0.99)
        
        if abs(var_99) > self.limits['max_var']:
            return False, {
                'var': var_99,
                'limit': self.limits['max_var']
            }
        return True, None
    
    def check_leverage(self):
        """
        Leverage limits
        """
        gross_leverage = self.portfolio.gross_leverage()
        net_leverage = self.portfolio.net_leverage()
        
        violations = []
        if gross_leverage > self.limits['max_gross_leverage']:
            violations.append({
                'type': 'gross',
                'value': gross_leverage,
                'limit': self.limits['max_gross_leverage']
            })
        
        if abs(net_leverage) > self.limits['max_net_leverage']:
            violations.append({
                'type': 'net',
                'value': net_leverage,
                'limit': self.limits['max_net_leverage']
            })
        
        return len(violations) == 0, violations
    
    def check_concentration(self):
        """
        Sector and geographic concentration
        """
        sector_concentration = self.portfolio.sector_exposure()
        violations = []
        
        for sector, exposure in sector_concentration.items():
            if exposure > self.limits['max_sector_concentration']:
                violations.append({
                    'sector': sector,
                    'exposure': exposure,
                    'limit': self.limits['max_sector_concentration']
                })
        
        return len(violations) == 0, violations
\`\`\`

## Industry Best Practices

### 1. Independent Risk Function

Risk management must be independent from profit centers:
- Separate reporting line (CRO reports to CEO/Board)
- Cannot be overruled by traders
- Compensation not tied to trading P&L
- Veto power over excessive risks

### 2. Comprehensive Limit Framework

Multi-layered limits:
- Position limits (individual holdings)
- Risk factor limits (duration, delta, vega)
- Loss limits (daily, weekly, monthly)
- Leverage limits
- Concentration limits

### 3. Real-Time Monitoring

Technology infrastructure for immediate risk visibility:
- Pre-trade risk checks
- Intraday position tracking
- Real-time P&L
- Automatic alerts
- Kill switches for limit breaches

### 4. Stress Testing

Regular stress testing across scenarios:
- Historical scenarios (past crises)
- Hypothetical scenarios (future risks)
- Reverse stress tests (what breaks us)
- Correlation breakdown scenarios

### 5. Risk Culture

Risk awareness embedded in culture:
- Tone from the top (CEO/Board support)
- Training for all employees
- Open discussion of risks
- Learning from near-misses
- Rewarding good risk management

## Regulatory Requirements

Financial institutions face extensive risk management requirements:

### Basel III (Banks)

- **Minimum Capital Requirements**: Risk-weighted assets
- **Leverage Ratio**: Balance sheet limit
- **Liquidity Coverage Ratio**: Short-term liquidity
- **Net Stable Funding Ratio**: Long-term liquidity

### SEC/FINRA (Broker-Dealers)

- **Net Capital Rule**: Minimum capital requirements
- **Customer Protection Rule**: Segregation of customer funds
- **Risk Management Controls**: Written policies and procedures

### CFTC (Futures/Swaps)

- **Initial Margin Requirements**: For uncleared swaps
- **Risk Management Program**: Comprehensive policies
- **Real-Time Reporting**: Swap data repositories

### Dodd-Frank Act

- **Stress Testing**: Annual comprehensive stress tests
- **Living Wills**: Resolution plans for failure
- **Volcker Rule**: Proprietary trading restrictions

## Common Risk Management Mistakes

### 1. Over-Reliance on VaR

VaR has limitations:
- Says nothing about tail risk
- Assumes normal distribution
- Backward-looking (historical data)
- Can be manipulated

**Mitigation**: Use multiple risk metrics (VaR, CVaR, stress tests, scenario analysis)

### 2. Ignoring Liquidity Risk

Many blow-ups occur not from market moves, but inability to exit:
- Assuming you can sell when needed
- Ignoring bid-ask spreads
- Not considering market depth
- Forgetting correlations spike in stress

**Mitigation**: Include liquidity considerations in all risk assessments

### 3. Model Risk

"All models are wrong, some are useful" - George Box

Models can fail:
- Wrong assumptions (normal distribution)
- Fat tails underestimated
- Regime changes not captured
- Implementation errors (bugs)

**Mitigation**: Model validation, stress testing, multiple models

### 4. Concentration Risk

Putting too many eggs in one basket:
- Single position too large
- Sector concentration
- Geographic concentration
- Factor concentration (all momentum trades)

**Mitigation**: Diversification limits, concentration monitoring

### 5. Tail Risk Complacency

99% VaR means 1% chance of exceedance:
- Over 250 trading days, expect 2-3 VaR breaks
- When it happens, could be much worse than VaR
- Tail events cluster

**Mitigation**: CVaR, stress testing, tail hedging strategies

## Conclusion

Risk management is not about eliminating risk - it's about:
- Understanding the risks you're taking
- Taking calculated risks for adequate compensation
- Staying within acceptable boundaries
- Surviving to trade another day

As Warren Buffett said: "Risk comes from not knowing what you're doing." Comprehensive risk management ensures you know exactly what you're doing, even when markets don't cooperate.

The remainder of this module will dive deep into specific risk types, measurement methodologies, and implementation of production-grade risk management systems. We'll build real risk systems used by hedge funds and banks, not just theoretical frameworks.

Remember: In trading, you can be right 99 times and wrong once, and that one time can wipe you out. Risk management ensures that one time doesn't destroy you.
`,
};
