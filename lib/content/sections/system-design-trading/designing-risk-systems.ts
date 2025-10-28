export const designingRiskSystems = {
  title: 'Designing Risk Systems',
  id: 'designing-risk-systems',
  content: `
# Designing Risk Systems

## Introduction

A **Risk System** is the guardian of trading operations—continuously monitoring positions, calculating exposures, enforcing limits, and preventing catastrophic losses. In 2012, Knight Capital lost $440 million in 45 minutes due to inadequate risk controls. Effective risk systems could have prevented this.

### Core Functions

1. **Real-time P&L**: Track profit/loss across all positions
2. **Position limits**: Enforce maximum position sizes
3. **Greeks aggregation**: Monitor options risk (Delta, Gamma, Vega)
4. **VaR/CVaR**: Estimate tail risk
5. **Stress testing**: Simulate extreme market scenarios
6. **Margin calculation**: Ensure sufficient capital
7. **Alerting**: Notify when limits approached or breached

By the end of this section, you'll understand:
- Real-time P&L calculation architecture
- VaR and CVaR implementation
- Greeks aggregation for options portfolios
- Risk limit enforcement
- Stress testing frameworks
- Production risk monitoring systems

---

## Real-Time P&L Calculation

### Architecture

\`\`\`
Market Data → Position Manager → P&L Engine → Risk Dashboard
                      ↓
                  Trade Feed
\`\`\`

### Implementation

\`\`\`python
"""
Real-Time P&L Calculation System
Updates P&L on every market data tick
"""

from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime
import pandas as pd
import numpy as np

@dataclass
class Position:
    """Position in a symbol"""
    symbol: str
    quantity: float  # Positive = long, Negative = short
    average_price: float  # Average entry price
    current_price: float
    
    @property
    def market_value (self) -> float:
        """Current market value"""
        return self.quantity * self.current_price
    
    @property
    def cost_basis (self) -> float:
        """Original cost"""
        return self.quantity * self.average_price
    
    @property
    def unrealized_pnl (self) -> float:
        """Unrealized profit/loss"""
        return self.market_value - self.cost_basis

class PLEngine:
    """
    Real-time P&L calculation
    Handles both realized and unrealized P&L
    """
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.realized_pnl = 0.0
        self.cash = 0.0
        
        # Track P&L history
        self.pnl_history = []
    
    def update_position (self, symbol: str, quantity_change: float, price: float):
        """
        Update position from trade
        quantity_change: +100 for buy, -100 for sell
        """
        if symbol not in self.positions:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity_change,
                average_price=price,
                current_price=price
            )
        else:
            pos = self.positions[symbol]
            
            # Check if closing or adding
            if (pos.quantity > 0 and quantity_change < 0) or (pos.quantity < 0 and quantity_change > 0):
                # Closing (at least partially)
                close_quantity = min (abs (quantity_change), abs (pos.quantity))
                
                if pos.quantity > 0:
                    # Closing long
                    realized = close_quantity * (price - pos.average_price)
                else:
                    # Closing short
                    realized = close_quantity * (pos.average_price - price)
                
                self.realized_pnl += realized
                self.cash += realized
                
                # Update position
                pos.quantity += quantity_change
                
                if abs (pos.quantity) < 0.0001:  # Fully closed
                    del self.positions[symbol]
                    return
            else:
                # Adding to position
                # Recalculate average price
                total_cost = pos.cost_basis + (quantity_change * price)
                pos.quantity += quantity_change
                pos.average_price = total_cost / pos.quantity if pos.quantity != 0 else price
        
        # Update cash
        self.cash -= quantity_change * price
    
    def update_market_price (self, symbol: str, price: float, timestamp: int):
        """Update market price and recalculate P&L"""
        if symbol in self.positions:
            self.positions[symbol].current_price = price
        
        # Record P&L snapshot
        total_pnl = self.get_total_pnl()
        self.pnl_history.append({
            'timestamp': timestamp,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.get_unrealized_pnl(),
            'total_pnl': total_pnl
        })
    
    def get_unrealized_pnl (self) -> float:
        """Total unrealized P&L across all positions"""
        return sum (pos.unrealized_pnl for pos in self.positions.values())
    
    def get_total_pnl (self) -> float:
        """Total P&L (realized + unrealized)"""
        return self.realized_pnl + self.get_unrealized_pnl()
    
    def get_position_pnl (self, symbol: str) -> dict:
        """Get P&L breakdown for specific position"""
        if symbol not in self.positions:
            return {'unrealized': 0.0}
        
        pos = self.positions[symbol]
        return {
            'quantity': pos.quantity,
            'avg_price': pos.average_price,
            'current_price': pos.current_price,
            'market_value': pos.market_value,
            'cost_basis': pos.cost_basis,
            'unrealized_pnl': pos.unrealized_pnl,
            'unrealized_pnl_pct': pos.unrealized_pnl / abs (pos.cost_basis) if pos.cost_basis != 0 else 0
        }

# Example
if __name__ == "__main__":
    engine = PLEngine()
    
    # Buy 100 AAPL @ $150
    engine.update_position('AAPL', 100, 150.0)
    print(f"After buy: Unrealized P&L = \\$\{engine.get_unrealized_pnl():.2f}")
    
    # Price moves to $155
    engine.update_market_price('AAPL', 155.0, 1000000)
    print(f"After price move: Unrealized P&L = \\$\{engine.get_unrealized_pnl():.2f}")
    
    # Sell 50 AAPL @ $155
    engine.update_position('AAPL', -50, 155.0)
    print(f"After partial close: Realized P&L = \\$\{engine.realized_pnl:.2f}")
    print(f"After partial close: Unrealized P&L = \\$\{engine.get_unrealized_pnl():.2f}")
\`\`\`

---

## Value at Risk (VaR)

### VaR Definition

**Value at Risk**: Maximum expected loss over time period at confidence level

Example: 1-day VaR at 95% confidence = $100K means:
- 95% of days, loss will be < $100K
- 5% of days, loss will be > $100K (tail risk)

### VaR Calculation Methods

\`\`\`python
"""
VaR Calculation Methods
1. Historical Simulation
2. Parametric (Variance-Covariance)
3. Monte Carlo Simulation
"""

import pandas as pd
import numpy as np
from scipy import stats

class VaRCalculator:
    """Calculate Value at Risk using multiple methods"""
    
    def __init__(self, positions: Dict[str, Position], price_history: pd.DataFrame):
        """
        positions: Current positions {symbol: Position}
        price_history: DataFrame with columns [date, symbol, close]
        """
        self.positions = positions
        self.price_history = price_history
    
    def historical_var (self, confidence_level: float = 0.95, horizon_days: int = 1) -> float:
        """
        Historical Simulation VaR
        Use actual historical returns distribution
        """
        # Calculate returns for each symbol
        returns_df = self.price_history.pivot (index='date', columns='symbol', values='close').pct_change()
        
        # Calculate portfolio returns
        portfolio_values = []
        for symbol, pos in self.positions.items():
            if symbol in returns_df.columns:
                portfolio_values.append (pos.market_value)
        
        total_value = sum (portfolio_values)
        
        portfolio_returns = []
        for _, row in returns_df.iterrows():
            portfolio_return = 0
            for symbol, pos in self.positions.items():
                if symbol in row.index and not pd.isna (row[symbol]):
                    weight = pos.market_value / total_value
                    portfolio_return += weight * row[symbol]
            portfolio_returns.append (portfolio_return)
        
        portfolio_returns = np.array (portfolio_returns)
        portfolio_returns = portfolio_returns[~np.isnan (portfolio_returns)]
        
        # Scale to horizon
        portfolio_returns = portfolio_returns * np.sqrt (horizon_days)
        
        # VaR = quantile of loss distribution
        var = -np.percentile (portfolio_returns, (1 - confidence_level) * 100) * total_value
        
        return var
    
    def parametric_var (self, confidence_level: float = 0.95, horizon_days: int = 1) -> float:
        """
        Parametric VaR (Variance-Covariance method)
        Assumes returns are normally distributed
        """
        # Calculate covariance matrix
        returns_df = self.price_history.pivot (index='date', columns='symbol', values='close').pct_change().dropna()
        
        # Portfolio weights
        symbols = list (self.positions.keys())
        weights = np.array([
            self.positions[s].market_value for s in symbols
        ])
        total_value = weights.sum()
        weights = weights / total_value
        
        # Filter returns to only our symbols
        returns_matrix = returns_df[symbols].values
        
        # Covariance matrix
        cov_matrix = np.cov (returns_matrix.T)
        
        # Portfolio variance
        portfolio_variance = np.dot (weights, np.dot (cov_matrix, weights))
        portfolio_std = np.sqrt (portfolio_variance)
        
        # VaR = z-score * std * portfolio_value
        z_score = stats.norm.ppf (confidence_level)
        var = z_score * portfolio_std * total_value * np.sqrt (horizon_days)
        
        return var
    
    def monte_carlo_var(
        self,
        confidence_level: float = 0.95,
        horizon_days: int = 1,
        n_simulations: int = 10000
    ) -> float:
        """
        Monte Carlo VaR
        Simulate many possible future scenarios
        """
        # Calculate returns statistics
        returns_df = self.price_history.pivot (index='date', columns='symbol', values='close').pct_change().dropna()
        
        symbols = list (self.positions.keys())
        returns_matrix = returns_df[symbols].values
        
        # Mean returns and covariance
        mean_returns = returns_matrix.mean (axis=0)
        cov_matrix = np.cov (returns_matrix.T)
        
        # Current portfolio value
        weights = np.array([self.positions[s].market_value for s in symbols])
        total_value = weights.sum()
        
        # Simulate returns
        simulated_returns = np.random.multivariate_normal(
            mean_returns * horizon_days,
            cov_matrix * horizon_days,
            n_simulations
        )
        
        # Calculate portfolio returns for each simulation
        portfolio_returns = (weights * (1 + simulated_returns)).sum (axis=1) - total_value
        
        # VaR = quantile of loss distribution
        var = -np.percentile (portfolio_returns, (1 - confidence_level) * 100)
        
        return var

# Example
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    data = []
    for symbol in ['AAPL', 'TSLA']:
        prices = 100 * np.exp (np.cumsum (np.random.normal(0.001, 0.02, len (dates))))
        for date, price in zip (dates, prices):
            data.append({'date': date, 'symbol': symbol, 'close': price})
    
    price_history = pd.DataFrame (data)
    
    # Current positions
    positions = {
        'AAPL': Position('AAPL', 100, 150.0, 155.0),
        'TSLA': Position('TSLA', 50, 200.0, 210.0)
    }
    
    calculator = VaRCalculator (positions, price_history)
    
    hist_var = calculator.historical_var (confidence_level=0.95, horizon_days=1)
    param_var = calculator.parametric_var (confidence_level=0.95, horizon_days=1)
    mc_var = calculator.monte_carlo_var (confidence_level=0.95, horizon_days=1)
    
    print(f"Historical VaR (95%, 1-day): \\$\{hist_var:,.2f}")
    print(f"Parametric VaR (95%, 1-day): \\$\{param_var:,.2f}")
    print(f"Monte Carlo VaR (95%, 1-day): \\$\{mc_var:,.2f}")
\`\`\`

---

## Conditional Value at Risk (CVaR)

**CVaR**: Expected loss in worst cases (beyond VaR)

\`\`\`python
def calculate_cvar (returns: np.array, confidence_level: float = 0.95) -> float:
    """
    Calculate CVaR (Expected Shortfall)
    CVaR = average loss in worst (1-confidence_level) cases
    """
    var_threshold = np.percentile (returns, (1 - confidence_level) * 100)
    
    # Average of returns below VaR threshold
    tail_returns = returns[returns <= var_threshold]
    cvar = -tail_returns.mean()
    
    return cvar

# CVaR is always >= VaR
# Example: VaR = $100K, CVaR = $150K
# Meaning: If loss exceeds VaR, expect average loss of $150K
\`\`\`

---

## Greeks Aggregation for Options

\`\`\`python
"""
Greeks Calculation and Aggregation
For options risk management
"""

from scipy.stats import norm
import numpy as np

class GreeksCalculator:
    """Calculate options Greeks using Black-Scholes"""
    
    @staticmethod
    def black_scholes_greeks(
        S: float,  # Spot price
        K: float,  # Strike price
        T: float,  # Time to expiration (years)
        r: float,  # Risk-free rate
        sigma: float,  # Volatility
        option_type: str = 'call'
    ) -> dict:
        """Calculate all Greeks"""
        
        # Black-Scholes formula components
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            # Delta
            delta = norm.cdf (d1)
            
            # Theta (per day)
            theta = (
                -S * norm.pdf (d1) * sigma / (2 * np.sqrt(T))
                - r * K * np.exp(-r * T) * norm.cdf (d2)
            ) / 365
        else:  # put
            delta = -norm.cdf(-d1)
            
            theta = (
                -S * norm.pdf (d1) * sigma / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T) * norm.cdf(-d2)
            ) / 365
        
        # Gamma (same for calls and puts)
        gamma = norm.pdf (d1) / (S * sigma * np.sqrt(T))
        
        # Vega (per 1% change in volatility)
        vega = S * norm.pdf (d1) * np.sqrt(T) / 100
        
        # Rho (per 1% change in interest rate)
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf (d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }

@dataclass
class OptionPosition:
    """Options position"""
    symbol: str
    quantity: int  # Number of contracts (1 contract = 100 shares)
    option_type: str  # 'call' or 'put'
    strike: float
    expiration: float  # Years to expiration
    underlying_price: float
    volatility: float
    risk_free_rate: float = 0.05

class PortfolioGreeks:
    """
    Aggregate Greeks across options portfolio
    Critical for managing options risk
    """
    
    def __init__(self):
        self.option_positions: List[OptionPosition] = []
        self.stock_positions: Dict[str, Position] = {}
    
    def add_option_position (self, pos: OptionPosition):
        """Add options position"""
        self.option_positions.append (pos)
    
    def add_stock_position (self, pos: Position):
        """Add stock position (Delta = 1 per share)"""
        self.stock_positions[pos.symbol] = pos
    
    def calculate_portfolio_greeks (self) -> dict:
        """
        Aggregate Greeks across portfolio
        """
        total_delta = 0
        total_gamma = 0
        total_vega = 0
        total_theta = 0
        total_rho = 0
        
        calculator = GreeksCalculator()
        
        # Options Greeks
        for pos in self.option_positions:
            greeks = calculator.black_scholes_greeks(
                S=pos.underlying_price,
                K=pos.strike,
                T=pos.expiration,
                r=pos.risk_free_rate,
                sigma=pos.volatility,
                option_type=pos.option_type
            )
            
            # Multiply by position size (contracts * 100 shares/contract)
            multiplier = pos.quantity * 100
            
            total_delta += greeks['delta'] * multiplier
            total_gamma += greeks['gamma'] * multiplier
            total_vega += greeks['vega'] * multiplier
            total_theta += greeks['theta'] * multiplier
            total_rho += greeks['rho'] * multiplier
        
        # Stock positions (Delta = 1, other Greeks = 0)
        for pos in self.stock_positions.values():
            total_delta += pos.quantity
        
        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'vega': total_vega,
            'theta': total_theta,
            'rho': total_rho
        }
    
    def get_risk_metrics (self) -> dict:
        """Get interpretable risk metrics"""
        greeks = self.calculate_portfolio_greeks()
        
        return {
            'delta': greeks['delta'],
            'delta_interpretation': f"Portfolio moves \${abs (greeks['delta']):.0f} for $1 move in underlying",
            'gamma': greeks['gamma'],
            'gamma_interpretation': f"Delta changes by {greeks['gamma']:.2f} for $1 move in underlying",
            'vega': greeks['vega'],
            'vega_interpretation': f"Portfolio changes \${greeks['vega']:.0f} for 1% change in volatility",
            'theta': greeks['theta'],
            'theta_interpretation': f"Portfolio loses \${-greeks['theta']:.0f} per day from time decay",
            'rho': greeks['rho']
        }

# Example
if __name__ == "__main__":
    portfolio = PortfolioGreeks()
    
    # Add AAPL call option
    portfolio.add_option_position(OptionPosition(
        symbol='AAPL',
        quantity=10,  # 10 contracts = 1000 shares
        option_type='call',
        strike=150.0,
        expiration=0.25,  # 3 months
        underlying_price=155.0,
        volatility=0.30
    ))
    
    # Add AAPL stock to hedge
    portfolio.add_stock_position(Position(
        symbol='AAPL',
        quantity=-500,  # Short 500 shares
        average_price=155.0,
        current_price=155.0
    ))
    
    metrics = portfolio.get_risk_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")
\`\`\`

---

## Risk Limits and Enforcement

\`\`\`python
"""
Risk Limit System
Enforce limits and generate alerts
"""

from dataclasses import dataclass
from typing import List
from enum import Enum

class LimitType(Enum):
    POSITION_SIZE = "POSITION_SIZE"
    NOTIONAL_EXPOSURE = "NOTIONAL_EXPOSURE"
    VAR = "VAR"
    DELTA = "DELTA"
    GAMMA = "GAMMA"
    MAX_LOSS = "MAX_LOSS"

@dataclass
class RiskLimit:
    """Risk limit definition"""
    limit_type: LimitType
    symbol: Optional[str]  # None for portfolio-level limits
    soft_limit: float  # Warning threshold
    hard_limit: float  # Absolute maximum
    current_value: float = 0.0
    
    @property
    def utilization (self) -> float:
        """How much of limit is used (%)"""
        return (self.current_value / self.hard_limit) * 100 if self.hard_limit != 0 else 0
    
    @property
    def breach_status (self) -> str:
        """Check if limit breached"""
        if self.current_value > self.hard_limit:
            return "HARD_BREACH"
        elif self.current_value > self.soft_limit:
            return "SOFT_BREACH"
        else:
            return "OK"

class RiskLimitMonitor:
    """
    Monitor and enforce risk limits
    Generate alerts when limits approached/breached
    """
    
    def __init__(self):
        self.limits: List[RiskLimit] = []
        self.alerts: List[dict] = []
    
    def add_limit (self, limit: RiskLimit):
        """Add risk limit"""
        self.limits.append (limit)
    
    def update_limits (self, pnl_engine: PLEngine, greeks: dict, var: float):
        """Update all limit values"""
        for limit in self.limits:
            if limit.limit_type == LimitType.POSITION_SIZE:
                if limit.symbol:
                    pos = pnl_engine.positions.get (limit.symbol)
                    limit.current_value = abs (pos.quantity) if pos else 0
            
            elif limit.limit_type == LimitType.NOTIONAL_EXPOSURE:
                if limit.symbol:
                    pos = pnl_engine.positions.get (limit.symbol)
                    limit.current_value = abs (pos.market_value) if pos else 0
                else:
                    # Total portfolio exposure
                    limit.current_value = sum(
                        abs (pos.market_value) for pos in pnl_engine.positions.values()
                    )
            
            elif limit.limit_type == LimitType.VAR:
                limit.current_value = var
            
            elif limit.limit_type == LimitType.DELTA:
                limit.current_value = abs (greeks.get('delta', 0))
            
            elif limit.limit_type == LimitType.GAMMA:
                limit.current_value = abs (greeks.get('gamma', 0))
            
            elif limit.limit_type == LimitType.MAX_LOSS:
                limit.current_value = abs (min (pnl_engine.get_total_pnl(), 0))
            
            # Check for breaches
            status = limit.breach_status
            if status != "OK":
                self.generate_alert (limit, status)
    
    def generate_alert (self, limit: RiskLimit, status: str):
        """Generate alert for limit breach"""
        alert = {
            'timestamp': datetime.now(),
            'limit_type': limit.limit_type.value,
            'symbol': limit.symbol,
            'status': status,
            'current': limit.current_value,
            'soft_limit': limit.soft_limit,
            'hard_limit': limit.hard_limit,
            'utilization': limit.utilization,
            'message': f"{status}: {limit.limit_type.value} = {limit.current_value:.2f} (limit: {limit.hard_limit:.2f})"
        }
        
        self.alerts.append (alert)
        print(f"ALERT: {alert['message']}")
    
    def get_limit_summary (self) -> pd.DataFrame:
        """Get summary of all limits"""
        data = []
        for limit in self.limits:
            data.append({
                'type': limit.limit_type.value,
                'symbol': limit.symbol or 'PORTFOLIO',
                'current': limit.current_value,
                'soft_limit': limit.soft_limit,
                'hard_limit': limit.hard_limit,
                'utilization_%': limit.utilization,
                'status': limit.breach_status
            })
        return pd.DataFrame (data)

# Example
if __name__ == "__main__":
    monitor = RiskLimitMonitor()
    
    # Add limits
    monitor.add_limit(RiskLimit(
        limit_type=LimitType.POSITION_SIZE,
        symbol='AAPL',
        soft_limit=1000,
        hard_limit=1500
    ))
    
    monitor.add_limit(RiskLimit(
        limit_type=LimitType.NOTIONAL_EXPOSURE,
        symbol=None,  # Portfolio level
        soft_limit=1_000_000,
        hard_limit=1_500_000
    ))
    
    monitor.add_limit(RiskLimit(
        limit_type=LimitType.VAR,
        symbol=None,
        soft_limit=50_000,
        hard_limit=100_000
    ))
    
    # Update with current values
    pnl_engine = PLEngine()  # Assume populated
    greeks = {'delta': 500}
    var = 75_000
    
    monitor.update_limits (pnl_engine, greeks, var)
    
    # Print summary
    print(monitor.get_limit_summary())
\`\`\`

---

## Stress Testing

\`\`\`python
"""
Stress Testing Framework
Test portfolio under extreme scenarios
"""

class StressTester:
    """
    Stress test portfolio under various scenarios
    """
    
    def __init__(self, positions: Dict[str, Position]):
        self.positions = positions
    
    def scenario_shock(
        self,
        symbol_shocks: Dict[str, float]  # symbol -> % change
    ) -> dict:
        """
        Apply price shocks to portfolio
        Returns: P&L impact
        """
        initial_value = sum (pos.market_value for pos in self.positions.values())
        
        shocked_value = 0
        for symbol, pos in self.positions.items():
            shock = symbol_shocks.get (symbol, 0)
            new_price = pos.current_price * (1 + shock)
            shocked_value += pos.quantity * new_price
        
        pnl_impact = shocked_value - initial_value
        
        return {
            'initial_value': initial_value,
            'shocked_value': shocked_value,
            'pnl_impact': pnl_impact,
            'return_%': (pnl_impact / initial_value) * 100
        }
    
    def standard_scenarios (self) -> pd.DataFrame:
        """
        Run standard stress scenarios
        - Market crash (-20%)
        - Flash crash (-10%)
        - Volatility spike (VIX +50%)
        - Sector rotation (tech -15%, value +10%)
        """
        scenarios = []
        
        # Scenario 1: Market crash
        all_symbols_shock = {s: -0.20 for s in self.positions.keys()}
        result = self.scenario_shock (all_symbols_shock)
        scenarios.append({
            'scenario': 'Market Crash (-20%)',
            **result
        })
        
        # Scenario 2: Flash crash
        all_symbols_shock = {s: -0.10 for s in self.positions.keys()}
        result = self.scenario_shock (all_symbols_shock)
        scenarios.append({
            'scenario': 'Flash Crash (-10%)',
            **result
        })
        
        # Add more scenarios...
        
        return pd.DataFrame (scenarios)

# Example
tester = StressTester (positions)
stress_results = tester.standard_scenarios()
print(stress_results)
\`\`\`

---

## Production Risk Dashboard

\`\`\`
Architecture:

Risk Service (Python)
    ↓
Redis (real-time metrics)
    ↓
Grafana Dashboard

Metrics to monitor:
- Total P&L (realized + unrealized)
- Position sizes by symbol
- Portfolio Greeks
- VaR and CVaR
- Limit utilization
- Number of alerts
\`\`\`

---

## Summary

A production risk system requires:

1. **Real-time P&L**: Update on every tick, track realized/unrealized
2. **VaR/CVaR**: Estimate tail risk, use multiple methods
3. **Greeks aggregation**: Monitor options risk across portfolio
4. **Limit enforcement**: Soft warnings, hard stops, automated risk reduction
5. **Stress testing**: Test portfolio under extreme scenarios
6. **Monitoring**: Real-time dashboard with alerts

In the next section, we'll design high-frequency trading systems with sub-microsecond latency requirements.
`,
};
