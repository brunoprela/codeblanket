export const derivativeRiskManagement = {
  title: 'Derivative Risk Management',
  id: 'derivative-risk-management',
  content: `
# Derivative Risk Management

## Introduction

Derivative risk management involves identifying, measuring, and controlling risks in derivatives portfolios to prevent catastrophic losses.

**Why critical for engineers**:
- 2008 crisis showed derivative risk gone wrong
- VaR and stress testing required by regulators
- Real-time risk monitoring systems
- Automated limit enforcement

**What you'll build**: VaR calculator, stress testing engine, risk limit monitor, exposure aggregator.

---

## Value at Risk (VaR)

**Definition**: Maximum loss at confidence level over time horizon.

**Example**: 95% 1-day VaR = $1M
- 95% confident loss won't exceed $1M tomorrow
- 5% chance loss exceeds $1M (1 in 20 days)

### VaR Methodologies

**1. Historical Simulation**:
\`\`\`
Apply past 250 days of market moves to current portfolio
Calculate P&L for each scenario
5th percentile = 95% VaR
\`\`\`

**2. Parametric (Variance-Covariance)**:
\`\`\`
VaR = Portfolio_value × σ × Z_α

Where:
σ = Portfolio volatility (from covariance matrix)
Z_α = 1.65 for 95% confidence (normal distribution)
\`\`\`

**3. Monte Carlo Simulation**:
\`\`\`
Simulate 10,000 future scenarios
Reprice portfolio under each scenario
5th percentile = 95% VaR
\`\`\`

---

## Stress Testing

**Purpose**: Test portfolio under extreme but plausible scenarios.

### Scenario Types

**1. Historical Scenarios**:
- 1987 Black Monday (stocks -22%)
- 1994 Bond crash (rates +300bp)
- 1998 LTCM (liquidity crisis)
- 2008 Financial crisis
- 2020 COVID crash (VIX 80)

**2. Hypothetical Scenarios**:
- Rates +200bp parallel shift
- Credit spreads widen 150bp
- Equity crash -30%
- Currency crisis (EUR/USD ±20%)
- Volatility spike (VIX → 60)

**3. Reverse Stress Tests**:
- Start with outcome (e.g., lose $100M)
- Work backwards: What scenario causes this?
- Identifies vulnerabilities

---

## Risk Limits

**Types of Limits**:

**1. Notional Limits**:
\`\`\`
Max $500M derivatives notional per desk
Prevents excessive leverage
\`\`\`

**2. Greek Limits**:
\`\`\`
Max DV01: ±$1M (interest rate sensitivity)
Max Vega: ±$500K (volatility sensitivity)
Max Delta: ±1,000 shares equivalent
\`\`\`

**3. VaR Limits**:
\`\`\`
Max 1-day VaR: $10M
Ensures manageable daily risk
\`\`\`

**4. Concentration Limits**:
\`\`\`
Max 10% of portfolio in single issuer
Max 25% in single sector
Prevents concentration risk
\`\`\`

**5. Loss Limits**:
\`\`\`
Stop-loss: Close position if down 15%
Daily loss limit: $5M
Prevents runaway losses
\`\`\`

---

## Regulatory Requirements

### Basel III

**Capital requirements** for derivatives:

**SA-CCR (Standardized Approach)**:
\`\`\`
EAD = α × (RC + PFE)

Where:
RC = Replacement cost (current MTM)
PFE = Potential future exposure
α = 1.4

Capital = 8% × EAD × Risk_weight
\`\`\`

**CVA Capital**: Capital for counterparty credit risk

### Dodd-Frank

**Requirements**:
- Central clearing for standardized derivatives
- Margin requirements (initial + variation)
- Reporting to swap data repositories
- Business conduct standards

---

## Python: VaR Calculator

\`\`\`python
"""
Value at Risk (VaR) Calculation Engine
"""
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)


class VaRMethod(Enum):
    """VaR calculation methodologies"""
    HISTORICAL = "historical_simulation"
    PARAMETRIC = "variance_covariance"
    MONTE_CARLO = "monte_carlo"


@dataclass
class Position:
    """Trading position"""
    instrument_id: str
    market_value: float
    instrument_type: str  # 'bond', 'option', 'swap'
    delta: float = 0.0  # For derivatives
    gamma: float = 0.0
    vega: float = 0.0


class VaRCalculator:
    """
    Calculate Value at Risk using multiple methodologies
    
    Example:
        >>> positions = [
        ...     Position('BOND_1', 10_000_000, 'bond', delta=7.0),
        ...     Position('SWAP_1', 5_000_000, 'swap', delta=5.5)
        ... ]
        >>> calc = VaRCalculator(positions, confidence=0.95, horizon_days=1)
        >>> var = calc.calculate(method=VaRMethod.HISTORICAL, historical_returns=returns_df)
        >>> print(f"95% 1-day VaR: \\$\{var:,.0f}")
        """
    
    def __init__(
    self,
    positions: List[Position],
    confidence: float = 0.95,
    horizon_days: int = 1
):
"""
        Initialize VaR calculator

Args:
positions: List of portfolio positions
confidence: Confidence level(e.g., 0.95 for 95 %)
    horizon_days: Time horizon in days
"""
if not 0 < confidence < 1:
            raise ValueError("Confidence must be between 0 and 1")
if horizon_days < 1:
            raise ValueError("Horizon must be at least 1 day")

self.positions = positions
self.confidence = confidence
self.horizon_days = horizon_days
        
        # Portfolio value
self.portfolio_value = sum(p.market_value for p in positions)

    logger.info(
        f"Initialized VaR: \${self.portfolio_value / 1e6:.1f
}M portfolio, "
            f"{confidence*100:.0f}% confidence, {horizon_days}day horizon"
    )
    
    def historical_var(
        self,
        historical_returns: pd.DataFrame,
        lookback_days: int = 250
) -> float:
"""
        Calculate VaR using historical simulation

Method:
1. Apply historical market moves to current portfolio
2. Calculate hypothetical P & L for each scenario
        3. VaR = percentile of P & L distribution

Args:
historical_returns: DataFrame with historical returns
lookback_days: Number of historical days to use

Returns:
VaR(positive number represents loss)
"""
if len(historical_returns) < lookback_days:
        logger.warning(
                f"Insufficient historical data: {len(historical_returns)} < {lookback_days}"
        )
lookback_days = len(historical_returns)
        
        # Use last N days
returns = historical_returns.tail(lookback_days)
        
        # Calculate hypothetical portfolio returns
portfolio_returns = self._calculate_portfolio_returns(returns)
        
        # Scale to horizon(square root of time)
scaled_returns = portfolio_returns * np.sqrt(self.horizon_days)
        
        # VaR = percentile
percentile = (1 - self.confidence) * 100
var_return = np.percentile(scaled_returns, percentile)
        
        # Convert to dollar VaR(negative return = loss)
var_dollar = abs(var_return * self.portfolio_value)

logger.debug(
        f"Historical VaR: {var_dollar/1e6:.2f}M "
            f"({var_return*100:.2f}% of portfolio)"
)

return var_dollar
    
    def parametric_var(
        self,
        expected_return: float = 0.0,
        volatility: float = 0.01  # 1 % daily vol
) -> float:
"""
        Calculate VaR using variance-covariance(parametric) method
        
        Assumes normal distribution:
VaR = Portfolio_value × σ × Z_α × √T

Where:
- σ = portfolio volatility
        - Z_α = standard normal quantile(e.g., 1.65 for 95 %)
        - T = time horizon

Args:
expected_return: Expected daily return
volatility: Daily portfolio volatility

Returns:
VaR(positive number)
"""
        # Z - score for confidence level
        z_score = norm.ppf(1 - self.confidence)  # Negative for left tail
        
        # Time scaling
horizon_vol = volatility * np.sqrt(self.horizon_days)
horizon_return = expected_return * self.horizon_days
        
        # VaR formula
var_return = horizon_return + z_score * horizon_vol
        
        # Dollar VaR
var_dollar = abs(var_return * self.portfolio_value)

logger.debug(
        f"Parametric VaR: \${var_dollar/1e6:.2f}M "
            f"(vol={volatility*100:.2f}%, z={z_score:.2f})"
)

return var_dollar
    
    def monte_carlo_var(
        self,
        num_simulations: int = 10000,
        volatility: float = 0.01,
        expected_return: float = 0.0
) -> float:
"""
        Calculate VaR using Monte Carlo simulation

Method:
1. Simulate N future price paths
2. Reprice portfolio under each scenario
3. VaR = percentile of P & L distribution

Args:
num_simulations: Number of Monte Carlo paths
volatility: Daily volatility
expected_return: Expected daily return

Returns:
VaR(positive number)
"""
        # Simulate returns(geometric Brownian motion)
dt = self.horizon_days / 252  # Convert days to years
        
        # Random shocks
z = np.random.standard_normal(num_simulations)
        
        # Simulated returns
simulated_returns = (
        expected_return * dt +
        volatility * np.sqrt(dt) * z
)
        
        # Portfolio P & L
portfolio_pnl = simulated_returns * self.portfolio_value
        
        # VaR = percentile of losses
percentile = (1 - self.confidence) * 100
var_dollar = abs(np.percentile(portfolio_pnl, percentile))

logger.debug(
        f"Monte Carlo VaR ({num_simulations} sims): \${var_dollar/1e6:.2f}M"
)

return var_dollar
    
    def incremental_var(
        self,
        position_index: int,
        method: VaRMethod,
        ** kwargs
) -> float:
"""
        Calculate incremental VaR(marginal contribution of position)
        
        Incremental VaR = VaR(portfolio) - VaR(portfolio without position)

Args:
position_index: Index of position to analyze
method: VaR calculation method
        ** kwargs: Additional args for VaR method
        
        Returns:
            Incremental VaR(positive = position adds risk)
"""
        # Calculate full portfolio VaR
full_var = self.calculate(method, ** kwargs)
        
        # Remove position temporarily
removed_position = self.positions.pop(position_index)
        
        # Recalculate VaR
reduced_var = self.calculate(method, ** kwargs)
        
        # Restore position
self.positions.insert(position_index, removed_position)
        
        # Incremental VaR
incremental = full_var - reduced_var

logger.debug(
        f"Incremental VaR for {removed_position.instrument_id}: "
            f"\${incremental/1e6:.2f}M"
)

return incremental
    
    def component_var(
        self,
        method: VaRMethod,
        ** kwargs
) -> Dict[str, float]:
"""
        Calculate component VaR for each position
        
        Shows how much each position contributes to total VaR

Returns:
            Dict mapping position ID to component VaR
"""
components = {}

for i, position in enumerate(self.positions):
        inc_var = self.incremental_var(i, method, ** kwargs)
components[position.instrument_id] = inc_var

return components
    
    def calculate(self, method: VaRMethod, ** kwargs) -> float:
"""
        Calculate VaR using specified method

Args:
method: VaR methodology
        ** kwargs: Method - specific parameters

Returns:
VaR in dollars
"""
if method == VaRMethod.HISTORICAL:
        if 'historical_returns' not in kwargs:
                raise ValueError("Historical method requires 'historical_returns'")
return self.historical_var(** kwargs)
        
        elif method == VaRMethod.PARAMETRIC:
return self.parametric_var(** kwargs)
        
        elif method == VaRMethod.MONTE_CARLO:
return self.monte_carlo_var(** kwargs)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _calculate_portfolio_returns(
        self,
        market_returns: pd.DataFrame
) -> np.ndarray:
"""
        Calculate portfolio returns from market returns

Simplified: assumes positions are linear in market factor
"""
        # If single market factor
if isinstance(market_returns, pd.Series):
        market_returns = market_returns.to_frame('market')
        
        # Weighted average(simplified)
weights = np.array([p.market_value for p in self.positions])
weights = weights / weights.sum()
        
        # Portfolio return = weighted market return
        #(Assumes all positions move with market)
port_returns = market_returns.iloc[:, 0].values

return port_returns
    
    def backtest(
        self,
        actual_returns: pd.Series,
        method: VaRMethod,
        ** kwargs
) -> Dict:
"""
        Backtest VaR model against actual returns

Measures:
- Exceedances: How often actual loss > VaR
        - Expected: Should be(1 - confidence) × num_days

Args:
actual_returns: Historical actual portfolio returns
method: VaR method to test
        ** kwargs: Method - specific parameters

Returns:
            Dict with backtest statistics
"""
        # Calculate VaR for each day(rolling)
        var_estimates = []
        
        for i in range(len(actual_returns) - 250):
            # Use trailing 250 days for VaR estimation
            if method == VaRMethod.HISTORICAL:
                hist_data = actual_returns.iloc[i: i + 250]
kwargs['historical_returns'] = hist_data

var = self.calculate(method, ** kwargs)
var_estimates.append(var)
        
        # Actual losses (convert to dollars)
actual_pnl = actual_returns.iloc[250:].values * self.portfolio_value
        
        # Exceedances(actual loss > VaR)
exceedances = np.sum(actual_pnl < -np.array(var_estimates))
        
        # Expected exceedances
expected_exceedances = len(var_estimates) * (1 - self.confidence)
        
        # Backtest result
results = {
        'num_observations': len(var_estimates),
        'exceedances': exceedances,
        'expected_exceedances': expected_exceedances,
        'exceedance_rate': exceedances / len(var_estimates),
        'expected_rate': 1 - self.confidence,
        'pass': abs(exceedances - expected_exceedances) <= 3 * np.sqrt(expected_exceedances),
}

logger.info(
        f"Backtest: {exceedances} exceedances vs {expected_exceedances:.1f} expected "
            f"({'PASS' if results['pass'] else 'FAIL'})"
)

return results


# Example usage
if __name__ == "__main__":
        print("=== VaR Calculation Example ===\\n")
    
    # Portfolio positions
positions = [
        Position('BOND_10YR', 10_000_000, 'bond', delta = 7.5),
        Position('BOND_5YR', 5_000_000, 'bond', delta = 4.5),
        Position('SWAP_7YR', 3_000_000, 'swap', delta = 6.0),
]
    
    # Create calculator
calc = VaRCalculator(
        positions = positions,
        confidence = 0.95,
        horizon_days = 1
)

print(f"Portfolio Value: \\$\{calc.portfolio_value/1e6:.1f}M")
print(f"Confidence Level: {calc.confidence*100:.0f}%")
print(f"Time Horizon: {calc.horizon_days} day\\n")
    
    # Parametric VaR
param_var = calc.parametric_var(volatility = 0.008)  # 0.8 % daily vol
print(f"Parametric VaR: \\$\{param_var/1e6:.2f}M ({param_var/calc.portfolio_value*100:.2f}%)")
    
    # Monte Carlo VaR
mc_var = calc.monte_carlo_var(num_simulations = 10000, volatility = 0.008)
print(f"Monte Carlo VaR: \\$\{mc_var/1e6:.2f}M ({mc_var/calc.portfolio_value*100:.2f}%)")
    
    # Generate fake historical returns for demonstration
    np.random.seed(42)
    historical_returns = pd.DataFrame({
        'market': np.random.normal(0.0001, 0.008, 500)
})
    
    hist_var = calc.historical_var(historical_returns, lookback_days = 250)
print(f"Historical VaR: \\$\{hist_var/1e6:.2f}M ({hist_var/calc.portfolio_value*100:.2f}%)")
    
    # Component VaR
print("\\n=== Component VaR Analysis ===\\n")

components = calc.component_var(VaRMethod.PARAMETRIC, volatility = 0.008)

for position_id, comp_var in components.items():
        print(f"{position_id:15}: \\$\{comp_var/1e6:.2f}M ({comp_var/calc.portfolio_value*100:.2f}%)")
    
    # Backtesting
print("\\n=== VaR Backtesting ===\\n")

actual_returns = pd.Series(np.random.normal(0.0001, 0.008, 500))

backtest_results = calc.backtest(
        actual_returns,
        VaRMethod.PARAMETRIC,
        volatility = 0.008
)

print(f"Observations: {backtest_results['num_observations']}")
print(f"Exceedances: {backtest_results['exceedances']}")
print(f"Expected: {backtest_results['expected_exceedances']:.1f}")
print(f"Exceedance Rate: {backtest_results['exceedance_rate']*100:.2f}%")
print(f"Expected Rate: {backtest_results['expected_rate']*100:.2f}%")
print(f"Result: {'PASS ✓' if backtest_results['pass'] else 'FAIL ✗'}")
\`\`\`

---

## Model Risk

**Model risk** = Risk that models are wrong or misused.

### Sources

**1. Model Assumptions**:
- Normal distributions (fat tails underestimated)
- Constant volatility (actually time-varying)
- Constant correlations (spike to 1 in crisis)

**2. Parameter Risk**:
- Wrong volatility input
- Correlation matrix errors
- Yield curve mis-specification

**3. Implementation Risk**:
- Coding bugs
- Data errors
- Wrong formula

### Mitigation

**Model validation**: Independent team tests models

**Backtesting**: Compare model predictions to reality

**Sensitivity analysis**: Test key parameters

**Model reserves**: Extra capital for model uncertainty

---

## Key Takeaways

1. **VaR**: Maximum loss at confidence level (95% 1-day typical), methods: historical, parametric, Monte Carlo
2. **Stress testing**: Extreme scenarios (historical, hypothetical, reverse), identify vulnerabilities
3. **Risk limits**: Notional, Greeks, VaR, concentration, loss limits prevent excessive risk
4. **Regulatory**: Basel III (SA-CCR capital), Dodd-Frank (clearing, margin, reporting)
5. **Model risk**: Assumptions, parameters, implementation errors, mitigate via validation, backtesting
6. **Real-time monitoring**: Automated systems check limits continuously, generate alerts

**Next Section**: Project - Build comprehensive fixed income analytics platform integrating all concepts.
`,
};
