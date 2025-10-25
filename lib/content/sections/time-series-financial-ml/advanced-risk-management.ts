export const advancedRiskManagement = {
  title: 'Advanced Risk Management',
  id: 'advanced-risk-management',
  content: `
# Advanced Risk Management

## Introduction

Advanced risk management is what separates amateurs from professionals. While basic risk management handles position sizing and stop-losses, advanced techniques address:

- **Tail Risk**: Fat-tailed distributions and black swan events
- **Portfolio Correlation**: Hedging and diversification optimization
- **Dynamic Risk**: Adjusting to changing market regimes
- **Stress Testing**: What-if scenarios and Monte Carlo simulation
- **Leverage Management**: Risk parity and volatility targeting
- **Liquidity Risk**: Managing execution in stressed markets

**Why It Matters**:
- **2008 Financial Crisis**: Correlation → 1 (diversification failed)
- **2020 COVID Crash**: -35% in 3 weeks
- **2021 Archegos**: $10B loss in 2 days
- **Fat Tails**: Market moves > 3σ happen 10x more than normal distribution predicts

---

## Value at Risk (VaR) & Expected Shortfall

### VaR Calculation Methods

\`\`\`python
"""
Complete VaR and CVaR Implementation
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union

class ValueAtRiskCalculator:
    """
    Calculate Value at Risk using multiple methods
    
    VaR = Maximum expected loss at given confidence level over time horizon
    """
    
    @staticmethod
    def parametric_var (returns: Union[pd.Series, np.ndarray],
                      confidence: float = 0.95,
                      horizon: int = 1) -> float:
        """
        Parametric VaR (assumes normal distribution)
        
        Fast but inaccurate for fat-tailed distributions
        
        Args:
            returns: Historical returns
            confidence: Confidence level (0.95 = 95%)
            horizon: Time horizon in days
        
        Returns:
            VaR as fraction
        """
        mean = np.mean (returns)
        std = np.std (returns)
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence)
        
        # Scale for horizon
        var = (mean + z_score * std) * np.sqrt (horizon)
        
        return var
    
    @staticmethod
    def historical_var (returns: Union[pd.Series, np.ndarray],
                      confidence: float = 0.95,
                      horizon: int = 1) -> float:
        """
        Historical VaR (uses actual historical distribution)
        
        More accurate for non-normal distributions
        No assumptions about distribution shape
        """
        # For multi-day horizon, use overlapping windows
        if horizon > 1:
            rolling_returns = pd.Series (returns).rolling (horizon).sum().dropna()
            returns_to_use = rolling_returns
        else:
            returns_to_use = returns
        
        var = np.percentile (returns_to_use, (1 - confidence) * 100)
        return var
    
    @staticmethod
    def monte_carlo_var (returns: Union[pd.Series, np.ndarray],
                       confidence: float = 0.95,
                       horizon: int = 1,
                       n_simulations: int = 10000) -> float:
        """
        Monte Carlo VaR
        
        Simulates future returns based on historical parameters
        """
        mean = np.mean (returns)
        std = np.std (returns)
        
        # Simulate future paths
        simulated = np.random.normal (mean, std, (n_simulations, horizon))
        
        # Calculate cumulative returns for each path
        cumulative = simulated.sum (axis=1)
        
        var = np.percentile (cumulative, (1 - confidence) * 100)
        return var
    
    @staticmethod
    def cornish_fisher_var (returns: Union[pd.Series, np.ndarray],
                          confidence: float = 0.95,
                          horizon: int = 1) -> float:
        """
        Cornish-Fisher VaR (adjusts for skewness and kurtosis)
        
        Better for fat-tailed, skewed distributions
        """
        mean = np.mean (returns)
        std = np.std (returns)
        skew = stats.skew (returns)
        kurt = stats.kurtosis (returns)
        
        # Standard normal quantile
        z = stats.norm.ppf(1 - confidence)
        
        # Cornish-Fisher adjustment
        z_cf = (z + (z**2 - 1) * skew / 6 +
               (z**3 - 3*z) * kurt / 24 -
               (2*z**3 - 5*z) * skew**2 / 36)
        
        var = (mean + z_cf * std) * np.sqrt (horizon)
        return var


class ConditionalValueAtRisk:
    """
    CVaR / Expected Shortfall: Expected loss beyond VaR
    
    CVaR always > VaR
    More informative about tail risk
    """
    
    @staticmethod
    def cvar (returns: Union[pd.Series, np.ndarray],
            confidence: float = 0.95) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall)
        
        CVaR = Average loss when loss exceeds VaR
        """
        var = ValueAtRiskCalculator.historical_var (returns, confidence)
        
        # Average of returns worse than VaR
        tail_losses = returns[returns <= var]
        cvar = tail_losses.mean()
        
        return cvar
    
    @staticmethod
    def marginal_cvar (returns: Union[pd.Series, np.ndarray],
                     position_returns: Union[pd.Series, np.ndarray],
                     confidence: float = 0.95) -> float:
        """
        Marginal CVaR: How much does this position contribute to portfolio CVaR?
        
        Used for risk budgeting
        """
        portfolio_cvar = ConditionalValueAtRisk.cvar (returns, confidence)
        
        # Calculate correlation in tail
        var = ValueAtRiskCalculator.historical_var (returns, confidence)
        tail_periods = returns <= var
        
        tail_correlation = np.corrcoef(
            returns[tail_periods],
            position_returns[tail_periods]
        )[0, 1]
        
        # Marginal CVaR
        position_vol = np.std (position_returns)
        portfolio_vol = np.std (returns)
        
        marginal = tail_correlation * position_vol / portfolio_vol
        
        return marginal * portfolio_cvar


# ============================================================================
# EXAMPLE: CALCULATE ALL VAR MEASURES
# ============================================================================

# Generate returns with fat tails
np.random.seed(42)
n = 1000

# Mix of normal and extreme events (fat tails)
normal_returns = np.random.normal(0.001, 0.015, int (n * 0.95))
extreme_returns = np.random.normal(0, 0.05, int (n * 0.05))
returns = np.concatenate([normal_returns, extreme_returns])
np.random.shuffle (returns)

print("="*70)
print("VALUE AT RISK COMPARISON")
print("="*70)

calculator = ValueAtRiskCalculator()

# Calculate VaR using different methods
var_parametric = calculator.parametric_var (returns, confidence=0.95)
var_historical = calculator.historical_var (returns, confidence=0.95)
var_monte_carlo = calculator.monte_carlo_var (returns, confidence=0.95, n_simulations=10000)
var_cornish_fisher = calculator.cornish_fisher_var (returns, confidence=0.95)

print(f"\\n1-Day VaR (95% confidence):")
print(f"  Parametric (Normal):    {var_parametric:.4f} ({var_parametric:.2%})")
print(f"  Historical:             {var_historical:.4f} ({var_historical:.2%})")
print(f"  Monte Carlo:            {var_monte_carlo:.4f} ({var_monte_carlo:.2%})")
print(f"  Cornish-Fisher:         {var_cornish_fisher:.4f} ({var_cornish_fisher:.2%})")

# Calculate CVaR
cvar_calc = ConditionalValueAtRisk()
cvar_95 = cvar_calc.cvar (returns, confidence=0.95)
cvar_99 = cvar_calc.cvar (returns, confidence=0.99)

print(f"\\nConditional VaR (Expected Shortfall):")
print(f"  CVaR 95%: {cvar_95:.4f} ({cvar_95:.2%})")
print(f"  CVaR 99%: {cvar_99:.4f} ({cvar_99:.2%})")

# Portfolio example with $100K
portfolio_value = 100000
var_dollar = abs (var_historical * portfolio_value)
cvar_dollar = abs (cvar_95 * portfolio_value)

print(f"\\nPortfolio Risk ($100,000):")
print(f"  1-Day VaR (95%): \${var_dollar:,.0f}")
print(f"  1-Day CVaR (95%): \${cvar_dollar:,.0f}")
print(f"  Worst expected day (99%): \${abs (cvar_99 * portfolio_value):,.0f}")

print("="*70)
\`\`\`

---

## Stress Testing & Scenario Analysis

\`\`\`python
"""
Comprehensive stress testing for portfolios
"""

class StressTester:
    """
    Test portfolio resilience to extreme scenarios
    """
    
    def __init__(self, positions: dict, prices: dict):
        """
        Args:
            positions: {symbol: quantity}
            prices: {symbol: current_price}
        """
        self.positions = positions
        self.prices = prices
        self.portfolio_value = sum (pos * prices[sym] for sym, pos in positions.items())
    
    def historical_stress_test (self) -> dict:
        """
        Test historical crisis scenarios
        """
        scenarios = {
            '1987_Black_Monday': {'SPY': -0.228, 'bonds': 0.02},
            '2000_Dot_Com_Crash': {'tech': -0.783, 'value': -0.20},
            '2008_Financial_Crisis': {'SPY': -0.569, 'credit': -0.40, 'real_estate': -0.70},
            '2020_COVID_Crash': {'SPY': -0.338, 'oil': -0.65, 'travel': -0.80},
            '2022_Rate_Hikes': {'SPY': -0.183, 'bonds': -0.13, 'tech': -0.32},
        }
        
        results = {}
        
        for scenario_name, shocks in scenarios.items():
            portfolio_loss = 0
            
            for symbol, quantity in self.positions.items():
                current_value = quantity * self.prices[symbol]
                
                # Apply shock (default to market shock if symbol not specified)
                shock = shocks.get (symbol, shocks.get('SPY', -0.20))
                shocked_value = current_value * (1 + shock)
                loss = current_value - shocked_value
                
                portfolio_loss += loss
            
            results[scenario_name] = {
                'loss': portfolio_loss,
                'loss_pct': portfolio_loss / self.portfolio_value,
                'new_value': self.portfolio_value - portfolio_loss
            }
        
        return results
    
    def factor_stress_test (self, factor_shocks: dict) -> dict:
        """
        Test factor-based shocks
        
        Args:
            factor_shocks: {'interest_rate': +0.02, 'volatility': +0.50, ...}
        """
        # Simplified: In reality, you'd have factor loadings (betas)
        results = {}
        
        # Interest rate shock
        if 'interest_rate' in factor_shocks:
            rate_shock = factor_shocks['interest_rate']
            # Duration-based impact on bonds
            bond_impact = -7.0 * rate_shock  # Duration ~7 years
            results['interest_rate'] = bond_impact
        
        # Volatility shock (VIX spike)
        if 'volatility' in factor_shocks:
            vol_shock = factor_shocks['volatility']
            # Options/leveraged positions affected
            results['volatility'] = -vol_shock * 0.5
        
        return results
    
    def correlation_breakdown_test (self) -> dict:
        """
        Test scenario where all correlations → 1
        
        (What happened in 2008: diversification failed)
        """
        # In crisis, everything moves together
        # Assume all assets drop by market average
        uniform_shock = -0.30  # 30% market-wide decline
        
        total_loss = self.portfolio_value * abs (uniform_shock)
        
        return {
            'scenario': 'Correlation Breakdown',
            'assumption': 'All correlations → 1.0',
            'loss': total_loss,
            'loss_pct': uniform_shock,
            'diversification_benefit': 0  # Diversification doesn't help
        }
    
    def liquidity_stress_test (self, liquidation_pct: float = 0.50) -> dict:
        """
        Test forced liquidation at bad prices
        
        Args:
            liquidation_pct: % of position to liquidate
        """
        # Model: Forced selling pushes price down
        # Price impact = β * √(quantity / daily_volume)
        
        liquidation_cost = 0
        
        for symbol, quantity in self.positions.items():
            liquidate_qty = quantity * liquidation_pct
            current_price = self.prices[symbol]
            
            # Simplified market impact
            # Real model would use: impact = λ * √(qty / volume)
            market_impact = 0.05  # 5% slippage on forced selling
            
            liquidation_price = current_price * (1 - market_impact)
            cost = liquidate_qty * (current_price - liquidation_price)
            
            liquidation_cost += cost
        
        return {
            'liquidation_pct': liquidation_pct,
            'liquidation_cost': liquidation_cost,
            'cost_bps': (liquidation_cost / (self.portfolio_value * liquidation_pct)) * 10000
        }
    
    def monte_carlo_stress_test (self, n_simulations: int = 10000,
                                correlation_matrix: np.ndarray = None) -> dict:
        """
        Monte Carlo simulation of portfolio under stress
        """
        if correlation_matrix is None:
            # Default: moderate correlation
            n_assets = len (self.positions)
            correlation_matrix = np.full((n_assets, n_assets), 0.3)
            np.fill_diagonal (correlation_matrix, 1.0)
        
        # Simulate correlated returns
        mean_returns = np.array([-0.05] * len (self.positions))  # Stress: negative drift
        volatilities = np.array([0.30] * len (self.positions))    # High vol
        
        # Cholesky decomposition for correlated random variables
        cov_matrix = np.outer (volatilities, volatilities) * correlation_matrix
        L = np.linalg.cholesky (cov_matrix)
        
        # Simulate
        simulated_losses = []
        
        for _ in range (n_simulations):
            z = np.random.normal(0, 1, len (self.positions))
            returns = mean_returns + L @ z
            
            portfolio_return = sum(
                (qty * self.prices[sym]) / self.portfolio_value * ret
                for (sym, qty), ret in zip (self.positions.items(), returns)
            )
            
            simulated_losses.append (portfolio_return)
        
        simulated_losses = np.array (simulated_losses)
        
        return {
            'mean_loss': simulated_losses.mean(),
            'var_95': np.percentile (simulated_losses, 5),
            'var_99': np.percentile (simulated_losses, 1),
            'worst_case': simulated_losses.min(),
            'best_case': simulated_losses.max()
        }


# ============================================================================
# EXAMPLE: COMPREHENSIVE STRESS TESTING
# ============================================================================

# Portfolio
positions = {
    'SPY': 100,
    'TLT': 50,   # Bonds
    'GLD': 20,   # Gold
}

prices = {
    'SPY': 450,
    'TLT': 95,
    'GLD': 180,
}

stress_tester = StressTester (positions, prices)

print("\\n" + "="*70)
print("STRESS TEST RESULTS")
print("="*70)

print(f"\\nPortfolio Value: \${stress_tester.portfolio_value:,.0f}")

# Historical scenarios
print("\\n1. Historical Crisis Scenarios:")
historical_results = stress_tester.historical_stress_test()

for scenario, result in historical_results.items():
    print(f"\\n{scenario}:")
    print(f"  Loss: \${result['loss']:,.0f} ({result['loss_pct']:.2%})")
    print(f"  New Value: \${result['new_value']:,.0f}")

# Correlation breakdown
print("\\n2. Correlation Breakdown (2008-style):")
corr_breakdown = stress_tester.correlation_breakdown_test()
print(f"  Scenario: {corr_breakdown['scenario']}")
print(f"  Loss: \${corr_breakdown['loss']:,.0f} ({corr_breakdown['loss_pct']:.2%})")

# Liquidity stress
print("\\n3. Forced Liquidation (50% of portfolio):")
liquidity_result = stress_tester.liquidity_stress_test (liquidation_pct=0.50)
print(f"  Liquidation Cost: \${liquidity_result['liquidation_cost']:,.0f}")
print(f"  Cost (bps): {liquidity_result['cost_bps']:.0f} bps")

# Monte Carlo
print("\\n4. Monte Carlo Stress Simulation (10,000 scenarios):")
mc_results = stress_tester.monte_carlo_stress_test (n_simulations=10000)
print(f"  Mean Loss: {mc_results['mean_loss']:.2%}")
print(f"  VaR 95%: {mc_results['var_95']:.2%}")
print(f"  VaR 99%: {mc_results['var_99']:.2%}")
print(f"  Worst Case: {mc_results['worst_case']:.2%}")

print("="*70)
\`\`\`

---

## Dynamic Risk Management

\`\`\`python
"""
Adjust risk exposure based on market conditions
"""

class DynamicRiskManager:
    """
    Dynamically adjust position sizes and leverage based on market regime
    """
    
    def __init__(self, target_volatility: float = 0.15):
        """
        Args:
            target_volatility: Target annual volatility (e.g., 0.15 = 15%)
        """
        self.target_volatility = target_volatility
    
    def volatility_targeting (self, current_volatility: float,
                            base_allocation: float) -> float:
        """
        Volatility Targeting: Scale position to maintain constant risk
        
        If vol doubles → cut position in half
        """
        scaling_factor = self.target_volatility / current_volatility
        adjusted_allocation = base_allocation * scaling_factor
        
        # Cap at reasonable levels
        adjusted_allocation = np.clip (adjusted_allocation, 0.1, 2.0)
        
        return adjusted_allocation
    
    def drawdown_based_scaling (self, current_drawdown: float,
                               max_drawdown_threshold: float = 0.10) -> float:
        """
        Reduce exposure as drawdown increases
        
        At max drawdown threshold → 50% exposure
        """
        if current_drawdown < 0:
            return 1.0
        
        if current_drawdown >= max_drawdown_threshold:
            return 0.5
        
        # Linear scaling
        scaling = 1.0 - 0.5 * (current_drawdown / max_drawdown_threshold)
        
        return scaling
    
    def regime_based_allocation (self, regime: str) -> dict:
        """
        Adjust allocation based on market regime
        
        Regimes: 'bull', 'bear', 'high_vol', 'low_vol', 'crisis'
        """
        allocations = {
            'bull': {'stocks': 0.70, 'bonds': 0.20, 'alternatives': 0.10},
            'bear': {'stocks': 0.30, 'bonds': 0.50, 'alternatives': 0.20},
            'high_vol': {'stocks': 0.40, 'bonds': 0.40, 'alternatives': 0.20},
            'low_vol': {'stocks': 0.60, 'bonds': 0.30, 'alternatives': 0.10},
            'crisis': {'stocks': 0.20, 'bonds': 0.60, 'alternatives': 0.20},
        }
        
        return allocations.get (regime, allocations['bull'])


# Example: Dynamic risk adjustment
drm = DynamicRiskManager (target_volatility=0.15)

# Current market conditions
current_vol = 0.25  # 25% vol (high)
base_allocation = 1.0  # 100% invested normally

# Volatility targeting
adjusted_allocation = drm.volatility_targeting (current_vol, base_allocation)
print(f"\\nVolatility Targeting:")
print(f"  Current Vol: {current_vol:.1%}")
print(f"  Target Vol: {drm.target_volatility:.1%}")
print(f"  Adjusted Allocation: {adjusted_allocation:.1%} (from {base_allocation:.1%})")

# Drawdown scaling
current_dd = 0.08  # 8% drawdown
dd_scaling = drm.drawdown_based_scaling (current_dd, max_drawdown_threshold=0.10)
print(f"\\nDrawdown Scaling:")
print(f"  Current Drawdown: {current_dd:.1%}")
print(f"  Position Scaling: {dd_scaling:.1%}")

# Regime-based
high_vol_allocation = drm.regime_based_allocation('high_vol')
print(f"\\nHigh Volatility Regime Allocation:")
for asset, weight in high_vol_allocation.items():
    print(f"  {asset}: {weight:.1%}")
\`\`\`

---

## Key Takeaways

**Risk Metrics Hierarchy**:
1. **VaR**: Quick estimate, but understates tail risk
2. **CVaR**: Better (actual expected loss in tail)
3. **Stress Testing**: Best (specific scenarios)
4. **Monte Carlo**: Most comprehensive

**Stress Testing Strategy**:
- Test **historical** crises (they can repeat)
- Test **hypothetical** worst-case scenarios
- Test **factor** shocks (rates, vol, credit)
- Test **liquidity** (forced selling)
- Test **correlations** (diversification failure)

**Dynamic Risk Management**:
- **Volatility targeting**: Scale down when vol spikes
- **Drawdown scaling**: Cut exposure in losses
- **Regime detection**: Adjust for market conditions
- **Leverage management**: Never over-leverage in stress

**Practical Implementation**:
- Calculate VaR daily
- Stress test weekly
- Adjust positions when thresholds breached
- Document all assumptions
- Review after large moves

**Remember**: Risk management is about surviving to trade another day. One bad tail event can wipe out years of gains.
`,
};
