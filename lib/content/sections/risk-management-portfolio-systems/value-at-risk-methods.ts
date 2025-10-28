export const valueAtRiskMethods = `
# Value at Risk (VaR) Methods

## Introduction

Value at Risk (VaR) is the most widely used risk metric in finance. Walk into any bank, hedge fund, or asset manager, and you'll see VaR numbers on dashboards, risk reports, and regulatory filings. Despite its limitations (which we'll discuss), VaR provides a simple, intuitive answer to the fundamental risk question:

**"How much could we lose?"**

More precisely, VaR answers: "What is the maximum loss over a given time horizon at a specified confidence level, under normal market conditions?"

## VaR Definition and Interpretation

### Mathematical Definition

VaR is a quantile of the loss distribution:

\`\`\`
VaR_α = inf{l : P(Loss > l) ≤ 1 - α}
\`\`\`

Where:
- α = confidence level (typically 95% or 99%)
- l = loss amount
- Loss = negative return

### Practical Interpretation

**Example**: 1-day 99% VaR = $1 million

This means:
- **With 99% confidence**, losses will not exceed $1M over the next day
- **1% of the time** (about 2-3 trading days per year), losses will exceed $1M
- **Says nothing** about how much you might lose in that 1% tail

### Three Key Parameters

1. **Time Horizon (H)**:
   - 1-day: Trading desks, market risk
   - 10-day: Basel regulatory capital
   - 1-month: Asset managers

2. **Confidence Level (α)**:
   - 95%: Common for internal risk management
   - 99%: Common for regulatory requirements
   - 99.9%: Conservative institutions

3. **Currency**:
   - Dollar VaR: Absolute loss amount
   - Percentage VaR: Relative to portfolio value

## VaR Methodologies

There are three main approaches to calculating VaR, each with trade-offs:

### 1. Historical Simulation VaR

The most intuitive method: use actual historical returns.

#### Methodology

1. **Collect Historical Data**: Typically 250-500 days of returns
2. **Apply to Current Portfolio**: Revalue portfolio using each historical return scenario
3. **Rank Scenarios**: Sort losses from worst to best
4. **Find Quantile**: Pick the appropriate percentile

#### Implementation

\`\`\`python
import numpy as np
import pandas as pd
from typing import Dict, List

class HistoricalVaR:
    def __init__(self, returns: pd.DataFrame, positions: Dict[str, float]):
        """
        Historical Simulation VaR Calculator
        
        Args:
            returns: DataFrame with columns for each asset, rows for dates
            positions: Dict mapping asset names to dollar positions
        """
        self.returns = returns
        self.positions = positions
        self.portfolio_value = sum(positions.values())
        
    def calculate_var(self, confidence_level: float = 0.99, horizon_days: int = 1) -> Dict:
        """
        Calculate VaR using historical simulation
        
        Returns:
            Dict with VaR, expected shortfall, and details
        """
        # Get position weights
        weights = np.array([self.positions[asset] / self.portfolio_value 
                           for asset in self.returns.columns])
        
        # Calculate portfolio returns for each historical scenario
        portfolio_returns = (self.returns @ weights).values
        
        # Scale to horizon (simple scaling - can improve with sqrt(t))
        horizon_returns = portfolio_returns * np.sqrt(horizon_days)
        
        # Convert returns to dollar P&L
        portfolio_pnl = horizon_returns * self.portfolio_value
        
        # Calculate VaR (negative because we want losses)
        var_percentile = (1 - confidence_level) * 100
        var_loss = np.percentile(portfolio_pnl, var_percentile)
        
        # Calculate Expected Shortfall (CVaR)
        tail_losses = portfolio_pnl[portfolio_pnl <= var_loss]
        expected_shortfall = tail_losses.mean() if len(tail_losses) > 0 else var_loss
        
        # Additional statistics
        worst_loss = portfolio_pnl.min()
        best_gain = portfolio_pnl.max()
        
        return {
            'var': abs(var_loss),
            'expected_shortfall': abs(expected_shortfall),
            'worst_historical': abs(worst_loss),
            'best_historical': best_gain,
            'confidence_level': confidence_level,
            'horizon_days': horizon_days,
            'portfolio_value': self.portfolio_value,
            'var_percentage': abs(var_loss) / self.portfolio_value,
            'num_scenarios': len(portfolio_pnl)
        }
    
    def var_breakdown(self, confidence_level: float = 0.99) -> pd.DataFrame:
        """
        VaR contribution by asset
        """
        weights = np.array([self.positions[asset] / self.portfolio_value 
                           for asset in self.returns.columns])
        
        # Portfolio returns
        portfolio_returns = self.returns @ weights
        
        # Calculate VaR
        var_percentile = (1 - confidence_level) * 100
        var_return = np.percentile(portfolio_returns, var_percentile)
        
        # Marginal VaR - how much each asset contributes
        contributions = []
        for asset in self.returns.columns:
            # Find scenarios where portfolio hits VaR
            var_scenarios = portfolio_returns <= var_return
            
            # Average return of this asset in those scenarios
            asset_return_in_var = self.returns.loc[var_scenarios, asset].mean()
            
            # Contribution = weight * return in VaR scenarios * portfolio value
            weight = self.positions[asset] / self.portfolio_value
            contribution = weight * asset_return_in_var * self.portfolio_value
            
            contributions.append({
                'asset': asset,
                'position': self.positions[asset],
                'weight': weight,
                'var_contribution': abs(contribution),
                'marginal_var': abs(asset_return_in_var)
            })
        
        return pd.DataFrame(contributions).sort_values('var_contribution', ascending=False)

# Example Usage
if __name__ == "__main__":
    # Sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Simulate returns for 5 assets
    returns_data = {
        'AAPL': np.random.normal(0.0005, 0.02, len(dates)),
        'GOOGL': np.random.normal(0.0006, 0.025, len(dates)),
        'MSFT': np.random.normal(0.0005, 0.018, len(dates)),
        'AMZN': np.random.normal(0.0007, 0.03, len(dates)),
        'SPY': np.random.normal(0.0004, 0.012, len(dates))
    }
    returns_df = pd.DataFrame(returns_data, index=dates)
    
    # Portfolio positions (dollar amounts)
    positions = {
        'AAPL': 500000,
        'GOOGL': 300000,
        'MSFT': 400000,
        'AMZN': 200000,
        'SPY': 600000
    }
    
    # Calculate VaR
    var_calc = HistoricalVaR(returns_df, positions)
    
    # 1-day 99% VaR
    var_results = var_calc.calculate_var(confidence_level=0.99, horizon_days=1)
    
    print("Historical VaR Analysis")
    print("="*50)
    print(f"Portfolio Value: ${var_results['portfolio_value']:,.0f}")
print(f"1-day 99% VaR: ${var_results['var']:,.0f} ({var_results['var_percentage']*100:.2f}%)")
print(f"Expected Shortfall: ${var_results['expected_shortfall']:,.0f}")
print(f"Worst Historical Loss: ${var_results['worst_historical']:,.0f}")
print(f"Based on {var_results['num_scenarios']} scenarios")
print()
    
    # VaR breakdown
breakdown = var_calc.var_breakdown(confidence_level = 0.99)
print("VaR Contribution by Asset:")
print(breakdown.to_string(index = False))
\`\`\`

#### Advantages

✅ **No Distribution Assumptions**: Uses actual returns  
✅ **Captures Fat Tails**: If they occurred historically  
✅ **Easy to Explain**: Simple conceptually  
✅ **Non-parametric**: Works for any asset class

#### Disadvantages

❌ **Limited by History**: Can't predict unprecedented events  
❌ **Sample Dependent**: Results vary with lookback period  
❌ **Ignores Current Conditions**: Treats all history equally  
❌ **Sparse Tail Data**: Few observations in extreme tail

### 2. Variance-Covariance (Parametric) VaR

Assumes returns follow a normal distribution.

#### Methodology

1. **Calculate Mean and Volatility**: From historical data
2. **Assume Normal Distribution**: Returns ~ N(μ, σ²)
3. **Apply Z-score**: VaR = μ - z_α × σ
4. **Scale to Portfolio**: Account for portfolio composition

#### Mathematical Foundation

For normally distributed returns:
\`\`\`
VaR_α = -(μ - z_α × σ) × Portfolio Value × √H

Where:
- μ = expected return
- σ = volatility
- z_α = normal quantile (1.65 for 95%, 2.33 for 99%)
- H = time horizon in days
\`\`\`

For portfolio of multiple assets:
\`\`\`
σ_portfolio = √(w^T × Σ × w)

Where:
- w = vector of weights
- Σ = covariance matrix
\`\`\`

#### Implementation

\`\`\`python
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict

class ParametricVaR:
    def __init__(self, returns: pd.DataFrame, positions: Dict[str, float]):
        """
        Variance-Covariance VaR Calculator
        
        Args:
            returns: DataFrame with asset returns
            positions: Dict of dollar positions
        """
        self.returns = returns
        self.positions = positions
        self.portfolio_value = sum(positions.values())
        
        # Calculate statistics
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.weights = self._calculate_weights()
        
    def _calculate_weights(self) -> np.ndarray:
        """Calculate portfolio weights"""
        return np.array([self.positions[asset] / self.portfolio_value 
                        for asset in self.returns.columns])
    
    def calculate_portfolio_statistics(self):
        """
        Calculate portfolio mean and volatility
        """
        # Portfolio expected return
        portfolio_mean = np.dot(self.weights, self.mean_returns)
        
        # Portfolio variance
        portfolio_variance = self.weights @ self.cov_matrix @ self.weights
        portfolio_vol = np.sqrt(portfolio_variance)
        
        return portfolio_mean, portfolio_vol
    
    def calculate_var(self, confidence_level: float = 0.99, 
                     horizon_days: int = 1, annualization: int = 252) -> Dict:
        """
        Calculate parametric VaR
        """
        # Get portfolio statistics
        portfolio_mean, portfolio_vol = self.calculate_portfolio_statistics()
        
        # Annualize
        annual_mean = portfolio_mean * annualization
        annual_vol = portfolio_vol * np.sqrt(annualization)
        
        # Get z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # Calculate VaR
        # VaR = -(mean - z * vol) * sqrt(horizon) * portfolio_value
        horizon_mean = annual_mean * (horizon_days / annualization)
        horizon_vol = annual_vol * np.sqrt(horizon_days / annualization)
        
        var_percentage = -(horizon_mean + z_score * horizon_vol)
        var_dollar = var_percentage * self.portfolio_value
        
        return {
            'var': abs(var_dollar),
            'var_percentage': abs(var_percentage),
            'portfolio_mean': portfolio_mean,
            'portfolio_vol': portfolio_vol,
            'annual_return': annual_mean,
            'annual_vol': annual_vol,
            'z_score': z_score,
            'confidence_level': confidence_level,
            'horizon_days': horizon_days,
            'portfolio_value': self.portfolio_value
        }
    
    def calculate_component_var(self, confidence_level: float = 0.99) -> pd.DataFrame:
        """
        Marginal and component VaR for each position
        """
        portfolio_mean, portfolio_vol = self.calculate_portfolio_statistics()
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # Marginal VaR = (Σ × w) / σ_p
        marginal_var = (self.cov_matrix @ self.weights) / portfolio_vol
        
        # Component VaR = weight × marginal VaR × total VaR
        var_percentage = abs(z_score * portfolio_vol)
        component_var = self.weights * marginal_var * var_percentage * self.portfolio_value
        
        results = []
        for i, asset in enumerate(self.returns.columns):
            results.append({
                'asset': asset,
                'position': self.positions[asset],
                'weight': self.weights[i],
                'marginal_var': marginal_var[i],
                'component_var': abs(component_var[i]),
                'percent_of_total_var': abs(component_var[i]) / (var_percentage * self.portfolio_value)
            })
        
        return pd.DataFrame(results).sort_values('component_var', ascending=False)
    
    def calculate_incremental_var(self, asset: str, position_change: float,
                                  confidence_level: float = 0.99) -> Dict:
        """
        How does VaR change if we add/remove position?
        """
        # Current VaR
        current_var = self.calculate_var(confidence_level)['var']
        
        # Modified positions
        modified_positions = self.positions.copy()
        modified_positions[asset] += position_change
        
        # Recalculate
        modified_calc = ParametricVaR(self.returns, modified_positions)
        new_var = modified_calc.calculate_var(confidence_level)['var']
        
        incremental_var = new_var - current_var
        
        return {
            'asset': asset,
            'position_change': position_change,
            'current_var': current_var,
            'new_var': new_var,
            'incremental_var': incremental_var,
            'incremental_var_per_dollar': incremental_var / abs(position_change)
        }

# Example
if __name__ == "__main__":
    # Sample data (same as before)
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    returns_data = {
        'AAPL': np.random.normal(0.0005, 0.02, len(dates)),
        'GOOGL': np.random.normal(0.0006, 0.025, len(dates)),
        'MSFT': np.random.normal(0.0005, 0.018, len(dates)),
        'AMZN': np.random.normal(0.0007, 0.03, len(dates)),
        'SPY': np.random.normal(0.0004, 0.012, len(dates))
    }
    returns_df = pd.DataFrame(returns_data, index=dates)
    
    positions = {
        'AAPL': 500000,
        'GOOGL': 300000,
        'MSFT': 400000,
        'AMZN': 200000,
        'SPY': 600000
    }
    
    # Calculate parametric VaR
    var_calc = ParametricVaR(returns_df, positions)
    var_results = var_calc.calculate_var(confidence_level=0.99)
    
    print("Parametric VaR Analysis")
    print("="*50)
    print(f"Portfolio Value: ${var_results['portfolio_value']:, .0f}")
print(f"Annual Return: {var_results['annual_return']*100:.2f}%")
print(f"Annual Volatility: {var_results['annual_vol']*100:.2f}%")
print(f"1-day 99% VaR: ${var_results['var']:,.0f} ({var_results['var_percentage']*100:.2f}%)")
print()
    
    # Component VaR
component_var = var_calc.calculate_component_var(confidence_level = 0.99)
print("Component VaR:")
print(component_var.to_string(index = False))
print()
    
    # Incremental VaR - what if we add $100k to AAPL ?
    incremental = var_calc.calculate_incremental_var('AAPL', 100000, 0.99)
    print(f"Incremental VaR for $100k AAPL:")
print(f"  Current VaR: ${incremental['current_var']:,.0f}")
print(f"  New VaR: ${incremental['new_var']:,.0f}")
print(f"  Incremental: ${incremental['incremental_var']:,.0f}")
\`\`\`

#### Advantages

✅ **Fast Computation**: Analytical formula  
✅ **Easy to Update**: Just need mean and covariance  
✅ **Smooth Results**: No jumps from dropping data  
✅ **Decomposable**: Easy to attribute to positions

#### Disadvantages

❌ **Normality Assumption**: Returns are NOT normal  
❌ **Underestimates Tail Risk**: Fat tails ignored  
❌ **Linear Relationships**: Doesn't work for options  
❌ **Constant Volatility**: Volatility changes over time

### 3. Monte Carlo Simulation VaR

Generate thousands of random scenarios based on statistical models.

#### Methodology

1. **Choose Distribution**: Normal, Student's t, or other
2. **Estimate Parameters**: From historical data
3. **Generate Scenarios**: Simulate thousands of return paths
4. **Revalue Portfolio**: Calculate P&L for each scenario
5. **Calculate VaR**: Find quantile of simulated P&L distribution

#### Implementation

\`\`\`python
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional

class MonteCarloVaR:
    def __init__(self, returns: pd.DataFrame, positions: Dict[str, float]):
        """
        Monte Carlo VaR Calculator
        
        Args:
            returns: Historical returns
            positions: Portfolio positions
        """
        self.returns = returns
        self.positions = positions
        self.portfolio_value = sum(positions.values())
        
        # Estimate parameters
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.weights = self._calculate_weights()
        
    def _calculate_weights(self) -> np.ndarray:
        return np.array([self.positions[asset] / self.portfolio_value 
                        for asset in self.returns.columns])
    
    def simulate_returns_normal(self, n_simulations: int = 10000,
                               horizon_days: int = 1) -> np.ndarray:
        """
        Simulate returns assuming multivariate normal
        """
        n_assets = len(self.returns.columns)
        
        # Generate correlated random returns
        simulated_returns = np.random.multivariate_normal(
            mean=self.mean_returns * horizon_days,
            cov=self.cov_matrix * horizon_days,
            size=n_simulations
        )
        
        return simulated_returns
    
    def simulate_returns_t_distribution(self, n_simulations: int = 10000,
                                       horizon_days: int = 1,
                                       df: int = 5) -> np.ndarray:
        """
        Simulate returns using Student's t (fat tails)
        """
        # Fit t-distribution to portfolio returns
        portfolio_returns = self.returns @ self.weights
        
        # Generate from t-distribution
        t_samples = stats.t.rvs(df=df, size=n_simulations)
        
        # Scale by portfolio statistics
        portfolio_mean = portfolio_returns.mean() * horizon_days
        portfolio_vol = portfolio_returns.std() * np.sqrt(horizon_days)
        
        scaled_returns = portfolio_mean + portfolio_vol * t_samples
        
        # Convert to asset-level returns (simplified - maintains correlation structure)
        asset_returns = []
        for asset in self.returns.columns:
            # Correlation with portfolio
            asset_portfolio_corr = self.returns[asset].corr(portfolio_returns)
            
            # Generate correlated returns
            asset_mean = self.mean_returns[asset] * horizon_days
            asset_vol = self.returns[asset].std() * np.sqrt(horizon_days)
            
            # Correlated component + idiosyncratic component
            correlated_part = scaled_returns * asset_portfolio_corr
            idiosyncratic_part = np.random.normal(0, asset_vol * np.sqrt(1 - asset_portfolio_corr**2), 
                                                 n_simulations)
            
            asset_return = asset_mean + correlated_part + idiosyncratic_part
            asset_returns.append(asset_return)
        
        return np.column_stack(asset_returns)
    
    def calculate_var(self, n_simulations: int = 10000,
                     confidence_level: float = 0.99,
                     horizon_days: int = 1,
                     distribution: str = 'normal') -> Dict:
        """
        Calculate VaR using Monte Carlo simulation
        """
        # Generate scenarios
        if distribution == 'normal':
            simulated_returns = self.simulate_returns_normal(n_simulations, horizon_days)
        elif distribution == 't':
            simulated_returns = self.simulate_returns_t_distribution(n_simulations, horizon_days)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        # Calculate portfolio returns for each simulation
        portfolio_returns = simulated_returns @ self.weights
        
        # Convert to dollar P&L
        portfolio_pnl = portfolio_returns * self.portfolio_value
        
        # Calculate VaR
        var_percentile = (1 - confidence_level) * 100
        var_loss = np.percentile(portfolio_pnl, var_percentile)
        
        # Expected Shortfall
        tail_losses = portfolio_pnl[portfolio_pnl <= var_loss]
        expected_shortfall = tail_losses.mean()
        
        # Additional statistics
        mean_pnl = portfolio_pnl.mean()
        median_pnl = np.median(portfolio_pnl)
        worst_pnl = portfolio_pnl.min()
        best_pnl = portfolio_pnl.max()
        
        return {
            'var': abs(var_loss),
            'expected_shortfall': abs(expected_shortfall),
            'mean_pnl': mean_pnl,
            'median_pnl': median_pnl,
            'worst_scenario': abs(worst_pnl),
            'best_scenario': best_pnl,
            'confidence_level': confidence_level,
            'horizon_days': horizon_days,
            'n_simulations': n_simulations,
            'distribution': distribution,
            'portfolio_value': self.portfolio_value,
            'var_percentage': abs(var_loss) / self.portfolio_value
        }
    
    def var_comparison(self, n_simulations: int = 10000,
                      confidence_level: float = 0.99) -> pd.DataFrame:
        """
        Compare VaR under different distributions
        """
        distributions = ['normal', 't']
        results = []
        
        for dist in distributions:
            var_result = self.calculate_var(n_simulations, confidence_level, 
                                          distribution=dist)
            results.append({
                'distribution': dist,
                'var': var_result['var'],
                'expected_shortfall': var_result['expected_shortfall'],
                'var_percentage': var_result['var_percentage'] * 100
            })
        
        return pd.DataFrame(results)
    
    def stress_test(self, scenarios: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Stress test specific scenarios
        
        Args:
            scenarios: Dict of scenario_name -> {asset: return}
        """
        results = []
        
        for scenario_name, asset_returns in scenarios.items():
            # Calculate portfolio return
            portfolio_return = sum(
                asset_returns[asset] * (self.positions[asset] / self.portfolio_value)
                for asset in asset_returns
            )
            
            portfolio_pnl = portfolio_return * self.portfolio_value
            
            results.append({
                'scenario': scenario_name,
                'portfolio_return': portfolio_return,
                'portfolio_pnl': portfolio_pnl,
                'portfolio_pnl_percentage': (portfolio_pnl / self.portfolio_value) * 100
            })
        
        return pd.DataFrame(results).sort_values('portfolio_pnl')

# Example
if __name__ == "__main__":
    # Setup (same as before)
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    returns_data = {
        'AAPL': np.random.normal(0.0005, 0.02, len(dates)),
        'GOOGL': np.random.normal(0.0006, 0.025, len(dates)),
        'MSFT': np.random.normal(0.0005, 0.018, len(dates)),
        'AMZN': np.random.normal(0.0007, 0.03, len(dates)),
        'SPY': np.random.normal(0.0004, 0.012, len(dates))
    }
    returns_df = pd.DataFrame(returns_data, index=dates)
    
    positions = {
        'AAPL': 500000,
        'GOOGL': 300000,
        'MSFT': 400000,
        'AMZN': 200000,
        'SPY': 600000
    }
    
    # Monte Carlo VaR
    mc_var = MonteCarloVaR(returns_df, positions)
    
    # Normal distribution
    normal_results = mc_var.calculate_var(n_simulations=10000, 
                                         confidence_level=0.99,
                                         distribution='normal')
    
    print("Monte Carlo VaR (Normal Distribution)")
    print("="*50)
    print(f"Portfolio Value: ${normal_results['portfolio_value']:, .0f}")
print(f"Simulations: {normal_results['n_simulations']:,}")
print(f"1-day 99% VaR: ${normal_results['var']:,.0f}")
print(f"Expected Shortfall: ${normal_results['expected_shortfall']:,.0f}")
print(f"Worst Scenario: ${normal_results['worst_scenario']:,.0f}")
print()
    
    # Compare distributions
comparison = mc_var.var_comparison(n_simulations = 10000, confidence_level = 0.99)
print("Distribution Comparison:")
print(comparison.to_string(index = False))
print()
    
    # Stress test
stress_scenarios = {
    'Market Crash': {
        'AAPL': -0.20, 'GOOGL': -0.25, 'MSFT': -0.18,
        'AMZN': -0.30, 'SPY': -0.15
    },
    'Tech Selloff': {
        'AAPL': -0.15, 'GOOGL': -0.20, 'MSFT': -0.12,
        'AMZN': -0.25, 'SPY': -0.05
    },
    'Bull Rally': {
        'AAPL': 0.10, 'GOOGL': 0.12, 'MSFT': 0.08,
        'AMZN': 0.15, 'SPY': 0.07
    }
}

stress_results = mc_var.stress_test(stress_scenarios)
print("Stress Test Results:")
print(stress_results.to_string(index = False))
\`\`\`

#### Advantages

✅ **Flexible Distributions**: Can use any distribution  
✅ **Handles Non-linearity**: Works for options  
✅ **Path Dependency**: Can model complex payoffs  
✅ **Scenario Analysis**: Easy to add specific scenarios

#### Disadvantages

❌ **Computationally Intensive**: Slow for large portfolios  
❌ **Model Risk**: Still depends on assumptions  
❌ **Requires Many Simulations**: Need 10,000+ for stability  
❌ **Calibration Challenges**: Parameter estimation critical

## Comparing VaR Methods

### Practical Comparison

\`\`\`python
def compare_var_methods(returns_df, positions, confidence_level=0.99):
    """
    Compare all three VaR methods
    """
    # Historical
    hist_calc = HistoricalVaR(returns_df, positions)
    hist_var = hist_calc.calculate_var(confidence_level)
    
    # Parametric
    param_calc = ParametricVaR(returns_df, positions)
    param_var = param_calc.calculate_var(confidence_level)
    
    # Monte Carlo
    mc_calc = MonteCarloVaR(returns_df, positions)
    mc_var = mc_calc.calculate_var(n_simulations=10000, confidence_level=confidence_level)
    
    comparison = pd.DataFrame({
        'Method': ['Historical', 'Parametric', 'Monte Carlo'],
        'VaR': [hist_var['var'], param_var['var'], mc_var['var']],
        'VaR_Percentage': [
            hist_var['var_percentage'] * 100,
            param_var['var_percentage'] * 100,
            mc_var['var_percentage'] * 100
        ]
    })
    
    return comparison
\`\`\`

### When to Use Each Method

**Historical Simulation**:
- ✅ Simple portfolios (stocks, bonds)
- ✅ Want to capture actual tail events
- ✅ Regulatory reporting (Basel uses historical)
- ❌ Options and derivatives
- ❌ Rapidly changing portfolios

**Parametric VaR**:
- ✅ Large portfolios (fast calculation)
- ✅ Risk attribution needed
- ✅ Daily monitoring
- ❌ Options and non-linear positions
- ❌ Fat-tailed distributions

**Monte Carlo**:
- ✅ Options and complex derivatives
- ✅ Path-dependent payoffs
- ✅ Custom scenarios
- ❌ Need quick results
- ❌ Very large portfolios

## VaR Backtesting

VaR must be validated - do actual losses exceed VaR as often as expected?

### Kupiec Test (POF Test)

Tests if number of VaR exceptions matches expected:

\`\`\`python
def kupiec_test(exceptions, total_days, confidence_level=0.99):
    """
    Kupiec Test (Proportion of Failures)
    
    Tests if exception rate matches confidence level
    """
    expected_exceptions = total_days * (1 - confidence_level)
    
    # Likelihood ratio test statistic
    p = exceptions / total_days
    p_expected = 1 - confidence_level
    
    if p == 0:
        lr_stat = -2 * np.log((1 - p_expected)**total_days)
    elif p == 1:
        lr_stat = -2 * np.log(p_expected**total_days)
    else:
        lr_stat = -2 * (
            exceptions * np.log(p_expected / p) +
            (total_days - exceptions) * np.log((1 - p_expected) / (1 - p))
        )
    
    # Chi-square test with 1 degree of freedom
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
    
    return {
        'exceptions': exceptions,
        'expected': expected_exceptions,
        'exception_rate': p,
        'expected_rate': p_expected,
        'lr_statistic': lr_stat,
        'p_value': p_value,
        'reject_model': p_value < 0.05  # Reject at 5% significance
    }

# Example: 5 exceptions in 250 days at 99% confidence
result = kupiec_test(exceptions=5, total_days=250, confidence_level=0.99)
print(f"Exceptions: {result['exceptions']} (expected: {result['expected']:.1f})")
print(f"P-value: {result['p_value']:.4f}")
print(f"Reject model: {result['reject_model']}")
\`\`\`

### Traffic Light System (Basel)

Basel Committee's traffic light approach:

| Zone | Exceptions (250 days, 99% VaR) | Interpretation | Action |
|------|-------------------------------|----------------|--------|
| Green | 0-4 | Model OK | No action |
| Yellow | 5-9 | Warning | Investigate |
| Red | 10+ | Model failed | Increase capital |

### Continuous Backtesting

\`\`\`python
class VaRBacktester:
    def __init__(self, var_estimates: pd.Series, actual_pnl: pd.Series):
        """
        Backtest VaR estimates vs actual P&L
        
        Args:
            var_estimates: Series of VaR estimates (positive numbers)
            actual_pnl: Series of actual P&L (negative = loss)
        """
        self.var_estimates = var_estimates
        self.actual_pnl = actual_pnl
        
    def identify_exceptions(self):
        """
        Find days where loss exceeded VaR
        """
        # Exception when actual loss > VaR
        exceptions = self.actual_pnl < -self.var_estimates
        return exceptions
    
    def calculate_statistics(self, confidence_level=0.99):
        """
        Calculate backtest statistics
        """
        exceptions = self.identify_exceptions()
        n_exceptions = exceptions.sum()
        n_days = len(exceptions)
        
        # Expected exceptions
        expected = n_days * (1 - confidence_level)
        
        # Kupiec test
        kupiec_result = kupiec_test(n_exceptions, n_days, confidence_level)
        
        # Average VaR vs average loss
        avg_var = self.var_estimates.mean()
        avg_loss = self.actual_pnl[self.actual_pnl < 0].mean()
        
        return {
            'total_days': n_days,
            'exceptions': n_exceptions,
            'expected_exceptions': expected,
            'exception_rate': n_exceptions / n_days,
            'expected_rate': 1 - confidence_level,
            'kupiec_p_value': kupiec_result['p_value'],
            'model_adequate': not kupiec_result['reject_model'],
            'avg_var': avg_var,
            'avg_loss': abs(avg_loss),
            'traffic_light': self._traffic_light_zone(n_exceptions)
        }
    
    def _traffic_light_zone(self, exceptions):
        """Basel traffic light zones"""
        if exceptions <= 4:
            return 'GREEN'
        elif exceptions <= 9:
            return 'YELLOW'
        else:
            return 'RED'
    
    def plot_backtest(self):
        """
        Visualize VaR vs actual losses
        """
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: VaR vs Actual P&L
        ax1.plot(self.actual_pnl.index, self.actual_pnl, label='Actual P&L', alpha=0.7)
        ax1.plot(self.var_estimates.index, -self.var_estimates, 
                label='VaR Limit', color='red', linestyle='--')
        
        # Mark exceptions
        exceptions = self.identify_exceptions()
        exception_dates = self.actual_pnl[exceptions].index
        exception_values = self.actual_pnl[exceptions].values
        ax1.scatter(exception_dates, exception_values, color='red', s=50, 
                   label='VaR Exceptions', zorder=5)
        
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_ylabel('P&L')
        ax1.set_title('VaR Backtest: Actual P&L vs VaR Limit')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative exceptions
        cumulative_exceptions = exceptions.cumsum()
        expected_cumulative = pd.Series(
            range(len(exceptions)), 
            index=exceptions.index
        ) * (1 - 0.99)
        
        ax2.plot(cumulative_exceptions.index, cumulative_exceptions, 
                label='Actual Exceptions')
        ax2.plot(expected_cumulative.index, expected_cumulative, 
                label='Expected Exceptions', linestyle='--')
        ax2.set_ylabel('Cumulative Exceptions')
        ax2.set_title('Cumulative VaR Exceptions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
\`\`\`

## Production VaR System

### Real-World Implementation

\`\`\`python
class ProductionVaRSystem:
    """
    Production-grade VaR calculation system
    """
    def __init__(self, config):
        self.config = config
        self.var_methods = {
            'historical': HistoricalVaR,
            'parametric': ParametricVaR,
            'monte_carlo': MonteCarloVaR
        }
        
    def calculate_comprehensive_var(self, returns_df, positions, 
                                   confidence_levels=[0.95, 0.99],
                                   horizons=[1, 10]):
        """
        Calculate VaR using multiple methods, confidence levels, and horizons
        """
        results = []
        
        for method_name in self.config['methods']:
            calc_class = self.var_methods[method_name]
            calculator = calc_class(returns_df, positions)
            
            for confidence in confidence_levels:
                for horizon in horizons:
                    try:
                        var_result = calculator.calculate_var(
                            confidence_level=confidence,
                            horizon_days=horizon
                        )
                        
                        results.append({
                            'method': method_name,
                            'confidence_level': confidence,
                            'horizon_days': horizon,
                            'var': var_result['var'],
                            'var_percentage': var_result.get('var_percentage', 
                                                            var_result['var'] / sum(positions.values())),
                            'timestamp': pd.Timestamp.now()
                        })
                    except Exception as e:
                        print(f"Error calculating {method_name} VaR: {e}")
        
        return pd.DataFrame(results)
    
    def generate_risk_report(self, returns_df, positions):
        """
        Comprehensive risk report
        """
        # Calculate VaR
        var_results = self.calculate_comprehensive_var(returns_df, positions)
        
        # Focus on 1-day 99% VaR
        primary_var = var_results[
            (var_results['horizon_days'] == 1) & 
            (var_results['confidence_level'] == 0.99)
        ]
        
        report = {
            'timestamp': pd.Timestamp.now(),
            'portfolio_value': sum(positions.values()),
            'num_positions': len(positions),
            'var_estimates': primary_var.to_dict('records'),
            'all_var_calculations': var_results.to_dict('records')
        }
        
        return report
\`\`\`

## Key Takeaways

1. **VaR is NOT**: The maximum possible loss
2. **VaR IS**: A quantile of the loss distribution under normal conditions
3. **Multiple Methods**: Use several methods and compare
4. **Backtest Regularly**: Validate your VaR estimates
5. **Understand Limitations**: VaR says nothing about tail losses
6. **Complement with CVaR**: Expected Shortfall addresses VaR weaknesses
7. **Stress Test**: VaR doesn't cover extreme scenarios

## Common Pitfalls

❌ **Over-reliance on VaR**: It's one tool, not the only tool  
❌ **Ignoring Model Assumptions**: Normal distribution assumption often wrong  
❌ **Not Backtesting**: Blindly trusting VaR without validation  
❌ **Wrong Time Horizon**: 1-day VaR != √10 × 10-day VaR (usually)  
❌ **Ignoring Liquidity**: VaR assumes you can exit positions

## Conclusion

VaR is the industry standard risk metric, but it must be used intelligently:

- Calculate using multiple methods
- Use multiple confidence levels
- Backtest continuously
- Complement with Expected Shortfall
- Stress test beyond VaR
- Understand and communicate limitations

In the next section, we'll explore CVaR (Conditional VaR), which addresses VaR's biggest weakness - telling us nothing about how bad tail losses can be.

Remember: "VaR is like a speedometer - useful but doesn't prevent crashes."
`;

