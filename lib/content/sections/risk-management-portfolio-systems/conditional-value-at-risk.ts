export const conditionalValueAtRisk = `
# Conditional Value at Risk (CVaR)

## Introduction

Value at Risk (VaR) tells us: "We're 99% confident losses won't exceed $X." But what about that remaining 1%? When VaR is breached, how bad can it get?

**This is VaR's fatal flaw**: It provides no information about tail losses.

**Conditional Value at Risk (CVaR)**, also called **Expected Shortfall (ES)** or **Average Value at Risk (AVaR)**, addresses this critical gap. CVaR answers:

**"Given that losses exceed VaR, what is the expected loss?"**

CVaR has become increasingly important in risk management, with regulatory bodies like the Basel Committee considering it superior to VaR for capital requirements.

## Understanding CVaR

### Mathematical Definition

\`\`\`
CVaR_α = E[Loss | Loss > VaR_α]
\`\`\`

In plain English: The average of all losses that exceed the VaR threshold.

### Graphical Intuition

Consider a loss distribution:

\`\`\`
|                    ┌─────┐
|           ┌────────┤     │
|      ┌────┤        │     │
|  ┌───┤    │        │     │      ← CVaR is the average
|──┴───┴────┴────────┴─────┴──────    of this tail region
      │                    ↑
      │                    VaR threshold
      └─────── Tail region ────────→
\`\`\`

**VaR**: The threshold (99th percentile)  
**CVaR**: The average of everything beyond VaR

### Example

Portfolio: $10M  
1-day 99% VaR: $500K

Interpretation:
- 99% of days: Losses ≤ $500K
- 1% of days: Losses > $500K

If CVaR = $750K:
- On those bad 1% of days, average loss is $750K
- Some days might lose $600K, others $1M+
- CVaR tells us the expected value in the tail

## Why CVaR Matters

### VaR's Blind Spot

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

# Two portfolios with SAME VaR but different tail risk

# Portfolio A: Normal distribution
np.random.seed(42)
returns_A = np.random.normal(-0.001, 0.02, 10000)

# Portfolio B: Fat tails (Student's t)
returns_B = 0.015 * np.random.standard_t(df=3, size=10000) - 0.001

# Calculate 99% VaR
var_A = np.percentile(returns_A, 1)
var_B = np.percentile(returns_B, 1)

# Calculate CVaR (average of worst 1%)
cvar_A = returns_A[returns_A <= var_A].mean()
cvar_B = returns_B[returns_B <= var_B].mean()

print(f"Portfolio A:")
print(f"  99% VaR: {abs(var_A)*100:.2f}%")
print(f"  99% CVaR: {abs(cvar_A)*100:.2f}%")
print()
print(f"Portfolio B:")
print(f"  99% VaR: {abs(var_B)*100:.2f}%")
print(f"  99% CVaR: {abs(cvar_B)*100:.2f}%")
print()
print(f"VaR difference: {abs(var_B - var_A)*100:.2f}%")
print(f"CVaR difference: {abs(cvar_B - cvar_A)*100:.2f}%")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(returns_A, bins=100, alpha=0.7, label='Portfolio A')
ax1.axvline(var_A, color='red', linestyle='--', label=f'VaR: {abs(var_A)*100:.2f}%')
ax1.axvline(cvar_A, color='darkred', linestyle='-', label=f'CVaR: {abs(cvar_A)*100:.2f}%')
ax1.set_title('Portfolio A: Normal Distribution')
ax1.legend()

ax2.hist(returns_B, bins=100, alpha=0.7, label='Portfolio B', color='orange')
ax2.axvline(var_B, color='red', linestyle='--', label=f'VaR: {abs(var_B)*100:.2f}%')
ax2.axvline(cvar_B, color='darkred', linestyle='-', label=f'CVaR: {abs(cvar_B)*100:.2f}%')
ax2.set_title('Portfolio B: Fat Tails')
ax2.legend()

plt.tight_layout()
\`\`\`

**Key Insight**: Portfolios can have similar VaR but dramatically different tail risk. CVaR captures this difference!

### Real-World Example: 2008 Financial Crisis

Many banks passed VaR tests right up until the crisis:
- Daily 99% VaR models looked fine
- Then October 2008 happened
- Losses far exceeded VaR for weeks
- CVaR would have warned of extreme tail risk

## Calculating CVaR

### Method 1: Historical Simulation

Most intuitive - average of worst historical scenarios:

\`\`\`python
import numpy as np
import pandas as pd
from typing import Dict, Tuple

class HistoricalCVaR:
    def __init__(self, returns: pd.DataFrame, positions: Dict[str, float]):
        """
        Historical CVaR Calculator
        
        Args:
            returns: DataFrame of asset returns
            positions: Dict of dollar positions
        """
        self.returns = returns
        self.positions = positions
        self.portfolio_value = sum(positions.values())
        self.weights = self._calculate_weights()
        
    def _calculate_weights(self) -> np.ndarray:
        return np.array([self.positions[asset] / self.portfolio_value 
                        for asset in self.returns.columns])
    
    def calculate_cvar(self, confidence_level: float = 0.99, 
                      horizon_days: int = 1) -> Dict:
        """
        Calculate CVaR using historical simulation
        
        Returns:
            Dict with VaR, CVaR, and tail statistics
        """
        # Portfolio returns
        portfolio_returns = (self.returns @ self.weights).values
        
        # Scale to horizon
        horizon_returns = portfolio_returns * np.sqrt(horizon_days)
        
        # Dollar P&L
        portfolio_pnl = horizon_returns * self.portfolio_value
        
        # Calculate VaR
        var_percentile = (1 - confidence_level) * 100
        var_loss = np.percentile(portfolio_pnl, var_percentile)
        
        # Calculate CVaR - average of losses beyond VaR
        tail_losses = portfolio_pnl[portfolio_pnl <= var_loss]
        cvar_loss = tail_losses.mean()
        
        # Tail statistics
        worst_loss = portfolio_pnl.min()
        tail_volatility = tail_losses.std()
        tail_scenarios = len(tail_losses)
        
        return {
            'var': abs(var_loss),
            'cvar': abs(cvar_loss),
            'tail_volatility': tail_volatility,
            'worst_loss': abs(worst_loss),
            'tail_ratio': abs(cvar_loss) / abs(var_loss),  # CVaR/VaR ratio
            'tail_scenarios': tail_scenarios,
            'total_scenarios': len(portfolio_pnl),
            'confidence_level': confidence_level,
            'horizon_days': horizon_days,
            'portfolio_value': self.portfolio_value,
            'var_percentage': abs(var_loss) / self.portfolio_value,
            'cvar_percentage': abs(cvar_loss) / self.portfolio_value
        }
    
    def tail_analysis(self, confidence_level: float = 0.99) -> pd.DataFrame:
        """
        Detailed analysis of tail scenarios
        """
        portfolio_returns = (self.returns @ self.weights).values
        portfolio_pnl = portfolio_returns * self.portfolio_value
        
        # Find VaR threshold
        var_percentile = (1 - confidence_level) * 100
        var_loss = np.percentile(portfolio_pnl, var_percentile)
        
        # Analyze tail scenarios
        tail_mask = portfolio_pnl <= var_loss
        tail_dates = self.returns.index[tail_mask]
        tail_pnl = portfolio_pnl[tail_mask]
        
        # Asset behavior in tail scenarios
        tail_returns = self.returns.loc[tail_mask]
        
        analysis = []
        for asset in self.returns.columns:
            asset_tail_returns = tail_returns[asset]
            
            analysis.append({
                'asset': asset,
                'position': self.positions[asset],
                'avg_tail_return': asset_tail_returns.mean(),
                'worst_tail_return': asset_tail_returns.min(),
                'tail_volatility': asset_tail_returns.std(),
                'contribution_to_tail_loss': (
                    asset_tail_returns.mean() * self.positions[asset]
                )
            })
        
        return pd.DataFrame(analysis).sort_values(
            'contribution_to_tail_loss', ascending=True
        )
    
    def cvar_contribution(self, confidence_level: float = 0.99) -> pd.DataFrame:
        """
        How much each position contributes to CVaR
        """
        portfolio_returns = (self.returns @ self.weights).values
        portfolio_pnl = portfolio_returns * self.portfolio_value
        
        # Find tail scenarios
        var_percentile = (1 - confidence_level) * 100
        var_loss = np.percentile(portfolio_pnl, var_percentile)
        tail_mask = portfolio_pnl <= var_loss
        
        # CVaR contributions
        contributions = []
        for asset in self.returns.columns:
            # Asset returns in tail scenarios
            asset_tail_returns = self.returns.loc[tail_mask, asset]
            
            # Contribution = average return in tail × position
            contribution = asset_tail_returns.mean() * self.positions[asset]
            
            contributions.append({
                'asset': asset,
                'position': self.positions[asset],
                'weight': self.positions[asset] / self.portfolio_value,
                'cvar_contribution': abs(contribution),
                'percent_of_cvar': abs(contribution) / abs(portfolio_pnl[tail_mask].mean())
            })
        
        return pd.DataFrame(contributions).sort_values(
            'cvar_contribution', ascending=False
        )

# Example Usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Create returns with fat tails for some assets
    returns_data = {
        'AAPL': np.random.normal(0.0005, 0.02, len(dates)),
        'GOOGL': 0.025 * np.random.standard_t(df=4, size=len(dates)) + 0.0006,  # Fat tails
        'MSFT': np.random.normal(0.0005, 0.018, len(dates)),
        'AMZN': 0.03 * np.random.standard_t(df=3, size=len(dates)) + 0.0007,  # Fatter tails
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
    
    # Calculate CVaR
    cvar_calc = HistoricalCVaR(returns_df, positions)
    results = cvar_calc.calculate_cvar(confidence_level=0.99)
    
    print("Historical CVaR Analysis")
    print("="*60)
    print(f"Portfolio Value: ${results['portfolio_value']:,.0f}")
print(f"Confidence Level: {results['confidence_level']*100}%")
print()
print(f"99% VaR: ${results['var']:,.0f} ({results['var_percentage']*100:.2f}%)")
print(f"99% CVaR: ${results['cvar']:,.0f} ({results['cvar_percentage']*100:.2f}%)")
print(f"Worst Loss: ${results['worst_loss']:,.0f}")
print()
print(f"CVaR/VaR Ratio: {results['tail_ratio']:.2f}x")
print(f"Tail Scenarios: {results['tail_scenarios']} / {results['total_scenarios']}")
print()
    
    # CVaR contributions
print("CVaR Contribution by Asset:")
print("=" * 60)
contributions = cvar_calc.cvar_contribution(confidence_level = 0.99)
print(contributions.to_string(index = False))
print()
    
    # Tail analysis
print("Tail Scenario Analysis:")
print("=" * 60)
tail_analysis = cvar_calc.tail_analysis(confidence_level = 0.99)
print(tail_analysis.to_string(index = False))
\`\`\`

### Method 2: Parametric CVaR

For normally distributed returns, CVaR has a closed-form solution:

\`\`\`python
from scipy import stats
import numpy as np

class ParametricCVaR:
    def __init__(self, returns: pd.DataFrame, positions: Dict[str, float]):
        self.returns = returns
        self.positions = positions
        self.portfolio_value = sum(positions.values())
        
        # Calculate portfolio statistics
        self.weights = np.array([positions[asset] / self.portfolio_value 
                                for asset in returns.columns])
        self.mean_return = (returns @ self.weights).mean()
        self.volatility = (returns @ self.weights).std()
        
    def calculate_cvar_normal(self, confidence_level: float = 0.99,
                            horizon_days: int = 1,
                            annualization: int = 252) -> Dict:
        """
        Calculate CVaR assuming normal distribution
        
        For normal distribution:
        CVaR = μ - σ × φ(z_α) / (1 - α)
        
        Where:
        - φ = standard normal PDF
        - z_α = normal quantile at confidence level α
        """
        # Annualize statistics
        annual_mean = self.mean_return * annualization
        annual_vol = self.volatility * np.sqrt(annualization)
        
        # Scale to horizon
        horizon_mean = annual_mean * (horizon_days / annualization)
        horizon_vol = annual_vol * np.sqrt(horizon_days / annualization)
        
        # Calculate VaR
        z_alpha = stats.norm.ppf(1 - confidence_level)
        var_percentage = -(horizon_mean + z_alpha * horizon_vol)
        
        # Calculate CVaR for normal distribution
        # CVaR = E[L | L > VaR] = μ - σ × φ(z_α) / (1 - α)
        phi_z = stats.norm.pdf(z_alpha)
        cvar_percentage = -(horizon_mean - horizon_vol * phi_z / (1 - confidence_level))
        
        # Convert to dollars
        var_dollar = var_percentage * self.portfolio_value
        cvar_dollar = cvar_percentage * self.portfolio_value
        
        return {
            'var': abs(var_dollar),
            'cvar': abs(cvar_dollar),
            'var_percentage': abs(var_percentage),
            'cvar_percentage': abs(cvar_percentage),
            'tail_ratio': abs(cvar_percentage) / abs(var_percentage),
            'mean_return': horizon_mean,
            'volatility': horizon_vol,
            'z_score': z_alpha,
            'confidence_level': confidence_level,
            'horizon_days': horizon_days,
            'portfolio_value': self.portfolio_value
        }
    
    def calculate_cvar_t_distribution(self, confidence_level: float = 0.99,
                                     horizon_days: int = 1,
                                     df: int = 5) -> Dict:
        """
        Calculate CVaR assuming Student's t distribution (fat tails)
        
        More realistic for financial returns
        """
        # Scale statistics to horizon
        horizon_mean = self.mean_return * horizon_days
        horizon_vol = self.volatility * np.sqrt(horizon_days)
        
        # t-distribution quantile
        t_alpha = stats.t.ppf(1 - confidence_level, df=df)
        
        # VaR for t-distribution
        var_percentage = -(horizon_mean + t_alpha * horizon_vol * np.sqrt((df - 2) / df))
        
        # CVaR for t-distribution
        # CVaR = μ - σ × (pdf(t_α) × (df + t_α²) / ((1-α) × (df-1)))
        t_pdf = stats.t.pdf(t_alpha, df=df)
        cvar_percentage = -(horizon_mean - horizon_vol * np.sqrt((df - 2) / df) * 
                           (t_pdf * (df + t_alpha**2) / ((1 - confidence_level) * (df - 1))))
        
        var_dollar = var_percentage * self.portfolio_value
        cvar_dollar = cvar_percentage * self.portfolio_value
        
        return {
            'var': abs(var_dollar),
            'cvar': abs(cvar_dollar),
            'var_percentage': abs(var_percentage),
            'cvar_percentage': abs(cvar_percentage),
            'tail_ratio': abs(cvar_percentage) / abs(var_percentage),
            'distribution': 't-distribution',
            'degrees_of_freedom': df,
            'confidence_level': confidence_level,
            'horizon_days': horizon_days,
            'portfolio_value': self.portfolio_value
        }

# Example
if __name__ == "__main__":
    # Using same data as before
    param_cvar = ParametricCVaR(returns_df, positions)
    
    # Normal distribution CVaR
    normal_results = param_cvar.calculate_cvar_normal(confidence_level=0.99)
    print("Parametric CVaR (Normal Distribution)")
    print("="*60)
    print(f"99% VaR: ${normal_results['var']:, .0f}")
print(f"99% CVaR: ${normal_results['cvar']:,.0f}")
print(f"CVaR/VaR Ratio: {normal_results['tail_ratio']:.2f}x")
print()
    
    # t - distribution CVaR(more realistic)
t_results = param_cvar.calculate_cvar_t_distribution(confidence_level = 0.99, df = 5)
print("Parametric CVaR (t-Distribution, df=5)")
print("=" * 60)
print(f"99% VaR: ${t_results['var']:,.0f}")
print(f"99% CVaR: ${t_results['cvar']:,.0f}")
print(f"CVaR/VaR Ratio: {t_results['tail_ratio']:.2f}x")
print()

print(f"Difference from Normal:")
print(f"  VaR: ${abs(t_results['var'] - normal_results['var']):,.0f}")
print(f"  CVaR: ${abs(t_results['cvar'] - normal_results['cvar']):,.0f}")
\`\`\`

### Method 3: Monte Carlo CVaR

Most flexible - can handle any distribution:

\`\`\`python
class MonteCarloCVaR:
    def __init__(self, returns: pd.DataFrame, positions: Dict[str, float]):
        self.returns = returns
        self.positions = positions
        self.portfolio_value = sum(positions.values())
        self.weights = np.array([positions[asset] / self.portfolio_value 
                                for asset in returns.columns])
        
    def simulate_scenarios(self, n_simulations: int = 10000,
                          horizon_days: int = 1,
                          distribution: str = 't',
                          df: int = 5) -> np.ndarray:
        """
        Simulate portfolio return scenarios
        """
        # Portfolio historical statistics
        portfolio_returns = self.returns @ self.weights
        mean_return = portfolio_returns.mean() * horizon_days
        volatility = portfolio_returns.std() * np.sqrt(horizon_days)
        
        if distribution == 'normal':
            # Normal distribution
            simulated_returns = np.random.normal(mean_return, volatility, n_simulations)
        elif distribution == 't':
            # Student's t for fat tails
            t_samples = stats.t.rvs(df=df, size=n_simulations)
            simulated_returns = mean_return + volatility * t_samples
        elif distribution == 'historical':
            # Bootstrap from historical returns
            simulated_returns = np.random.choice(
                portfolio_returns * np.sqrt(horizon_days), 
                size=n_simulations, 
                replace=True
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        return simulated_returns
    
    def calculate_cvar(self, n_simulations: int = 10000,
                      confidence_level: float = 0.99,
                      horizon_days: int = 1,
                      distribution: str = 't',
                      df: int = 5) -> Dict:
        """
        Calculate CVaR using Monte Carlo simulation
        """
        # Generate scenarios
        simulated_returns = self.simulate_scenarios(
            n_simulations, horizon_days, distribution, df
        )
        
        # Convert to dollar P&L
        simulated_pnl = simulated_returns * self.portfolio_value
        
        # Calculate VaR
        var_percentile = (1 - confidence_level) * 100
        var_loss = np.percentile(simulated_pnl, var_percentile)
        
        # Calculate CVaR
        tail_losses = simulated_pnl[simulated_pnl <= var_loss]
        cvar_loss = tail_losses.mean()
        
        # Tail statistics
        worst_loss = simulated_pnl.min()
        tail_volatility = tail_losses.std()
        
        # Percentiles of tail
        tail_percentiles = np.percentile(tail_losses, [10, 25, 50, 75, 90])
        
        return {
            'var': abs(var_loss),
            'cvar': abs(cvar_loss),
            'worst_simulated': abs(worst_loss),
            'tail_volatility': tail_volatility,
            'tail_ratio': abs(cvar_loss) / abs(var_loss),
            'tail_10th_percentile': abs(tail_percentiles[0]),
            'tail_25th_percentile': abs(tail_percentiles[1]),
            'tail_median': abs(tail_percentiles[2]),
            'tail_75th_percentile': abs(tail_percentiles[3]),
            'tail_90th_percentile': abs(tail_percentiles[4]),
            'n_simulations': n_simulations,
            'n_tail_scenarios': len(tail_losses),
            'confidence_level': confidence_level,
            'horizon_days': horizon_days,
            'distribution': distribution,
            'portfolio_value': self.portfolio_value
        }
    
    def tail_distribution_analysis(self, n_simulations: int = 10000,
                                   confidence_level: float = 0.99) -> Dict:
        """
        Analyze the shape of the tail distribution
        """
        simulated_returns = self.simulate_scenarios(n_simulations)
        simulated_pnl = simulated_returns * self.portfolio_value
        
        # Find tail
        var_percentile = (1 - confidence_level) * 100
        var_loss = np.percentile(simulated_pnl, var_percentile)
        tail_losses = simulated_pnl[simulated_pnl <= var_loss]
        
        # Tail statistics
        from scipy.stats import skew, kurtosis
        
        return {
            'mean': tail_losses.mean(),
            'median': np.median(tail_losses),
            'std': tail_losses.std(),
            'skewness': skew(tail_losses),
            'kurtosis': kurtosis(tail_losses),
            'min': tail_losses.min(),
            'max': tail_losses.max()
        }

# Example
if __name__ == "__main__":
    mc_cvar = MonteCarloCVaR(returns_df, positions)
    
    # Compare distributions
    distributions = ['normal', 't', 'historical']
    results = []
    
    for dist in distributions:
        result = mc_cvar.calculate_cvar(
            n_simulations=10000,
            confidence_level=0.99,
            distribution=dist
        )
        results.append({
            'distribution': dist,
            'var': result['var'],
            'cvar': result['cvar'],
            'tail_ratio': result['tail_ratio'],
            'worst': result['worst_simulated']
        })
    
    comparison = pd.DataFrame(results)
    print("Monte Carlo CVaR - Distribution Comparison")
    print("="*60)
    print(comparison.to_string(index=False))
\`\`\`

## CVaR Optimization

CVaR can be used as an objective function for portfolio optimization:

### Minimize CVaR Portfolio

\`\`\`python
from scipy.optimize import minimize

def optimize_cvar_portfolio(returns_df: pd.DataFrame, 
                           confidence_level: float = 0.95,
                           target_return: float = None) -> Dict:
    """
    Find portfolio weights that minimize CVaR
    
    This is a LINEAR PROGRAMMING problem (unlike VaR)
    """
    n_assets = len(returns_df.columns)
    n_scenarios = len(returns_df)
    
    # Convert returns to numpy array
    returns_matrix = returns_df.values
    
    def portfolio_cvar(weights):
        """Calculate CVaR for given weights"""
        portfolio_returns = returns_matrix @ weights
        var_threshold = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        tail_losses = portfolio_returns[portfolio_returns <= var_threshold]
        return -tail_losses.mean()  # Negative because we want losses
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
    ]
    
    # If target return specified, add constraint
    if target_return is not None:
        mean_returns = returns_df.mean().values
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.dot(w, mean_returns) - target_return
        })
    
    # Bounds (no short selling)
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial guess - equal weights
    w0 = np.ones(n_assets) / n_assets
    
    # Optimize
    result = minimize(
        portfolio_cvar,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    optimal_weights = result.x
    
    # Calculate final metrics
    optimal_returns = returns_matrix @ optimal_weights
    optimal_var = np.percentile(optimal_returns, (1 - confidence_level) * 100)
    optimal_cvar = optimal_returns[optimal_returns <= optimal_var].mean()
    
    return {
        'weights': dict(zip(returns_df.columns, optimal_weights)),
        'var': abs(optimal_var),
        'cvar': abs(optimal_cvar),
        'expected_return': optimal_returns.mean(),
        'volatility': optimal_returns.std(),
        'success': result.success
    }

# Example
if __name__ == "__main__":
    optimal = optimize_cvar_portfolio(returns_df, confidence_level=0.95)
    
    print("CVaR-Optimal Portfolio")
    print("="*60)
    print("Optimal Weights:")
    for asset, weight in optimal['weights'].items():
        print(f"  {asset}: {weight*100:.1f}%")
    print()
    print(f"Expected Return: {optimal['expected_return']*252*100:.1f}% annual")
    print(f"Volatility: {optimal['volatility']*np.sqrt(252)*100:.1f}% annual")
    print(f"95% VaR: {optimal['var']*100:.2f}%")
    print(f"95% CVaR: {optimal['cvar']*100:.2f}%")
\`\`\`

## CVaR vs VaR: Key Differences

| Aspect | VaR | CVaR |
|--------|-----|------|
| **Question Answered** | Maximum loss at confidence level | Average loss beyond VaR |
| **Tail Risk** | Ignores | Captures |
| **Mathematical Properties** | Not sub-additive | Sub-additive (coherent) |
| **Optimization** | Non-convex (hard) | Convex (easier) |
| **Interpretation** | Simpler | Slightly more complex |
| **Regulatory** | Basel II | Basel Committee considering |
| **Portfolio Selection** | Can give bad portfolios | Better portfolio selection |

### Sub-Additivity (Coherent Risk Measure)

VaR can INCREASE when you diversify (violates sub-additivity).  
CVaR is sub-additive: CVaR(A + B) ≤ CVaR(A) + CVaR(B)

This means diversification always reduces CVaR (as it should).

## Production CVaR System

\`\`\`python
class ProductionCVaRSystem:
    """
    Enterprise-grade CVaR calculation and monitoring
    """
    def __init__(self, returns_df, positions, config):
        self.returns_df = returns_df
        self.positions = positions
        self.config = config
        self.portfolio_value = sum(positions.values())
        
    def calculate_comprehensive_cvar(self):
        """
        Calculate CVaR using all methods
        """
        results = {}
        
        # Historical CVaR
        hist_calc = HistoricalCVaR(self.returns_df, self.positions)
        results['historical'] = hist_calc.calculate_cvar(
            confidence_level=self.config['confidence_level']
        )
        
        # Parametric CVaR (normal)
        param_calc = ParametricCVaR(self.returns_df, self.positions)
        results['parametric_normal'] = param_calc.calculate_cvar_normal(
            confidence_level=self.config['confidence_level']
        )
        
        # Parametric CVaR (t-distribution)
        results['parametric_t'] = param_calc.calculate_cvar_t_distribution(
            confidence_level=self.config['confidence_level'],
            df=5
        )
        
        # Monte Carlo CVaR
        mc_calc = MonteCarloCVaR(self.returns_df, self.positions)
        results['monte_carlo'] = mc_calc.calculate_cvar(
            n_simulations=self.config['n_simulations'],
            confidence_level=self.config['confidence_level']
        )
        
        return results
    
    def generate_cvar_report(self):
        """
        Comprehensive CVaR risk report
        """
        results = self.calculate_comprehensive_cvar()
        
        # Summary statistics
        cvar_estimates = [r['cvar'] for r in results.values()]
        var_estimates = [r['var'] for r in results.values()]
        
        report = {
            'timestamp': pd.Timestamp.now(),
            'portfolio_value': self.portfolio_value,
            'confidence_level': self.config['confidence_level'],
            'cvar_estimates': results,
            'cvar_range': {
                'min': min(cvar_estimates),
                'max': max(cvar_estimates),
                'mean': np.mean(cvar_estimates),
                'median': np.median(cvar_estimates)
            },
            'var_range': {
                'min': min(var_estimates),
                'max': max(var_estimates),
                'mean': np.mean(var_estimates)
            },
            'recommended_capital': max(cvar_estimates) * self.config['capital_multiplier']
        }
        
        return report
    
    def monitor_cvar_breaches(self, actual_loss: float):
        """
        Track when actual losses exceed CVaR estimates
        """
        results = self.calculate_comprehensive_cvar()
        
        breaches = []
        for method, result in results.items():
            if actual_loss > result['cvar']:
                breaches.append({
                    'method': method,
                    'cvar_estimate': result['cvar'],
                    'actual_loss': actual_loss,
                    'excess': actual_loss - result['cvar'],
                    'excess_percentage': (actual_loss - result['cvar']) / result['cvar']
                })
        
        return breaches
\`\`\`

## Regulatory Perspective

### Basel Committee View

The Basel Committee has expressed concerns about VaR:

1. **VaR Doesn't Capture Tail Risk**: By definition
2. **Procyclicality**: VaR drops when markets are calm (right before crashes)
3. **Model Gaming**: Easy to manipulate

**Shift Toward CVaR/Expected Shortfall**:
- Basel Committee considering ES for market risk capital
- Already used in some jurisdictions
- More conservative than VaR

### Capital Requirements

\`\`\`python
def calculate_regulatory_capital(cvar: float, 
                                multiplier: float = 3.0,
                                buffer: float = 0.25) -> Dict:
    """
    Regulatory capital calculation using CVaR
    
    Args:
        cvar: Expected Shortfall estimate
        multiplier: Regulatory multiplier (typically 3-4)
        buffer: Additional buffer (25%)
    """
    base_capital = cvar * multiplier
    buffer_capital = base_capital * buffer
    total_capital = base_capital + buffer_capital
    
    return {
        'base_capital': base_capital,
        'buffer_capital': buffer_capital,
        'total_required_capital': total_capital,
        'cvar': cvar,
        'multiplier': multiplier,
        'buffer_percentage': buffer
    }
\`\`\`

## Key Takeaways

1. **CVaR > VaR**: Always use CVaR alongside VaR
2. **Captures Tail Risk**: CVaR tells you about extreme losses
3. **Coherent Risk Measure**: CVaR has better mathematical properties
4. **Portfolio Optimization**: CVaR optimization is more tractable than VaR
5. **Regulatory Trend**: Moving toward CVaR/Expected Shortfall
6. **Multiple Methods**: Calculate CVaR using historical, parametric, and Monte Carlo
7. **Fat Tails Matter**: Use t-distribution or historical simulation for realistic tail risk

## Common Pitfalls

❌ **Using Only VaR**: VaR alone is insufficient  
❌ **Ignoring Distribution**: Normal assumption understates CVaR  
❌ **Not Backtesting**: Validate CVaR estimates  
❌ **Confusing CVaR and VaR**: They measure different things  
❌ **Static CVaR**: Tail risk changes over time

## Conclusion

CVaR addresses VaR's fundamental flaw - it tells us what happens in the tail. While VaR says "losses won't exceed $X with 99% confidence," CVaR says "when losses do exceed $X (that 1% of the time), expect to lose $Y on average."

In risk management, understanding tail losses is critical. Black swan events (2008 crisis, COVID-19 crash) occur in the tail. CVaR helps quantify and prepare for these events.

**Best Practice**: Always report both VaR and CVaR. VaR for day-to-day risk monitoring, CVaR for capital allocation and stress testing.

Next, we'll explore stress testing and scenario analysis - complementary approaches to understanding extreme risks beyond what VaR and CVaR capture from historical data.
`;

