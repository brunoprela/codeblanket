export const riskBudgeting = {
  id: 'risk-budgeting',
  title: 'Risk Budgeting',
  content: `
# Risk Budgeting

## Introduction

"Don't allocate capital - allocate risk."

Traditional portfolio construction allocates **capital** (dollars). Risk budgeting allocates **risk** - a fundamentally different approach that recognizes not all dollars contribute equally to risk.

Consider:
- $100K in cash: Near-zero risk
- $100K in Tesla stock: High risk
- $100K in short-dated Treasuries: Low risk

Risk budgeting answers: **"How much risk should each position/strategy consume?"**

This approach, pioneered by firms like Bridgewater Associates (All Weather Portfolio), has become standard in institutional portfolio management.

## Core Concepts

### Risk Budget vs Capital Budget

**Capital Budget**: Allocate $100M
- 60% stocks ($60M)
- 40% bonds ($40M)

**Risk Budget**: Allocate risk budget
- 50% from stocks
- 30% from bonds  
- 20% from alternatives

With risk budgeting, you might end up with:
- 30% stocks (high vol → small allocation for target risk)
- 60% bonds (low vol → large allocation for target risk)
- 10% alternatives

\`\`\`python
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.optimize import minimize
from dataclasses import dataclass

class RiskBudgetingSystem:
    """
    Implement risk budgeting for portfolio construction
    """
    def __init__(self, 
                 asset_returns: pd.DataFrame,
                 risk_budgets: Dict[str, float]):
        """
        Args:
            asset_returns: Historical returns by asset
            risk_budgets: Dict of {asset: target_risk_contribution}
                         Risk contributions should sum to 1.0
        """
        self.returns = asset_returns
        self.assets = asset_returns.columns.tolist()
        self.risk_budgets = risk_budgets
        
        # Validate risk budgets sum to 1
        total_budget = sum(risk_budgets.values())
        assert abs(total_budget - 1.0) < 0.01, f"Risk budgets must sum to 1.0, got {total_budget}"
        
        # Calculate covariance matrix
        self.cov_matrix = asset_returns.cov().values
        
    def calculate_risk_contributions(self, weights: np.ndarray) -> np.ndarray:
        """
        Calculate each asset's contribution to total portfolio risk
        
        RC_i = w_i × (Σ × w)_i / σ_portfolio
        """
        # Portfolio variance
        portfolio_var = weights @ self.cov_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_var)
        
        # Marginal contributions
        marginal_contrib = self.cov_matrix @ weights
        
        # Risk contributions
        risk_contributions = (weights * marginal_contrib) / portfolio_vol
        
        return risk_contributions
    
    def optimize_risk_parity(self) -> Dict:
        """
        Risk Parity: Equal risk contribution from all assets
        
        Minimize: Σ (RC_i - 1/N)²
        """
        n_assets = len(self.assets)
        
        def objective(weights):
            """Minimize deviation from equal risk contribution"""
            risk_contribs = self.calculate_risk_contributions(weights)
            target = 1.0 / n_assets
            return np.sum((risk_contribs - target) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]
        
        # Bounds (long only)
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))
        
        # Initial guess - equal weights
        w0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        optimal_weights = result.x
        risk_contribs = self.calculate_risk_contributions(optimal_weights)
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(optimal_weights, self.returns.mean()) * 252
        portfolio_vol = np.sqrt(optimal_weights @ self.cov_matrix @ optimal_weights) * np.sqrt(252)
        
        return {
            'weights': dict(zip(self.assets, optimal_weights)),
            'risk_contributions': dict(zip(self.assets, risk_contribs)),
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_vol,
            'sharpe_ratio': portfolio_return / portfolio_vol if portfolio_vol > 0 else 0,
            'success': result.success
        }
    
    def optimize_custom_risk_budget(self) -> Dict:
        """
        Custom Risk Budget: Target specific risk allocations
        
        Minimize: Σ (RC_i - Target_i)²
        """
        n_assets = len(self.assets)
        
        # Target risk contributions
        targets = np.array([self.risk_budgets.get(asset, 1.0/n_assets) 
                           for asset in self.assets])
        
        def objective(weights):
            """Minimize deviation from target risk contributions"""
            risk_contribs = self.calculate_risk_contributions(weights)
            return np.sum((risk_contribs - targets) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        # Bounds
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))
        
        # Initial guess
        w0 = targets.copy()  # Start with risk budgets as weights
        w0 = w0 / np.sum(w0)  # Normalize
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        optimal_weights = result.x
        risk_contribs = self.calculate_risk_contributions(optimal_weights)
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(optimal_weights, self.returns.mean()) * 252
        portfolio_vol = np.sqrt(optimal_weights @ self.cov_matrix @ optimal_weights) * np.sqrt(252)
        
        # Compare actual vs target risk contributions
        comparison = []
        for i, asset in enumerate(self.assets):
            comparison.append({
                'asset': asset,
                'target_risk_contrib': targets[i],
                'actual_risk_contrib': risk_contribs[i],
                'weight': optimal_weights[i],
                'deviation': abs(risk_contribs[i] - targets[i])
            })
        
        return {
            'weights': dict(zip(self.assets, optimal_weights)),
            'risk_contributions': dict(zip(self.assets, risk_contribs)),
            'comparison': pd.DataFrame(comparison),
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_vol,
            'sharpe_ratio': portfolio_return / portfolio_vol if portfolio_vol > 0 else 0,
            'success': result.success,
            'total_deviation': np.sum([c['deviation'] for c in comparison])
        }
    
    def hierarchical_risk_budgeting(self, 
                                    asset_groups: Dict[str, List[str]],
                                    group_risk_budgets: Dict[str, float]) -> Dict:
        """
        Two-level risk budgeting
        
        Level 1: Allocate risk across asset classes
        Level 2: Allocate risk within asset classes
        
        Args:
            asset_groups: Dict of {group_name: [asset_list]}
            group_risk_budgets: Risk budget for each group
        """
        # Step 1: Allocate across groups
        group_weights = {}
        group_volatilities = {}
        
        for group_name, assets in asset_groups.items():
            # Get returns for this group
            group_returns = self.returns[assets]
            group_cov = group_returns.cov().values
            
            # Equal weight within group (or could optimize)
            n_assets_in_group = len(assets)
            group_w = np.ones(n_assets_in_group) / n_assets_in_group
            
            # Group volatility
            group_vol = np.sqrt(group_w @ group_cov @ group_w)
            group_volatilities[group_name] = group_vol
        
        # Allocate capital to groups based on risk budget and volatility
        for group_name in asset_groups.keys():
            risk_budget = group_risk_budgets.get(group_name, 0.0)
            vol = group_volatilities[group_name]
            
            # Weight inversely proportional to volatility, scaled by risk budget
            group_weights[group_name] = risk_budget / vol if vol > 0 else 0
        
        # Normalize group weights
        total_weight = sum(group_weights.values())
        group_weights = {k: v/total_weight for k, v in group_weights.items()}
        
        # Step 2: Allocate within groups (equal risk contribution within group)
        final_weights = {}
        
        for group_name, assets in asset_groups.items():
            group_weight = group_weights[group_name]
            
            # Risk parity within group
            group_returns = self.returns[assets]
            group_rb = RiskBudgetingSystem(
                group_returns,
                {asset: 1.0/len(assets) for asset in assets}
            )
            
            group_result = group_rb.optimize_risk_parity()
            
            # Scale by group weight
            for asset in assets:
                final_weights[asset] = group_result['weights'][asset] * group_weight
        
        # Calculate final portfolio metrics
        final_weights_array = np.array([final_weights.get(asset, 0) for asset in self.assets])
        portfolio_return = np.dot(final_weights_array, self.returns.mean()) * 252
        portfolio_vol = np.sqrt(final_weights_array @ self.cov_matrix @ final_weights_array) * np.sqrt(252)
        
        return {
            'group_weights': group_weights,
            'group_risk_budgets': group_risk_budgets,
            'final_weights': final_weights,
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_vol,
            'sharpe_ratio': portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        }

# Example Usage
if __name__ == "__main__":
    # Sample data
    np.random.seed(42)
    n_days = 252 * 3
    
    # Generate returns with different volatilities
    assets_data = {
        'Stocks': np.random.normal(0.0008, 0.015, n_days),  # High vol
        'Bonds': np.random.normal(0.0003, 0.005, n_days),   # Low vol
        'Commodities': np.random.normal(0.0005, 0.020, n_days),  # Highest vol
        'REITs': np.random.normal(0.0006, 0.012, n_days)    # Medium vol
    }
    
    returns_df = pd.DataFrame(assets_data)
    
    # Risk budgets (equal risk contribution)
    risk_budgets = {
        'Stocks': 0.25,
        'Bonds': 0.25,
        'Commodities': 0.25,
        'REITs': 0.25
    }
    
    rb_system = RiskBudgetingSystem(returns_df, risk_budgets)
    
    print("Risk Budgeting Analysis")
    print("="*70)
    print()
    
    # Risk Parity (equal risk contribution)
    print("1. RISK PARITY (Equal Risk Contribution)")
    print("-"*70)
    rp_result = rb_system.optimize_risk_parity()
    
    print("Optimal Weights:")
    for asset, weight in rp_result['weights'].items():
        print(f"  {asset}: {weight*100:.1f}%")
    print()
    
    print("Risk Contributions:")
    total_rc = sum(rp_result['risk_contributions'].values())
    for asset, rc in rp_result['risk_contributions'].items():
        pct = (rc / total_rc) * 100
        print(f"  {asset}: {pct:.1f}%")
    print()
    
    print(f"Portfolio Return: {rp_result['portfolio_return']*100:.2f}%")
    print(f"Portfolio Volatility: {rp_result['portfolio_volatility']*100:.2f}%")
    print(f"Sharpe Ratio: {rp_result['sharpe_ratio']:.2f}")
    print()
    print()
    
    # Custom Risk Budget
    print("2. CUSTOM RISK BUDGET")
    print("-"*70)
    custom_budgets = {
        'Stocks': 0.40,      # 40% of risk from stocks
        'Bonds': 0.20,       # 20% from bonds
        'Commodities': 0.20, # 20% from commodities
        'REITs': 0.20        # 20% from REITs
    }
    
    rb_system_custom = RiskBudgetingSystem(returns_df, custom_budgets)
    custom_result = rb_system_custom.optimize_custom_risk_budget()
    
    print("Target vs Actual Risk Contributions:")
    print(custom_result['comparison'].to_string(index=False))
    print()
    
    print(f"Total Deviation: {custom_result['total_deviation']:.4f}")
    print(f"Portfolio Return: {custom_result['portfolio_return']*100:.2f}%")
    print(f"Portfolio Volatility: {custom_result['portfolio_volatility']*100:.2f}%")
    print(f"Sharpe Ratio: {custom_result['sharpe_ratio']:.2f}")
\`\`\`

## Comparing Risk Budgeting Approaches

\`\`\`python
def compare_allocation_methods(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare different allocation approaches
    """
    results = []
    
    # 1. Equal Weight (traditional)
    n_assets = len(returns_df.columns)
    equal_weights = np.ones(n_assets) / n_assets
    cov_matrix = returns_df.cov().values
    
    eq_return = np.dot(equal_weights, returns_df.mean()) * 252
    eq_vol = np.sqrt(equal_weights @ cov_matrix @ equal_weights) * np.sqrt(252)
    
    results.append({
        'method': 'Equal Weight',
        'annual_return': eq_return,
        'annual_volatility': eq_vol,
        'sharpe_ratio': eq_return / eq_vol
    })
    
    # 2. Minimum Variance
    def min_var_objective(w):
        return w @ cov_matrix @ w
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    
    mv_result = minimize(min_var_objective, equal_weights, method='SLSQP',
                        bounds=bounds, constraints=constraints)
    mv_weights = mv_result.x
    
    mv_return = np.dot(mv_weights, returns_df.mean()) * 252
    mv_vol = np.sqrt(mv_weights @ cov_matrix @ mv_weights) * np.sqrt(252)
    
    results.append({
        'method': 'Minimum Variance',
        'annual_return': mv_return,
        'annual_volatility': mv_vol,
        'sharpe_ratio': mv_return / mv_vol
    })
    
    # 3. Risk Parity
    risk_budgets = {col: 1.0/n_assets for col in returns_df.columns}
    rb_system = RiskBudgetingSystem(returns_df, risk_budgets)
    rp_result = rb_system.optimize_risk_parity()
    
    results.append({
        'method': 'Risk Parity',
        'annual_return': rp_result['portfolio_return'],
        'annual_volatility': rp_result['portfolio_volatility'],
        'sharpe_ratio': rp_result['sharpe_ratio']
    })
    
    # 4. Maximum Sharpe (mean-variance optimization)
    mean_returns = returns_df.mean() * 252
    
    def neg_sharpe(w):
        ret = np.dot(w, mean_returns)
        vol = np.sqrt(w @ cov_matrix @ w) * np.sqrt(252)
        return -ret / vol if vol > 0 else 0
    
    ms_result = minimize(neg_sharpe, equal_weights, method='SLSQP',
                        bounds=bounds, constraints=constraints)
    ms_weights = ms_result.x
    
    ms_return = np.dot(ms_weights, mean_returns)
    ms_vol = np.sqrt(ms_weights @ cov_matrix @ ms_weights) * np.sqrt(252)
    
    results.append({
        'method': 'Maximum Sharpe',
        'annual_return': ms_return,
        'annual_volatility': ms_vol,
        'sharpe_ratio': ms_return / ms_vol
    })
    
    df = pd.DataFrame(results)
    df['annual_return'] = df['annual_return'] * 100
    df['annual_volatility'] = df['annual_volatility'] * 100
    
    return df

# Compare approaches
comparison = compare_allocation_methods(returns_df)
print("Allocation Method Comparison")
print("="*70)
print(comparison.to_string(index=False))
\`\`\`

## Risk Budgeting for Multi-Strategy Funds

\`\`\`python
class MultiStrategyRiskBudget:
    """
    Risk budgeting across multiple strategies
    """
    def __init__(self, strategy_returns: Dict[str, pd.Series]):
        """
        Args:
            strategy_returns: Dict of {strategy_name: return_series}
        """
        self.strategies = strategy_returns
        self.strategy_names = list(strategy_returns.keys())
        
        # Combine into DataFrame
        self.returns_df = pd.DataFrame(strategy_returns)
        self.cov_matrix = self.returns_df.cov().values
        
    def allocate_by_risk_budget(self, 
                                risk_budgets: Dict[str, float],
                                total_capital: float = 100000000) -> Dict:
        """
        Allocate capital across strategies based on risk budgets
        
        Args:
            risk_budgets: Target risk contribution per strategy (sum to 1.0)
            total_capital: Total capital to allocate
        """
        rb_system = RiskBudgetingSystem(self.returns_df, risk_budgets)
        result = rb_system.optimize_custom_risk_budget()
        
        # Convert weights to dollar allocations
        capital_allocations = {
            strategy: result['weights'][strategy] * total_capital
            for strategy in self.strategy_names
        }
        
        # Calculate strategy-level metrics
        strategy_metrics = []
        for strategy in self.strategy_names:
            returns = self.strategies[strategy]
            
            strategy_metrics.append({
                'strategy': strategy,
                'capital_allocation': capital_allocations[strategy],
                'capital_pct': result['weights'][strategy] * 100,
                'risk_budget': risk_budgets[strategy] * 100,
                'actual_risk_contrib': result['risk_contributions'][strategy] * 100,
                'expected_return': returns.mean() * 252 * 100,
                'volatility': returns.std() * np.sqrt(252) * 100,
                'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252)
            })
        
        return {
            'capital_allocations': capital_allocations,
            'strategy_metrics': pd.DataFrame(strategy_metrics),
            'portfolio_return': result['portfolio_return'],
            'portfolio_volatility': result['portfolio_volatility'],
            'portfolio_sharpe': result['sharpe_ratio']
        }
    
    def dynamic_risk_budgeting(self, 
                              current_risk_budgets: Dict[str, float],
                              target_portfolio_vol: float = 0.10) -> Dict:
        """
        Dynamically adjust risk budgets to hit target volatility
        
        Uses volatility scaling
        """
        # Calculate current portfolio
        rb_system = RiskBudgetingSystem(self.returns_df, current_risk_budgets)
        result = rb_system.optimize_custom_risk_budget()
        
        current_vol = result['portfolio_volatility']
        
        # Scaling factor to hit target
        scale_factor = target_portfolio_vol / current_vol if current_vol > 0 else 1.0
        
        # Apply leverage/de-leverage
        scaled_weights = {
            strategy: result['weights'][strategy] * scale_factor
            for strategy in self.strategy_names
        }
        
        # Adjust for leverage constraints (e.g., max 1.5x gross)
        total_gross = sum(abs(w) for w in scaled_weights.values())
        if total_gross > 1.5:
            # Scale down to max leverage
            scaled_weights = {k: v * 1.5 / total_gross for k, v in scaled_weights.items()}
        
        return {
            'scaled_weights': scaled_weights,
            'current_volatility': current_vol,
            'target_volatility': target_portfolio_vol,
            'scale_factor': scale_factor,
            'gross_exposure': sum(abs(w) for w in scaled_weights.values())
        }

# Example
if __name__ == "__main__":
    # Multi-strategy fund
    np.random.seed(42)
    n_days = 252 * 2
    
    strategy_returns = {
        'Long/Short Equity': pd.Series(np.random.normal(0.0004, 0.008, n_days)),
        'Fixed Income Arb': pd.Series(np.random.normal(0.0002, 0.003, n_days)),
        'Global Macro': pd.Series(np.random.normal(0.0005, 0.012, n_days)),
        'Merger Arb': pd.Series(np.random.normal(0.0003, 0.004, n_days))
    }
    
    ms_rb = MultiStrategyRiskBudget(strategy_returns)
    
    # Risk budgets
    risk_budgets = {
        'Long/Short Equity': 0.35,
        'Fixed Income Arb': 0.15,
        'Global Macro': 0.35,
        'Merger Arb': 0.15
    }
    
    print("Multi-Strategy Risk Budgeting")
    print("="*70)
    
    allocation = ms_rb.allocate_by_risk_budget(risk_budgets, total_capital=1000000000)
    
    print("Strategy Allocations:")
    print(allocation['strategy_metrics'].to_string(index=False))
    print()
    print(f"Portfolio Expected Return: {allocation['portfolio_return']*100:.2f}%")
    print(f"Portfolio Volatility: {allocation['portfolio_volatility']*100:.2f}%")
    print(f"Portfolio Sharpe: {allocation['portfolio_sharpe']:.2f}")
\`\`\`

## Key Takeaways

1. **Allocate Risk, Not Capital**: Risk budgeting focuses on risk contribution
2. **Risk Parity**: Equal risk contribution from all assets
3. **Custom Budgets**: Allocate risk based on views/constraints
4. **Leverage Aware**: Risk parity often requires leverage
5. **Multi-Level**: Can apply hierarchically (asset class → assets)
6. **Dynamic**: Adjust allocations as volatilities change
7. **Better Diversification**: More balanced risk exposure

## Common Pitfalls

❌ **Ignoring Correlations**: Risk budgeting requires accurate correlations  
❌ **Static Allocations**: Volatilities change, budgets should adapt  
❌ **Leverage Constraints**: Risk parity may require leverage unavailable to some investors  
❌ **Transaction Costs**: Rebalancing to maintain risk budgets can be expensive  
❌ **Estimation Error**: Garbage in, garbage out

## Conclusion

Risk budgeting represents a fundamental shift from capital-centric to risk-centric portfolio construction:

**Traditional**: "Invest 60% in stocks" → High concentration of risk in stocks  
**Risk Budgeting**: "Take 50% of risk from stocks" → More balanced risk exposure

Bridgewater's All Weather portfolio popularized this approach, showing that balanced risk allocation can lead to better risk-adjusted returns and more stable performance across market environments.

The key insight: **Capital is abundant, risk tolerance is scarce**. Allocate your scarce resource (risk capacity) wisely.

Next: Margin and Collateral Management - managing collateral requirements for derivatives and securities financing.
`,
};
