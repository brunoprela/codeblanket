export const riskAttributionAnalysis = {
  id: 'risk-attribution-analysis',
  title: 'Risk Attribution Analysis',
  content: `
# Risk Attribution Analysis

## Introduction

"Where did my returns come from? Where did my risk come from?"

Performance attribution answers the first question. **Risk attribution** answers the second. Understanding the sources of risk is essential for:

- Evaluating if you're taking the right risks
- Identifying unintended risk exposures
- Allocating risk budgets effectively
- Understanding how risk changes over time

Just as portfolio managers want credit for alpha generation, they need to understand their risk contributions.

## What is Risk Attribution?

Risk attribution decomposes portfolio risk into components:

\`\`\`
Total Portfolio Risk = Σ Individual Risk Contributions
\`\`\`

Key questions:
- **Which positions contribute most to portfolio risk?**
- **How much risk comes from market exposure vs. stock selection?**
- **Is risk concentrated or diversified?**
- **How did risk contributions change over time?**

\`\`\`python
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass

class RiskAttributionAnalyzer:
    """
    Analyze risk contributions across portfolio
    """
    def __init__(self, 
                 weights: np.ndarray,
                 returns: pd.DataFrame,
                 covariance_matrix: np.ndarray = None):
        """
        Args:
            weights: Portfolio weights (array)
            returns: Historical returns (DataFrame with asset columns)
            covariance_matrix: Optional covariance matrix
        """
        self.weights = weights
        self.returns = returns
        self.assets = returns.columns.tolist()
        
        if covariance_matrix is None:
            self.cov_matrix = returns.cov().values
        else:
            self.cov_matrix = covariance_matrix
    
    def calculate_portfolio_variance(self) -> float:
        """
        Portfolio variance = w^T × Σ × w
        """
        return self.weights @ self.cov_matrix @ self.weights
    
    def calculate_portfolio_volatility(self, annualization_factor: int = 252) -> float:
        """
        Portfolio volatility (annualized)
        """
        portfolio_variance = self.calculate_portfolio_variance()
        return np.sqrt(portfolio_variance * annualization_factor)
    
    def calculate_marginal_risk_contributions(self) -> np.ndarray:
        """
        Marginal Contribution to Risk (MCR)
        
        MCR_i = (Σ × w)_i / σ_portfolio
        
        How much portfolio risk increases with small increase in position i
        """
        portfolio_vol = np.sqrt(self.calculate_portfolio_variance())
        
        # Marginal risk = (Covariance matrix × weights) / portfolio volatility
        marginal_risk = (self.cov_matrix @ self.weights) / portfolio_vol
        
        return marginal_risk
    
    def calculate_component_risk_contributions(self) -> pd.DataFrame:
        """
        Component Contribution to Risk (CCR)
        
        CCR_i = w_i × MCR_i
        
        Absolute risk contribution from position i
        """
        marginal_risk = self.calculate_marginal_risk_contributions()
        
        # Component risk = weight × marginal risk
        component_risk = self.weights * marginal_risk
        
        # Percentage contribution
        total_risk = self.calculate_portfolio_volatility()
        pct_contribution = (component_risk / total_risk) * 100
        
        results = []
        for i, asset in enumerate(self.assets):
            results.append({
                'asset': asset,
                'weight': self.weights[i],
                'marginal_risk': marginal_risk[i],
                'component_risk': component_risk[i],
                'pct_risk_contribution': pct_contribution[i]
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('pct_risk_contribution', ascending=False)
        
        return df
    
    def verify_risk_attribution(self) -> Dict:
        """
        Verify that component risks sum to total risk
        
        Σ CCR_i should equal portfolio volatility
        """
        component_risks = self.calculate_component_risk_contributions()
        sum_component_risks = component_risks['component_risk'].sum()
        portfolio_vol = self.calculate_portfolio_volatility()
        
        difference = abs(sum_component_risks - portfolio_vol)
        
        return {
            'sum_of_components': sum_component_risks,
            'portfolio_volatility': portfolio_vol,
            'difference': difference,
            'verified': difference < 0.001  # Within rounding error
        }
    
    def calculate_incremental_risk(self, 
                                   asset_index: int,
                                   weight_change: float) -> Dict:
        """
        Incremental risk from changing position
        
        How does portfolio risk change if we increase position i by Δw?
        """
        # Current portfolio risk
        current_vol = self.calculate_portfolio_volatility()
        
        # Modified weights
        new_weights = self.weights.copy()
        new_weights[asset_index] += weight_change
        
        # Recalculate with new weights
        new_analyzer = RiskAttributionAnalyzer(
            new_weights,
            self.returns,
            self.cov_matrix
        )
        new_vol = new_analyzer.calculate_portfolio_volatility()
        
        incremental_risk = new_vol - current_vol
        
        return {
            'asset': self.assets[asset_index],
            'weight_change': weight_change,
            'current_volatility': current_vol,
            'new_volatility': new_vol,
            'incremental_risk': incremental_risk,
            'incremental_risk_pct': (incremental_risk / current_vol) * 100
        }
    
    def calculate_diversification_ratio(self) -> float:
        """
        Diversification Ratio = Weighted Average Vol / Portfolio Vol
        
        Measures benefit of diversification
        DR > 1 means diversification is working
        """
        # Individual asset volatilities
        individual_vols = np.sqrt(np.diag(self.cov_matrix))
        
        # Weighted average volatility
        weighted_avg_vol = np.dot(abs(self.weights), individual_vols)
        
        # Portfolio volatility
        portfolio_vol = np.sqrt(self.calculate_portfolio_variance())
        
        # Diversification ratio
        div_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0
        
        return {
            'weighted_avg_volatility': weighted_avg_vol,
            'portfolio_volatility': portfolio_vol,
            'diversification_ratio': div_ratio,
            'diversification_benefit': (1 - 1/div_ratio) * 100 if div_ratio > 1 else 0
        }

# Example Usage
if __name__ == "__main__":
    # Sample portfolio
    np.random.seed(42)
    
    # Generate sample returns
    n_assets = 5
    n_days = 252
    
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'SPY']
    returns_data = np.random.multivariate_normal(
        mean=[0.0005] * n_assets,
        cov=np.eye(n_assets) * 0.0004 + np.ones((n_assets, n_assets)) * 0.0001,
        size=n_days
    )
    
    returns_df = pd.DataFrame(returns_data, columns=assets)
    
    # Portfolio weights
    weights = np.array([0.25, 0.20, 0.25, 0.15, 0.15])
    
    # Risk attribution
    risk_attr = RiskAttributionAnalyzer(weights, returns_df)
    
    print("Risk Attribution Analysis")
    print("="*70)
    print()
    
    # Portfolio risk
    portfolio_vol = risk_attr.calculate_portfolio_volatility()
    print(f"Portfolio Volatility (annualized): {portfolio_vol*100:.2f}%")
    print()
    
    # Component risk contributions
    component_risks = risk_attr.calculate_component_risk_contributions()
    print("Risk Contributions by Asset:")
    print(component_risks.to_string(index=False))
    print()
    
    # Verify attribution
    verification = risk_attr.verify_risk_attribution()
    print("Attribution Verification:")
    print(f"  Sum of Components: {verification['sum_of_components']:.6f}")
    print(f"  Portfolio Volatility: {verification['portfolio_volatility']:.6f}")
    print(f"  Verified: {'✓' if verification['verified'] else '✗'}")
    print()
    
    # Diversification ratio
    div_ratio = risk_attr.calculate_diversification_ratio()
    print(f"Diversification Ratio: {div_ratio['diversification_ratio']:.2f}")
    print(f"Diversification Benefit: {div_ratio['diversification_benefit']:.1f}%")
    print()
    
    # Incremental risk
    incremental = risk_attr.calculate_incremental_risk(asset_index=0, weight_change=0.05)
    print(f"Incremental Risk Analysis (Add 5% to {incremental['asset']}):")
    print(f"  Current Vol: {incremental['current_volatility']*100:.2f}%")
    print(f"  New Vol: {incremental['new_volatility']*100:.2f}%")
    print(f"  Incremental Risk: {incremental['incremental_risk']*100:.2f}% ({incremental['incremental_risk_pct']:.1f}%)")
\`\`\`

## Factor Risk Attribution

Decompose risk by factors (market, sectors, styles):

\`\`\`python
class FactorRiskAttribution:
    """
    Attribute risk to systematic factors
    """
    def __init__(self,
                 portfolio_returns: pd.Series,
                 factor_returns: pd.DataFrame):
        """
        Args:
            portfolio_returns: Portfolio return series
            factor_returns: Factor returns (market, size, value, etc.)
        """
        self.portfolio_returns = portfolio_returns
        self.factor_returns = factor_returns
        
    def run_factor_regression(self) -> Dict:
        """
        Regress portfolio returns on factors
        
        R_portfolio = α + Σ β_i × F_i + ε
        """
        from sklearn.linear_model import LinearRegression
        
        # Prepare data
        X = self.factor_returns.values
        y = self.portfolio_returns.values
        
        # Run regression
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        
        # Results
        alpha = model.intercept_
        betas = model.coef_
        r_squared = model.score(X, y)
        
        # Predicted returns
        predicted = model.predict(X)
        residuals = y - predicted
        
        # Factor contributions to return
        factor_contributions = {}
        for i, factor in enumerate(self.factor_returns.columns):
            contribution = betas[i] * self.factor_returns[factor].mean()
            factor_contributions[factor] = contribution
        
        return {
            'alpha': alpha,
            'betas': dict(zip(self.factor_returns.columns, betas)),
            'r_squared': r_squared,
            'factor_contributions': factor_contributions,
            'residuals': residuals
        }
    
    def decompose_risk_by_factors(self) -> pd.DataFrame:
        """
        Decompose portfolio variance into factor components
        
        σ²_portfolio = Σ β²_i × σ²_factor_i + σ²_residual
        """
        regression = self.run_factor_regression()
        betas = regression['betas']
        
        # Factor variances
        factor_variances = self.factor_returns.var()
        
        # Risk contributions
        risk_decomposition = []
        total_var = 0
        
        for factor, beta in betas.items():
            factor_var = factor_variances[factor]
            factor_risk_contribution = (beta ** 2) * factor_var
            total_var += factor_risk_contribution
            
            risk_decomposition.append({
                'factor': factor,
                'beta': beta,
                'factor_volatility': np.sqrt(factor_var),
                'risk_contribution': factor_risk_contribution,
                'volatility_contribution': np.sqrt(factor_risk_contribution)
            })
        
        # Residual/idiosyncratic risk
        residual_var = np.var(regression['residuals'])
        total_var += residual_var
        
        risk_decomposition.append({
            'factor': 'Idiosyncratic',
            'beta': 1.0,
            'factor_volatility': np.sqrt(residual_var),
            'risk_contribution': residual_var,
            'volatility_contribution': np.sqrt(residual_var)
        })
        
        df = pd.DataFrame(risk_decomposition)
        
        # Calculate percentages
        df['pct_of_total_variance'] = (df['risk_contribution'] / total_var) * 100
        
        return df
    
    def calculate_factor_exposures(self) -> pd.DataFrame:
        """
        Current factor exposures (betas)
        """
        regression = self.run_factor_regression()
        betas = regression['betas']
        
        exposures = []
        for factor, beta in betas.items():
            exposures.append({
                'factor': factor,
                'exposure': beta,
                'interpretation': self._interpret_beta(beta)
            })
        
        return pd.DataFrame(exposures)
    
    @staticmethod
    def _interpret_beta(beta: float) -> str:
        """Interpret beta magnitude"""
        if abs(beta) < 0.2:
            return 'Neutral'
        elif beta > 0.8:
            return 'Strong Positive'
        elif beta > 0.2:
            return 'Moderate Positive'
        elif beta < -0.8:
            return 'Strong Negative'
        else:
            return 'Moderate Negative'

# Example
if __name__ == "__main__":
    # Sample data
    np.random.seed(42)
    n_days = 252
    
    # Generate factor returns (Market, Size, Value)
    market_returns = np.random.normal(0.0004, 0.01, n_days)
    size_returns = np.random.normal(0.0002, 0.008, n_days)
    value_returns = np.random.normal(0.0001, 0.007, n_days)
    
    factor_returns_df = pd.DataFrame({
        'Market': market_returns,
        'Size': size_returns,
        'Value': value_returns
    })
    
    # Generate portfolio returns as combination of factors + alpha
    betas_true = [1.2, 0.3, -0.1]  # True betas
    portfolio_returns_array = (
        0.0001 +  # Alpha
        1.2 * market_returns +
        0.3 * size_returns +
        -0.1 * value_returns +
        np.random.normal(0, 0.005, n_days)  # Idiosyncratic
    )
    
    portfolio_returns_series = pd.Series(portfolio_returns_array)
    
    # Factor attribution
    factor_attr = FactorRiskAttribution(portfolio_returns_series, factor_returns_df)
    
    print("Factor Risk Attribution")
    print("="*70)
    print()
    
    # Regression results
    regression = factor_attr.run_factor_regression()
    print(f"Alpha: {regression['alpha']*252*100:.2f}% annual")
    print(f"R-squared: {regression['r_squared']:.3f}")
    print()
    print("Factor Betas:")
    for factor, beta in regression['betas'].items():
        print(f"  {factor}: {beta:.3f}")
    print()
    
    # Risk decomposition
    risk_decomp = factor_attr.decompose_risk_by_factors()
    print("Risk Decomposition:")
    print(risk_decomp[['factor', 'beta', 'volatility_contribution', 'pct_of_total_variance']].to_string(index=False))
\`\`\`

## Tracking Error Attribution

For portfolios managed against a benchmark:

\`\`\`python
class TrackingErrorAttribution:
    """
    Attribute tracking error (active risk) to sources
    """
    def __init__(self,
                 portfolio_weights: np.ndarray,
                 benchmark_weights: np.ndarray,
                 returns: pd.DataFrame):
        """
        Args:
            portfolio_weights: Portfolio weights
            benchmark_weights: Benchmark weights
            returns: Asset returns
        """
        self.portfolio_weights = portfolio_weights
        self.benchmark_weights = benchmark_weights
        self.active_weights = portfolio_weights - benchmark_weights
        self.returns = returns
        self.cov_matrix = returns.cov().values
        
    def calculate_tracking_error(self, annualization_factor: int = 252) -> float:
        """
        Tracking Error = std(portfolio return - benchmark return)
        
        Also: TE = √(active_weights^T × Σ × active_weights)
        """
        active_variance = self.active_weights @ self.cov_matrix @ self.active_weights
        tracking_error = np.sqrt(active_variance * annualization_factor)
        
        return tracking_error
    
    def attribute_tracking_error(self) -> pd.DataFrame:
        """
        Decompose tracking error by position
        """
        te = np.sqrt(self.active_weights @ self.cov_matrix @ self.active_weights)
        
        # Marginal contribution to TE
        marginal_te = (self.cov_matrix @ self.active_weights) / te
        
        # Component contribution
        component_te = self.active_weights * marginal_te
        
        results = []
        assets = self.returns.columns.tolist()
        
        for i, asset in enumerate(assets):
            results.append({
                'asset': asset,
                'portfolio_weight': self.portfolio_weights[i],
                'benchmark_weight': self.benchmark_weights[i],
                'active_weight': self.active_weights[i],
                'marginal_te': marginal_te[i],
                'component_te': component_te[i],
                'pct_te_contribution': (component_te[i] / te) * 100 if te > 0 else 0
            })
        
        df = pd.DataFrame(results)
        df = df[df['active_weight'].abs() > 0.001]  # Filter near-zero active positions
        df = df.sort_values('pct_te_contribution', ascending=False, key=abs)
        
        return df
    
    def calculate_information_ratio(self, 
                                   portfolio_returns: pd.Series,
                                   benchmark_returns: pd.Series) -> Dict:
        """
        Information Ratio = Active Return / Tracking Error
        
        Measures active management skill
        """
        active_returns = portfolio_returns - benchmark_returns
        
        active_return_ann = active_returns.mean() * 252
        tracking_error = active_returns.std() * np.sqrt(252)
        
        information_ratio = active_return_ann / tracking_error if tracking_error > 0 else 0
        
        return {
            'active_return': active_return_ann,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio
        }

# Example
if __name__ == "__main__":
    # Portfolio vs benchmark
    n_assets = 5
    portfolio_weights = np.array([0.25, 0.20, 0.25, 0.20, 0.10])
    benchmark_weights = np.array([0.20, 0.20, 0.20, 0.20, 0.20])  # Equal weight benchmark
    
    # Returns
    np.random.seed(42)
    returns_data = np.random.multivariate_normal(
        mean=[0.0005] * n_assets,
        cov=np.eye(n_assets) * 0.0004 + np.ones((n_assets, n_assets)) * 0.0001,
        size=252
    )
    
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'SPY']
    returns_df = pd.DataFrame(returns_data, columns=assets)
    
    te_attr = TrackingErrorAttribution(portfolio_weights, benchmark_weights, returns_df)
    
    print("Tracking Error Attribution")
    print("="*70)
    print()
    
    # Tracking error
    te = te_attr.calculate_tracking_error()
    print(f"Tracking Error: {te*100:.2f}%")
    print()
    
    # Attribution
    te_decomp = te_attr.attribute_tracking_error()
    print("Tracking Error Decomposition:")
    print(te_decomp.to_string(index=False))
\`\`\`

## Key Takeaways

1. **Marginal vs Component**: Marginal = incremental, Component = total contribution
2. **Attribution Verification**: Components must sum to total risk
3. **Factor Decomposition**: Systematic vs idiosyncratic risk
4. **Tracking Error**: Active risk for benchmark-relative portfolios
5. **Diversification Benefit**: Quantify value of diversification
6. **Incremental Risk**: Impact of trade on portfolio risk
7. **Time-Varying**: Risk attribution changes as markets evolve

## Conclusion

Risk attribution transforms a black box (portfolio risk number) into actionable insights:
- Which positions drive risk?
- Are we taking intended risks?
- How concentrated is risk?
- Where should we adjust?

Just as performance attribution shows where returns came from, risk attribution shows where risk comes from - essential for intelligent risk management.

Next: Risk Budgeting - allocating risk intentionally across strategies.
`,
};
