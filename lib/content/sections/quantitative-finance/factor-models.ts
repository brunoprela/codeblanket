export const factorModels = {
  id: 'factor-models',
  title: 'Factor Models & Attribution',
  content: `
# Factor Models & Attribution

## Introduction

Factor models are the foundation of modern quantitative portfolio management, providing a systematic framework for understanding sources of return and risk. Moving beyond the single-factor CAPM, multi-factor models decompose asset returns into exposures to various systematic risk factors (market, size, value, momentum, quality, low volatility) and idiosyncratic risk.

The **Fama-French Three-Factor Model** (1992) revolutionized finance by showing that size (SMB - Small Minus Big) and value (HML - High Minus Low) factors explain cross-sectional return variation better than CAPM alone. This was later extended to the **Five-Factor Model** (2015) adding profitability (RMW - Robust Minus Weak) and investment (CMA - Conservative Minus Aggressive) factors.

**Factor investing** has become a trillion-dollar industry, with investors seeking to harvest systematic risk premia through **smart beta** ETFs and quantitative strategies. Understanding factor models is essential for:
- **Portfolio construction**: Building diversified factor exposures
- **Performance attribution**: Decomposing returns into factor contributions
- **Risk management**: Identifying concentrated factor bets
- **Alpha generation**: Exploiting factor mispricings and anomalies

This section covers factor model theory, construction methodologies, empirical evidence, and Python implementation for practical factor analysis and portfolio optimization.

---

## Fama-French Three-Factor Model

### Mathematical Formulation

The Fama-French Three-Factor Model extends CAPM by adding size and value factors:

\[
R_i - R_f = \\alpha_i + \\beta_{i,M}(R_M - R_f) + \\beta_{i,SMB} \\cdot SMB + \\beta_{i,HML} \\cdot HML + \\epsilon_i
\]

**Variables:**
- \(R_i - R_f\): Excess return of asset \(i\) over risk-free rate
- \(\\alpha_i\): Jensen\'s alpha (risk-adjusted excess return)
- \(\\beta_{i,M}\): Market factor loading (systematic risk)
- \(R_M - R_f\): Market risk premium (equity market excess return)
- \(\\beta_{i,SMB}\): Size factor loading (small-cap exposure)
- \(SMB\): **Small Minus Big** - return difference between small and large cap portfolios
- \(\\beta_{i,HML}\): Value factor loading (value stock exposure)
- \(HML\): **High Minus Low** - return difference between high and low book-to-market portfolios
- \(\\epsilon_i\): Idiosyncratic error term (firm-specific risk)

### Factor Construction Methodology

**SMB (Size Factor):**
1. Sort all stocks by market capitalization
2. Split into two groups: Small (bottom 50%) and Big (top 50%)
3. Calculate value-weighted returns for each group
4. \(SMB = R_{Small} - R_{Big}\)
5. Historical premium: ~2-3% annually (varies by period/market)

**HML (Value Factor):**
1. Sort stocks by book-to-market ratio (BE/ME = Book Equity / Market Equity)
2. Split into three groups: High (top 30%), Medium (middle 40%), Low (bottom 30%)
3. Calculate value-weighted returns for High and Low portfolios
4. \(HML = R_{High\\_BE/ME} - R_{Low\\_BE/ME}\)
5. Historical premium: ~3-5% annually (strong value premium)

### Economic Intuition

**Why does the size premium exist?**
- **Liquidity risk**: Small caps are less liquid → higher required return
- **Information asymmetry**: Less analyst coverage → higher uncertainty
- **Financial distress**: Higher bankruptcy risk during recessions
- **Transaction costs**: Higher bid-ask spreads and price impact

**Why does the value premium exist?**
- **Distress risk**: Value stocks often have poor recent performance → perceived higher risk
- **Behavioral bias**: Investors overextrapolate growth for glamour stocks (growth stocks)
- **Mean reversion**: Value stocks are temporarily undervalued, revert to fair value
- **Risk compensation**: Value stocks more sensitive to economic cycles

### Regression Example

Regressing a stock's excess returns on the three factors:

\[
R_{AAPL} - R_f = \\alpha + 1.2(R_M - R_f) - 0.3 \\cdot SMB - 0.5 \\cdot HML + \\epsilon
\]

**Interpretation:**
- \(\\beta_M = 1.2\): AAPL is 20% more volatile than market (tech stock, high beta)
- \(\\beta_{SMB} = -0.3\): Negative exposure to size factor (large cap stock)
- \(\\beta_{HML} = -0.5\): Negative exposure to value factor (growth stock, low BE/ME)
- \(\\alpha = 0.02\) (2%): Outperforming by 2% annually after adjusting for factor exposures

**Portfolio implication:** AAPL behaves like a large-cap growth stock with high market sensitivity. To hedge, go long small-cap value stocks.

---

## Fama-French Five-Factor Model

### Extended Formulation

The Five-Factor Model adds profitability and investment factors:

\[
R_i - R_f = \\alpha_i + \\beta_{M}(R_M - R_f) + \\beta_{SMB} \\cdot SMB + \\beta_{HML} \\cdot HML + \\beta_{RMW} \\cdot RMW + \\beta_{CMA} \\cdot CMA + \\epsilon_i
\]

**New Factors:**

**RMW (Robust Minus Weak - Profitability Factor):**
- \(RMW = R_{High\\_Profitability} - R_{Low\\_Profitability}\)
- Profitability measured as: Operating Profit / Book Equity
- Sort stocks into Robust (top 30%) and Weak (bottom 30%) by profitability
- **Economic rationale**: More profitable firms generate higher returns (quality premium)
- Historical premium: ~2-3% annually

**CMA (Conservative Minus Aggressive - Investment Factor):**
- \(CMA = R_{Low\\_Investment} - R_{High\\_Investment}\)
- Investment measured as: Annual change in total assets / lagged total assets
- Conservative = low asset growth, Aggressive = high asset growth
- **Economic rationale**: Firms with low investment (conservative) outperform high-investment firms
- Theory: High investment firms often overinvest, destroying shareholder value
- Historical premium: ~2-3% annually

### Improvement Over Three-Factor Model

**Why add RMW and CMA?**
1. **Better explanatory power**: Five-factor model explains 71-73% of variance vs 68-70% for three-factor (higher \(R^2\))
2. **Subsumes HML**: In many tests, HML becomes insignificant when RMW and CMA are added
3. **Profitability**: Directly captures quality dimension (profitable firms outperform unprofitable)
4. **Investment**: Captures growth vs value more precisely than book-to-market alone

**Empirical evidence:**
- Five-factor model reduces average absolute alpha from 0.30% to 0.18% monthly
- Works better for explaining small-cap anomalies
- International evidence: Factors also work in Europe, Asia, emerging markets

---

## Momentum Factor

### Carhart Four-Factor Model

Mark Carhart (1997) added momentum to Fama-French three-factor model:

\[
R_i - R_f = \\alpha_i + \\beta_{M}(R_M - R_f) + \\beta_{SMB} \\cdot SMB + \\beta_{HML} \\cdot HML + \\beta_{MOM} \\cdot MOM + \\epsilon_i
\]

**MOM (Momentum Factor):**
- \(MOM = R_{Winners} - R_{Losers}\) or UMD (Up Minus Down)
- **Construction**: Sort stocks by past 12-month returns (skipping most recent month)
- Winners = top 30% performers, Losers = bottom 30%
- Rebalance monthly
- Historical premium: ~6-8% annually (strongest factor premium!)

### Momentum Mechanics

**Why skip the most recent month?**
- Avoids short-term reversal effect (bid-ask bounce, microstructure noise)
- One-month reversal: Stocks that went up last month tend to reverse slightly
- But 12-month momentum (skipping last month) is strongly positive

**Why does momentum work?**
1. **Underreaction**: Investors slow to incorporate new information → drift continues
2. **Behavioral biases**: Anchoring, representativeness, confirmation bias
3. **Herding**: Positive feedback trading amplifies trends
4. **Risk-based**: Momentum crashes during market reversals (especially in panic recoveries)

**Momentum characteristics:**
- Works across asset classes: equities, bonds, commodities, currencies
- Time-series momentum: Asset's own past return predicts future return
- Cross-sectional momentum: Relative performance (winners vs losers) persists
- **Momentum crashes**: Large losses during sharp market reversals (e.g., 2009 rebound)

### Momentum Risk Management

**Drawdowns during reversals:**
- 2009: Momentum lost ~75% during market recovery (losers bounced more than winners)
- Solution: **Dynamic allocation** - reduce momentum exposure when dispersion is high
- **Volatility scaling**: Scale positions inversely with realized volatility

**Time-varying momentum:**
\[
w_{MOM,t} = \\frac{1}{\\sigma_{MOM,t}} \\cdot \\text{base\\_weight}
\]
- When momentum volatility is high → reduce exposure
- When volatility is low → increase exposure
- Improves Sharpe ratio by ~30% vs static momentum

---

## Factor Attribution

### Performance Attribution Framework

Factor attribution decomposes portfolio returns into systematic and idiosyncratic components:

\[
R_p - R_f = \\sum_{k=1}^{K} \\beta_{p,k} \\cdot F_k + \\alpha_p + \\epsilon_p
\]

Where:
- \(R_p - R_f\): Portfolio excess return
- \(\\beta_{p,k}\): Portfolio\'s exposure to factor \(k\)
- \(F_k\): Return of factor \(k\) (e.g., market, size, value, momentum)
- \(\\alpha_p\): True skill-based alpha (after adjusting for all factor exposures)
- \(\\epsilon_p\): Residual (unexplained return)

### Attribution Process

**Step 1: Estimate factor loadings**
Run time-series regression of portfolio returns on factor returns:
\[
R_{p,t} - R_{f,t} = \\alpha + \\beta_M F_{M,t} + \\beta_{SMB} F_{SMB,t} + \\beta_{HML} F_{HML,t} + \\epsilon_t
\]

**Step 2: Calculate factor contributions**
For each factor \(k\):
\[
\\text{Contribution}_k = \\beta_{p,k} \\times \\bar{F}_k \\times T
\]
where \(\\bar{F}_k\) is average factor return over period \(T\).

**Step 3: Decompose total return**
\[
\\text{Total Return} = R_f \\times T + \\sum_{k=1}^{K} \\text{Contribution}_k + \\alpha_p \\times T + \\epsilon
\]

### Example Attribution

Portfolio returned 15% over one year. Risk-free rate = 2%.

Factor | Beta | Factor Return | Contribution
-------|------|---------------|-------------
Market | 1.1 | 8% | 8.8%
SMB | 0.3 | 2% | 0.6%
HML | 0.5 | 4% | 2.0%
MOM | 0.2 | 6% | 1.2%
**Total Factor** | | | **12.6%**
Alpha | | | **0.4%**
Risk-free | | 2% | **2.0%**
**Total Return** | | | **15.0%**

**Interpretation:**
- 12.6% from factor exposures (systematic risk)
- 0.4% from alpha (manager skill or luck)
- 84% of return explained by factors, 2.7% from alpha
- **Conclusion**: Manager is mostly taking systematic factor bets, minimal true alpha

---

## Factor Investing Strategies

### Smart Beta

**Definition:** Rules-based strategies that target specific factor exposures (alternative to market-cap weighting).

**Common smart beta strategies:**
1. **Equal weight**: All stocks get 1/N weight → overweights small caps vs market-cap
2. **Minimum volatility**: Optimize for lowest portfolio volatility → defensive strategy
3. **Fundamental weighting**: Weight by fundamentals (sales, earnings, book value) → value tilt
4. **Multi-factor**: Combine multiple factors (value + momentum + quality)

**Advantages:**
- Transparent, rules-based (no black box)
- Lower fees than active management (10-30 bps vs 50-100 bps)
- Systematic factor harvesting
- Tax-efficient (lower turnover than active)

**Disadvantages:**
- Can be crowded (factor premia compress when too popular)
- Backward-looking (based on historical data)
- Factor timing is hard (factors go through long periods of underperformance)
- Value factor struggled 2015-2020 (longest drawdown in history)

### Factor Timing

**Can you time factors?** (Active debate)

**Timing signals:**
1. **Valuation**: When value spread (valuation difference between value and growth) is wide → value should outperform
2. **Macro regime**: Momentum works in trending markets, mean reversion in choppy markets
3. **Volatility**: Low-vol factor outperforms during high-volatility regimes
4. **Economic cycle**: Value outperforms in early recovery, momentum in mid-cycle

**Empirical evidence:** Mixed. Factor timing has worked historically but hard to execute:
- High transaction costs (rebalancing between factors)
- Whipsaw risk (switching at wrong times)
- Out-of-sample performance weaker than in-sample

**Practical approach:** Core-satellite
- **Core**: Static multi-factor allocation (50% market, 20% value, 20% momentum, 10% quality)
- **Satellite**: Tactical tilts based on valuation/macro signals (small adjustments)

---

## Python Implementation: Factor Analysis

### Loading Factor Data

\`\`\`python
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Load Fama-French factors from Ken French\'s data library
# For this example, we'll simulate factor data
# In practice: download from https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

np.random.seed(42)
dates = pd.date_range('2015-01-01', '2023-12-31', freq='M')

# Simulate factor returns (monthly %)
ff_factors = pd.DataFrame({
    'Mkt-RF': np.random.normal(0.8, 4.5, len (dates)),  # Market excess return
    'SMB': np.random.normal(0.2, 3.0, len (dates)),      # Size factor
    'HML': np.random.normal(0.3, 3.5, len (dates)),      # Value factor
    'RMW': np.random.normal(0.25, 2.5, len (dates)),     # Profitability factor
    'CMA': np.random.normal(0.2, 2.0, len (dates)),      # Investment factor
    'MOM': np.random.normal(0.5, 4.0, len (dates)),      # Momentum factor
    'RF': np.random.normal(0.15, 0.3, len (dates))       # Risk-free rate
}, index=dates)

# Simulate portfolio returns
portfolio_returns = (
    0.02 +  # Alpha (2% monthly)
    1.2 * ff_factors['Mkt-RF'] +
    0.3 * ff_factors['SMB'] +
    -0.2 * ff_factors['HML'] +
    0.4 * ff_factors['MOM'] +
    np.random.normal(0, 2, len (dates))  # Idiosyncratic risk
)

portfolio_returns = pd.Series (portfolio_returns, index=dates, name='Portfolio')

print("Factor Statistics (Monthly %):")
print(ff_factors.describe())
print(f"\\nPortfolio Returns: Mean={portfolio_returns.mean():.2f}%, Std={portfolio_returns.std():.2f}%")
\`\`\`

### Factor Model Regression

\`\`\`python
class FactorModel:
    """
    Multi-factor model regression and attribution.
    """
    
    def __init__(self, returns, factors, model_type='FF5'):
        """
        Parameters:
        - returns: pd.Series of portfolio returns
        - factors: pd.DataFrame of factor returns
        - model_type: 'CAPM', 'FF3', 'FF5', 'Carhart', 'FF6'
        """
        self.returns = returns
        self.factors = factors
        self.model_type = model_type
        self.results = None
        
        # Align data
        self.data = pd.concat([returns, factors], axis=1).dropna()
    
    def select_factors (self):
        """Select factors based on model type."""
        factor_sets = {
            'CAPM': ['Mkt-RF'],
            'FF3': ['Mkt-RF', 'SMB', 'HML'],
            'Carhart': ['Mkt-RF', 'SMB', 'HML', 'MOM'],
            'FF5': ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'],
            'FF6': ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
        }
        return factor_sets.get (self.model_type, factor_sets['FF5'])
    
    def run_regression (self):
        """Run OLS regression of returns on factors."""
        factor_names = self.select_factors()
        
        # Calculate excess returns
        y = self.data[self.returns.name] - self.data['RF']
        X = self.data[factor_names]
        X = sm.add_constant(X)  # Add intercept (alpha)
        
        # Run regression
        model = sm.OLS(y, X)
        self.results = model.fit()
        
        return self.results
    
    def print_summary (self):
        """Print regression summary."""
        if self.results is None:
            self.run_regression()
        
        print(f"\\n{'='*60}")
        print(f"{self.model_type} Factor Model Regression")
        print(f"{'='*60}")
        print(self.results.summary())
        
        # Additional statistics
        print(f"\\n{'='*60}")
        print("Key Statistics:")
        print(f"{'='*60}")
        
        alpha_annual = self.results.params['const'] * 12
        alpha_tstat = self.results.tvalues['const']
        r_squared = self.results.rsquared
        adj_r_squared = self.results.rsquared_adj
        
        print(f"Alpha (monthly): {self.results.params['const']:.4f}% (t={alpha_tstat:.2f})")
        print(f"Alpha (annualized): {alpha_annual:.2f}%")
        print(f"Significant at 5%: {'YES' if abs (alpha_tstat) > 1.96 else 'NO'}")
        print(f"R-squared: {r_squared:.4f} ({r_squared*100:.2f}% variance explained)")
        print(f"Adjusted R-squared: {adj_r_squared:.4f}")
        print(f"Residual Std Dev: {np.sqrt (self.results.mse_resid):.4f}%")
        
        return self.results
    
    def attribution (self):
        """Calculate factor attribution."""
        if self.results is None:
            self.run_regression()
        
        factor_names = self.select_factors()
        
        # Calculate contributions
        betas = self.results.params[factor_names]
        factor_returns = self.data[factor_names].mean()
        contributions = betas * factor_returns
        
        # Total attribution
        alpha = self.results.params['const']
        total_factor = contributions.sum()
        rf_avg = self.data['RF'].mean()
        total_return = self.data[self.returns.name].mean()
        residual = total_return - rf_avg - total_factor - alpha
        
        # Create attribution dataframe
        attribution_df = pd.DataFrame({
            'Beta': betas,
            'Factor Return (%)': factor_returns,
            'Contribution (%)': contributions,
            'Contribution (% of excess return)': contributions / (total_return - rf_avg) * 100
        })
        
        # Summary row
        summary = pd.DataFrame({
            'Component': ['Risk-Free', 'Total Factors', 'Alpha', 'Residual', 'Total Return'],
            'Value (%)': [rf_avg, total_factor, alpha, residual, total_return],
            '% of Total': [
                rf_avg/total_return*100,
                total_factor/total_return*100,
                alpha/total_return*100,
                residual/total_return*100,
                100.0
            ]
        })
        
        print(f"\\n{'='*60}")
        print("Factor Attribution (Monthly Averages)")
        print(f"{'='*60}")
        print(attribution_df)
        print(f"\\n{summary}")
        
        return attribution_df, summary
    
    def rolling_beta (self, window=36):
        """Calculate rolling factor betas."""
        if self.results is None:
            self.run_regression()
        
        factor_names = self.select_factors()
        
        y = self.data[self.returns.name] - self.data['RF']
        X = self.data[factor_names]
        X = sm.add_constant(X)
        
        # Rolling regression
        rolling_model = RollingOLS(y, X, window=window)
        rolling_results = rolling_model.fit()
        
        # Extract rolling betas
        rolling_betas = rolling_results.params[factor_names]
        
        return rolling_betas
    
    def plot_rolling_betas (self, window=36):
        """Plot rolling factor betas over time."""
        rolling_betas = self.rolling_beta (window)
        
        fig, axes = plt.subplots (len (rolling_betas.columns), 1, 
                                 figsize=(12, 3*len (rolling_betas.columns)))
        
        if len (rolling_betas.columns) == 1:
            axes = [axes]
        
        for i, factor in enumerate (rolling_betas.columns):
            axes[i].plot (rolling_betas.index, rolling_betas[factor], 
                        linewidth=2, color='steelblue')
            axes[i].axhline (y=0, color='red', linestyle='--', alpha=0.7)
            axes[i].axhline (y=rolling_betas[factor].mean(), 
                          color='green', linestyle='--', alpha=0.7, 
                          label=f'Mean: {rolling_betas[factor].mean():.2f}')
            axes[i].set_title (f'Rolling {factor} Beta ({window}-month window)', 
                            fontsize=12, fontweight='bold')
            axes[i].set_ylabel('Beta', fontsize=10)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Date', fontsize=10)
        plt.tight_layout()
        plt.savefig('rolling_factor_betas.png', dpi=300, bbox_inches='tight')
        plt.show()

# Run factor analysis
print("="*60)
print("FACTOR MODEL ANALYSIS")
print("="*60)

# Compare different models
models = ['CAPM', 'FF3', 'Carhart', 'FF5', 'FF6']
results_summary = []

for model_type in models:
    print(f"\\n{'#'*60}")
    print(f"Model: {model_type}")
    print(f"{'#'*60}")
    
    model = FactorModel (portfolio_returns, ff_factors, model_type=model_type)
    results = model.run_regression()
    
    results_summary.append({
        'Model': model_type,
        'Alpha (%)': results.params['const'],
        't-stat': results.tvalues['const'],
        'R²': results.rsquared,
        'Adj R²': results.rsquared_adj,
        'AIC': results.aic,
        'BIC': results.bic
    })
    
    # Print key results
    print(f"Alpha: {results.params['const']:.4f}% (t={results.tvalues['const']:.2f})")
    print(f"R²: {results.rsquared:.4f}, Adj R²: {results.rsquared_adj:.4f}")

# Compare models
comparison_df = pd.DataFrame (results_summary)
print(f"\\n{'='*60}")
print("MODEL COMPARISON")
print(f"{'='*60}")
print(comparison_df.to_string (index=False))
print(f"\\nBest by Adj R²: {comparison_df.loc[comparison_df['Adj R²'].idxmax(), 'Model']}")
print(f"Best by AIC: {comparison_df.loc[comparison_df['AIC'].idxmin(), 'Model']}")
\`\`\`

### Factor Portfolio Construction

\`\`\`python
def construct_factor_portfolio (target_factors, factor_cov, max_weight=0.4):
    """
    Construct portfolio with target factor exposures.
    
    Parameters:
    - target_factors: dict of desired factor exposures {'Mkt-RF': 1.0, 'HML': 0.5, ...}
    - factor_cov: covariance matrix of factors
    - max_weight: maximum weight per factor
    
    Returns:
    - Optimal factor weights
    """
    n_factors = len (target_factors)
    target_vector = np.array (list (target_factors.values()))
    
    # Objective: minimize tracking error variance
    def objective (weights):
        return weights @ factor_cov @ weights
    
    # Constraint: match target factor exposures
    def factor_constraint (weights):
        return weights - target_vector
    
    # Bounds: 0 to max_weight
    bounds = [(0, max_weight) for _ in range (n_factors)]
    
    # Initial guess
    x0 = target_vector / target_vector.sum()
    
    # Optimize
    constraints = {'type': 'eq', 'fun': factor_constraint}
    result = minimize (objective, x0, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    if result.success:
        weights = pd.Series (result.x, index=list (target_factors.keys()))
        return weights
    else:
        raise ValueError("Optimization failed")

# Example: Construct multi-factor portfolio
print("\\n" + "="*60)
print("FACTOR PORTFOLIO CONSTRUCTION")
print("="*60)

# Target exposures
target_factors = {
    'Mkt-RF': 1.0,   # Market beta = 1.0
    'SMB': 0.3,      # Small-cap tilt
    'HML': 0.5,      # Value tilt
    'MOM': 0.4       # Momentum exposure
}

# Calculate factor covariance (from historical data)
factor_list = list (target_factors.keys())
factor_cov = ff_factors[factor_list].cov()

print("\\nTarget Factor Exposures:")
for factor, beta in target_factors.items():
    print(f"  {factor}: {beta:.2f}")

print("\\nFactor Covariance Matrix:")
print(factor_cov)

# Construct portfolio
weights = construct_factor_portfolio (target_factors, factor_cov)

print("\\nOptimal Factor Weights:")
print(weights)

# Calculate expected return and risk
factor_means = ff_factors[factor_list].mean()
expected_return = weights @ factor_means
portfolio_variance = weights @ factor_cov @ weights
portfolio_std = np.sqrt (portfolio_variance)

print(f"\\nExpected Monthly Return: {expected_return:.4f}%")
print(f"Expected Annual Return: {expected_return * 12:.2f}%")
print(f"Monthly Std Dev: {portfolio_std:.4f}%")
print(f"Annual Std Dev: {portfolio_std * np.sqrt(12):.2f}%")
print(f"Sharpe Ratio (ann.): {(expected_return * 12) / (portfolio_std * np.sqrt(12)):.2f}")
\`\`\`

### Factor Risk Decomposition

\`\`\`python
def factor_risk_decomposition (portfolio_betas, factor_cov, idio_var):
    """
    Decompose portfolio variance into factor and idiosyncratic components.
    
    Variance = β' * Σ * β + σ²_ε
    """
    # Factor contribution to variance
    factor_variance = portfolio_betas @ factor_cov @ portfolio_betas
    
    # Total variance
    total_variance = factor_variance + idio_var
    
    # Individual factor contributions (marginal contribution to variance)
    factor_contrib = {}
    for i, factor in enumerate (portfolio_betas.index):
        # MCTR = ∂σ²/∂β_i = 2 * (Σ * β)_i
        marginal_contrib = 2 * (factor_cov @ portfolio_betas)[i] * portfolio_betas[i]
        factor_contrib[factor] = marginal_contrib
    
    risk_decomp = pd.DataFrame({
        'Beta': portfolio_betas,
        'Marginal Risk': factor_contrib,
        '% of Total Var': [v / total_variance * 100 for v in factor_contrib.values()]
    })
    
    # Add idiosyncratic
    idio_row = pd.DataFrame({
        'Beta': [np.nan],
        'Marginal Risk': [idio_var],
        '% of Total Var': [idio_var / total_variance * 100]
    }, index=['Idiosyncratic'])
    
    risk_decomp = pd.concat([risk_decomp, idio_row])
    
    print("\\n" + "="*60)
    print("FACTOR RISK DECOMPOSITION")
    print("="*60)
    print(risk_decomp)
    print(f"\\nTotal Variance: {total_variance:.6f}")
    print(f"Factor Variance: {factor_variance:.6f} ({factor_variance/total_variance*100:.2f}%)")
    print(f"Idiosyncratic Variance: {idio_var:.6f} ({idio_var/total_variance*100:.2f}%)")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart: Factor vs Idiosyncratic
    labels = ['Factor Risk', 'Idiosyncratic Risk']
    sizes = [factor_variance/total_variance*100, idio_var/total_variance*100]
    colors = ['#FF6B6B', '#4ECDC4']
    ax1.pie (sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.set_title('Risk Decomposition: Factor vs Idiosyncratic', fontweight='bold')
    
    # Bar chart: Individual factor contributions
    factor_pct = risk_decomp.loc[risk_decomp.index != 'Idiosyncratic', '% of Total Var']
    ax2.bar (range (len (factor_pct)), factor_pct, color='steelblue', alpha=0.7)
    ax2.set_xticks (range (len (factor_pct)))
    ax2.set_xticklabels (factor_pct.index, rotation=45, ha='right')
    ax2.set_ylabel('% of Total Variance', fontsize=10)
    ax2.set_title('Individual Factor Risk Contributions', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('factor_risk_decomposition.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return risk_decomp

# Example: Decompose portfolio risk
print("\\nAnalyzing portfolio from FF6 model...")
model = FactorModel (portfolio_returns, ff_factors, model_type='FF6')
results = model.run_regression()

# Extract betas (excluding alpha)
factor_names = model.select_factors()
portfolio_betas = results.params[factor_names]

# Factor covariance
factor_cov_matrix = ff_factors[factor_names].cov()

# Idiosyncratic variance (residual variance)
idio_var = results.mse_resid

# Decompose risk
risk_decomp = factor_risk_decomposition (portfolio_betas, factor_cov_matrix, idio_var)
\`\`\`

---

## Real-World Applications

### 1. **Hedge Fund Performance Evaluation**

**Scenario:** Evaluating if a hedge fund's 15% annual return is due to skill or factor exposures.

**Approach:**
1. Run FF6 regression on fund returns
2. Extract alpha (after adjusting for market, size, value, profitability, investment, momentum)
3. Test statistical significance (t-stat > 2.0 for 5% significance)
4. If alpha is insignificant, fund is just a "beta multiplier" (expensive factor exposure)

**Red flags:**
- High factor loadings but low alpha → could replicate with ETFs
- Negative alpha → destroying value after fees
- \(R^2 < 0.5\) → high idiosyncratic risk (stock-picking risk without compensation)

### 2. **Smart Beta ETF Construction**

**Multi-factor ETF design:**
- Target: 30% value, 30% momentum, 20% quality, 20% low-vol
- Constraint: Market beta = 1.0 (no market timing)
- Rebalance quarterly to maintain factor exposures
- Expected premium: Value (3%) + Momentum (5%) + Quality (2%) + Low-vol (2%) = ~12% above market

**Backtesting considerations:**
- Transaction costs (0.1-0.3% per rebalance) → drag on returns
- Crowding: When AUM grows, factor premia compress (capacity constraints)
- Factor crashes: Momentum can lose 50% in reversals → need risk management

### 3. **Factor Timing with Valuation Spreads**

**Value timing signal:**
- When value spread (P/E ratio of growth / P/E ratio of value) is at 90th percentile historically → extreme valuations
- Action: Overweight value factor (HML loading from 0.5 to 0.8)
- Mean reversion: Wide spreads tend to contract → value outperformance

**Historical example:**
- 2000: Value spread at all-time high (dot-com bubble) → value massively outperformed 2000-2007
- 2020: Value spread very wide → value rallied 2021-2022

### 4. **Risk Parity with Factor Models**

**Factor risk parity:**
- Traditional risk parity: Equal risk contribution from asset classes
- Factor risk parity: Equal risk contribution from FACTORS (not assets)
- Formula: Weight factor \(k\) such that \(w_k \\cdot \\text{MCTR}_k = \\frac{1}{K} \\cdot \\sigma_p^2\)

**Benefits:**
- Diversifies across risk factors (market, value, momentum, carry)
- Reduces concentration risk (often 80%+ risk from equity beta alone)
- Smoother returns (lower drawdowns than equity-heavy portfolios)

---

## Key Takeaways

1. **Multi-factor models** (FF3, FF5, FF6) explain 70-95% of cross-sectional return variation-far better than CAPM alone
2. **Fama-French factors** (size, value, profitability, investment) capture systematic risk premia with economic and behavioral rationales
3. **Momentum** is the strongest factor premium (~6-8% annually) but has severe crash risk during market reversals
4. **Factor attribution** decomposes returns into systematic (factor) and idiosyncratic (alpha) components-essential for performance evaluation
5. **Smart beta strategies** provide systematic factor exposure at low cost, but suffer from crowding and time-varying premia
6. **Factor timing** is possible but difficult-valuation spreads and macro regimes provide some predictive power
7. **Risk decomposition** reveals that most portfolio risk comes from factor exposures, not stock selection (typically 80%+ factor, 20% idiosyncratic)
8. **Practical implementation** requires careful attention to transaction costs, rebalancing frequency, and capacity constraints

Understanding factor models is foundational for modern quantitative investing-enabling systematic risk harvesting, intelligent portfolio construction, and rigorous performance attribution.
`,
};
