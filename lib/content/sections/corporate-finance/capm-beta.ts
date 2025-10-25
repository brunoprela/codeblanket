export const capmBeta = {
  title: 'CAPM & Beta',
  id: 'capm-beta',
  content: `
# CAPM & Beta

## Introduction

The **Capital Asset Pricing Model (CAPM)** is one of the most important theories in finance. Despite its simplifications and criticisms, it remains the primary tool for calculating cost of equity.

**CAPM answers:** "What return should I require for taking on risk?"

**Beta (β) measures:** "How risky is this investment relative to the market?"

By the end of this section, you'll understand:
- The theory and assumptions behind CAPM
- How to calculate and interpret beta
- Different types of betas (levered, unlevered, adjusted)
- Limitations of CAPM and alternatives
- How to apply CAPM in practice

### The Fundamental Insight

**Not all risk is rewarded!**

- **Diversifiable risk** (unsystematic): Can be eliminated → Not rewarded
- **Market risk** (systematic): Cannot be diversified away → Rewarded

**CAPM focuses only on systematic risk (beta)**

---

## The CAPM Equation

### Basic Formula

\`\`\`
E(Ri) = Rf + βi × [E(Rm) - Rf]

Where:
- E(Ri) = Expected return on asset i
- Rf = Risk-free rate
- βi = Beta of asset i
- E(Rm) = Expected return on market
- [E(Rm) - Rf] = Market risk premium
\`\`\`

### Intuition

Think of CAPM as a recipe:

1. **Start with risk-free rate** (Rf): Compensation for time value of money
2. **Add risk premium** (β × MRP): Compensation for market risk

**Example:**
- Risk-free rate = 4% (10-year Treasury)
- Market risk premium = 6.5%
- Stock beta = 1.3

Required return = 4% + 1.3(6.5%) = **12.45%**

**Interpretation:** Investors require 12.45% return to hold this stock given its risk.

### Python Implementation

\`\`\`python
"""
Capital Asset Pricing Model (CAPM)
"""

import numpy as np
import pandas as pd
from typing import Union, List

def capm(
    risk_free_rate: float,
    beta: float,
    market_risk_premium: float
) -> float:
    """
    Calculate expected return using CAPM.
    
    Args:
        risk_free_rate: Risk-free rate (e.g., 10-year Treasury)
        beta: Stock\'s beta (systematic risk)
        market_risk_premium: Expected market return - risk-free rate
    
    Returns:
        Expected return (cost of equity)
        
    Example:
        >>> expected_return = capm(0.04, 1.2, 0.065)
        >>> print(f"Expected return: {expected_return:.2%}")
        Expected return: 11.80%
    """
    return risk_free_rate + beta * market_risk_premium


def capm_batch(
    risk_free_rate: float,
    betas: List[float],
    market_risk_premium: float
) -> pd.DataFrame:
    """
    Calculate CAPM returns for multiple assets.
    
    Useful for portfolio analysis.
    """
    returns = [capm (risk_free_rate, beta, market_risk_premium) for beta in betas]
    
    return pd.DataFrame({
        'Beta': betas,
        'Expected Return': returns
    })


# Example: Portfolio of stocks
rf = 0.04      # 4% risk-free rate
mrp = 0.065    # 6.5% market risk premium

stocks = {
    'Utility Stock': 0.6,
    'Consumer Staple': 0.8,
    'S&P 500 ETF': 1.0,
    'Tech Stock': 1.4,
    'Small Cap Growth': 1.8,
}

print("CAPM Expected Returns:")
print("=" * 60)
for name, beta in stocks.items():
    expected_return = capm (rf, beta, mrp)
    print(f"{name:.<25} β={beta:.1f}  →  E(R)={expected_return:.2%}")

print("=" * 60)

# Output:
# CAPM Expected Returns:
# ============================================================
# Utility Stock............. β=0.6  →  E(R)=7.90%
# Consumer Staple........... β=0.8  →  E(R)=9.20%
# S&P 500 ETF............... β=1.0  →  E(R)=10.50%
# Tech Stock................ β=1.4  →  E(R)=13.10%
# Small Cap Growth.......... β=1.8  →  E(R)=15.70%
# ============================================================
\`\`\`

**Key observations:**
- β = 1.0 (market): Expected return = 10.5% (Rf + MRP)
- β < 1.0: Less volatile than market → Lower return
- β > 1.0: More volatile than market → Higher return

---

## Understanding Beta (β)

### Definition

**Beta measures systematic risk:**

\`\`\`
β = Covariance(Ri, Rm) / Variance(Rm)

Or from regression: Ri = α + β × Rm + ε
\`\`\`

### Interpretation

| Beta | Meaning | Example |
|------|---------|---------|
| β = 0 | No market risk | Cash, T-bills |
| β = 0.5 | Half as volatile as market | Utilities, consumer staples |
| β = 1.0 | Same as market | S&P 500 index fund |
| β = 1.5 | 50% more volatile | Technology stocks |
| β = 2.0 | Twice as volatile | Leveraged ETFs, volatile small caps |
| β < 0 | Moves opposite to market | Gold (sometimes) |

### Calculating Beta from Historical Data

\`\`\`python
"""
Calculate Beta from Returns Data
"""

import yfinance as yf
from scipy import stats

def calculate_beta(
    ticker: str,
    market_ticker: str = '^GSPC',  # S&P 500
    period: str = '5y',
    frequency: str = 'monthly'
) -> dict:
    """
    Calculate beta from historical price data.
    
    Args:
        ticker: Stock ticker symbol
        market_ticker: Market index ticker (default S&P 500)
        period: Historical period ('1y', '3y', '5y')
        frequency: Return frequency ('daily', 'weekly', 'monthly')
    
    Returns:
        Dictionary with beta and statistics
    """
    # Download data
    stock = yf.download (ticker, period=period, progress=False)
    market = yf.download (market_ticker, period=period, progress=False)
    
    # Align dates
    combined = pd.DataFrame({
        'stock': stock['Adj Close'],
        'market': market['Adj Close']
    }).dropna()
    
    # Calculate returns based on frequency
    if frequency == 'daily':
        returns = combined.pct_change().dropna()
    elif frequency == 'weekly':
        returns = combined.resample('W').last().pct_change().dropna()
    elif frequency == 'monthly':
        returns = combined.resample('M').last().pct_change().dropna()
    else:
        raise ValueError("Frequency must be 'daily', 'weekly', or 'monthly'")
    
    # Calculate beta using regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        returns['market'],
        returns['stock']
    )
    
    # Beta is the slope
    beta = slope
    
    # Calculate using covariance formula (should match)
    cov = returns['stock'].cov (returns['market'])
    var = returns['market'].var()
    beta_cov = cov / var
    
    # Additional statistics
    r_squared = r_value ** 2
    correlation = returns['stock'].corr (returns['market'])
    
    # Annualize alpha (intercept)
    if frequency == 'monthly':
        alpha_annual = intercept * 12
    elif frequency == 'weekly':
        alpha_annual = intercept * 52
    elif frequency == 'daily':
        alpha_annual = intercept * 252
    
    return {
        'beta': beta,
        'beta_cov': beta_cov,  # Should match beta
        'alpha': intercept,
        'alpha_annual': alpha_annual,
        'r_squared': r_squared,
        'correlation': correlation,
        'std_error': std_err,
        'observations': len (returns)
    }


# Example: Calculate Tesla\'s beta
tesla_beta_analysis = calculate_beta('TSLA', period='5y', frequency='monthly')

print("Tesla Beta Analysis (5-year monthly returns):")
print("=" * 60)
print(f"Beta: {tesla_beta_analysis['beta']:.3f}")
print(f"Alpha (annual): {tesla_beta_analysis['alpha_annual']:.2%}")
print(f"R-squared: {tesla_beta_analysis['r_squared']:.3f}")
print(f"Correlation: {tesla_beta_analysis['correlation']:.3f}")
print(f"Standard error: {tesla_beta_analysis['std_error']:.4f}")
print(f"Observations: {tesla_beta_analysis['observations']}")
print("=" * 60)

# Interpretation
beta = tesla_beta_analysis['beta']
if beta > 1.5:
    risk_level = "Very High Risk"
elif beta > 1.2:
    risk_level = "High Risk"
elif beta > 0.8:
    risk_level = "Moderate Risk"
else:
    risk_level = "Low Risk"

print(f"\\nRisk Classification: {risk_level}")
print(f"Interpretation: Tesla is {beta:.1f}x as volatile as S&P 500")

# Expected return using CAPM
rf = 0.04
mrp = 0.065
expected_return = capm (rf, beta, mrp)
print(f"\\nCAPM Expected Return: {expected_return:.2%}")
\`\`\`

### Visualizing Beta

\`\`\`python
"""
Visualize Beta through Regression Plot
"""

import matplotlib.pyplot as plt

def plot_beta_regression(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    ticker: str
) -> None:
    """
    Create scatter plot with regression line showing beta.
    """
    # Calculate beta
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        market_returns,
        stock_returns
    )
    
    # Create plot
    fig, ax = plt.subplots (figsize=(12, 8))
    
    # Scatter plot
    ax.scatter(
        market_returns * 100,
        stock_returns * 100,
        alpha=0.5,
        s=50,
        label='Monthly returns'
    )
    
    # Regression line
    x_line = np.linspace (market_returns.min(), market_returns.max(), 100)
    y_line = intercept + slope * x_line
    ax.plot(
        x_line * 100,
        y_line * 100,
        'r-',
        linewidth=2,
        label=f'Beta = {slope:.2f}'
    )
    
    # Reference line (Beta = 1)
    ax.plot(
        [market_returns.min() * 100, market_returns.max() * 100],
        [market_returns.min() * 100, market_returns.max() * 100],
        'k--',
        alpha=0.3,
        label='Beta = 1.0 (Market)'
    )
    
    # Labels and formatting
    ax.set_xlabel('Market Return (%)', fontsize=12)
    ax.set_ylabel (f'{ticker} Return (%)', fontsize=12)
    ax.set_title(
        f'{ticker} vs Market\\nBeta = {slope:.2f}, R² = {r_value**2:.3f}',
        fontsize=14,
        fontweight='bold'
    )
    ax.grid (alpha=0.3)
    ax.legend (fontsize=10)
    
    # Add interpretation text
    if slope > 1.0:
        interpretation = f"For every 1% market move, {ticker} moves {slope:.1f}% (more volatile)"
    elif slope < 1.0:
        interpretation = f"For every 1% market move, {ticker} moves {slope:.1f}% (less volatile)"
    else:
        interpretation = f"{ticker} moves in line with the market"
    
    ax.text(
        0.05, 0.95,
        interpretation,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict (boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    plt.savefig (f'{ticker}_beta.png', dpi=300, bbox_inches='tight')
    print(f"Chart saved: {ticker}_beta.png")

# Example usage (with actual data)
# plot_beta_regression (returns['stock'], returns['market'], 'TSLA')
\`\`\`

---

## Levered vs Unlevered Beta

### Why It Matters

**Financial leverage amplifies equity risk**

- **Unlevered beta (βU)**: Business risk only
- **Levered beta (βL)**: Business risk + financial risk

### The Formulas

**Unlevering beta** (remove leverage):
\`\`\`
βU = βL / [1 + (1 - T) × (D/E)]
\`\`\`

**Relevering beta** (add leverage):
\`\`\`
βL = βU × [1 + (1 - T) × (D/E)]
\`\`\`

### Python Implementation

\`\`\`python
"""
Levered and Unlevered Beta Calculations
"""

def unlever_beta(
    levered_beta: float,
    debt_to_equity: float,
    tax_rate: float
) -> float:
    """
    Remove effect of financial leverage from beta.
    
    Args:
        levered_beta: Observed beta (includes leverage effect)
        debt_to_equity: Debt-to-equity ratio (market values)
        tax_rate: Corporate tax rate
    
    Returns:
        Unlevered beta (asset beta)
        
    Example:
        >>> bu = unlever_beta(1.5, 0.5, 0.25)
        >>> print(f"Unlevered beta: {bu:.3f}")
        Unlevered beta: 1.200
    """
    return levered_beta / (1 + (1 - tax_rate) * debt_to_equity)


def relever_beta(
    unlevered_beta: float,
    debt_to_equity: float,
    tax_rate: float
) -> float:
    """
    Add effect of financial leverage to beta.
    
    Args:
        unlevered_beta: Asset beta (no leverage)
        debt_to_equity: Target debt-to-equity ratio
        tax_rate: Corporate tax rate
    
    Returns:
        Levered beta (equity beta)
    """
    return unlevered_beta * (1 + (1 - tax_rate) * debt_to_equity)


# Example: Comparing companies with different leverage
print("Impact of Leverage on Beta:")
print("=" * 80)

# Asset (unlevered) beta for industry
industry_asset_beta = 1.0
tax_rate = 0.25

leverage_scenarios = [
    ('No Debt', 0.0),
    ('Conservative', 0.3),
    ('Moderate', 0.5),
    ('Aggressive', 1.0),
    ('Highly Leveraged', 2.0),
]

print(f"Industry Asset Beta (unlevered): {industry_asset_beta:.2f}")
print(f"Tax Rate: {tax_rate:.0%}")
print()
print(f"{'Capital Structure':<20} {'D/E Ratio':<12} {'Levered Beta':<15} {'Risk Level'}")
print("-" * 80)

for scenario_name, de_ratio in leverage_scenarios:
    levered_beta = relever_beta (industry_asset_beta, de_ratio, tax_rate)
    
    if levered_beta < 0.8:
        risk = "Low"
    elif levered_beta < 1.2:
        risk = "Moderate"
    elif levered_beta < 1.5:
        risk = "High"
    else:
        risk = "Very High"
    
    print(f"{scenario_name:<20} {de_ratio:<12.1f} {levered_beta:<15.2f} {risk}")

print("=" * 80)

# Output:
# Impact of Leverage on Beta:
# ================================================================================
# Industry Asset Beta (unlevered): 1.00
# Tax Rate: 25%
#
# Capital Structure    D/E Ratio    Levered Beta    Risk Level
# --------------------------------------------------------------------------------
# No Debt              0.0          1.00            Moderate
# Conservative         0.3          1.23            High
# Moderate             0.5          1.38            High
# Aggressive           1.0          1.75            Very High
# Highly Leveraged     2.0          2.50            Very High
# ================================================================================
\`\`\`

**Key insight:** A company with D/E = 2.0 has equity beta of 2.50, even though its business (asset) beta is only 1.00!

### Practical Application: Comparable Company Analysis

\`\`\`python
"""
Estimate Beta Using Comparable Companies
"""

def estimate_beta_from_comps(
    comp_betas: List[float],
    comp_debt_equity: List[float],
    comp_tax_rates: List[float],
    target_debt_equity: float,
    target_tax_rate: float
) -> dict:
    """
    Estimate target company beta using comparable companies.
    
    Process:
    1. Unlever each comp's beta
    2. Calculate median unlevered beta (industry asset beta)
    3. Relever at target's capital structure
    
    Args:
        comp_betas: List of comparable company betas
        comp_debt_equity: List of D/E ratios for comps
        comp_tax_rates: List of tax rates for comps
        target_debt_equity: Target company's D/E ratio
        target_tax_rate: Target company's tax rate
    
    Returns:
        Dictionary with analysis results
    """
    # Step 1: Unlever each comp
    unlevered_betas = []
    for i in range (len (comp_betas)):
        bu = unlever_beta (comp_betas[i], comp_debt_equity[i], comp_tax_rates[i])
        unlevered_betas.append (bu)
    
    # Step 2: Calculate industry asset beta (median is more robust than mean)
    industry_asset_beta = np.median (unlevered_betas)
    
    # Step 3: Relever for target
    target_beta = relever_beta (industry_asset_beta, target_debt_equity, target_tax_rate)
    
    return {
        'comparable_betas': comp_betas,
        'unlevered_betas': unlevered_betas,
        'industry_asset_beta': industry_asset_beta,
        'target_levered_beta': target_beta,
        'target_debt_equity': target_debt_equity
    }


# Example: Estimate beta for a private company
comps = pd.DataFrame({
    'Company': ['Comp A', 'Comp B', 'Comp C', 'Comp D', 'Comp E'],
    'Beta': [1.1, 1.3, 0.9, 1.2, 1.4],
    'D/E': [0.3, 0.5, 0.2, 0.6, 0.4],
    'Tax Rate': [0.25, 0.25, 0.20, 0.25, 0.25]
})

print("Comparable Company Beta Analysis:")
print("=" * 80)
print(comps.to_string (index=False))
print()

# Estimate for target with D/E = 0.8
analysis = estimate_beta_from_comps(
    comp_betas=comps['Beta'].tolist(),
    comp_debt_equity=comps['D/E'].tolist(),
    comp_tax_rates=comps['Tax Rate'].tolist(),
    target_debt_equity=0.8,
    target_tax_rate=0.25
)

print("Analysis Results:")
print("-" * 80)
print(f"Industry asset beta (median unlevered): {analysis['industry_asset_beta']:.3f}")
print(f"Target company D/E ratio: {analysis['target_debt_equity']:.1f}")
print(f"Estimated target beta: {analysis['target_levered_beta']:.3f}")
print("=" * 80)

# Calculate cost of equity
rf = 0.04
mrp = 0.065
cost_of_equity = capm (rf, analysis['target_levered_beta'], mrp)
print(f"\\nImplied cost of equity: {cost_of_equity:.2%}")
\`\`\`

---

## Adjusted Beta

### The Problem with Historical Beta

**Past beta may not equal future beta**

Companies and markets change over time.

### Blume Adjustment

**Betas tend to revert to 1.0 over time**

Adjusted Beta = (2/3) × Historical Beta + (1/3) × 1.0

\`\`\`python
def adjusted_beta(
    historical_beta: float,
    market_beta: float = 1.0,
    adjustment_weight: float = 0.33
) -> float:
    """
    Calculate adjusted beta using Blume adjustment.
    
    Recognizes that betas tend to revert toward market beta.
    
    Args:
        historical_beta: Calculated from historical data
        market_beta: Market beta (usually 1.0)
        adjustment_weight: Weight on market beta (default 1/3)
    
    Returns:
        Adjusted beta
    """
    return (1 - adjustment_weight) * historical_beta + adjustment_weight * market_beta


# Example: Bloomberg/FactSet use adjusted betas
historical_betas = [0.6, 1.0, 1.5, 2.0]

print("Historical vs Adjusted Beta:")
print("=" * 60)
print(f"{'Historical Beta':<20} {'Adjusted Beta':<20} {'Change'}")
print("-" * 60)

for hist_beta in historical_betas:
    adj_beta = adjusted_beta (hist_beta)
    change = adj_beta - hist_beta
    print(f"{hist_beta:<20.2f} {adj_beta:<20.2f} {change:+.2f}")

print("=" * 60)

# Output:
# Historical vs Adjusted Beta:
# ============================================================
# Historical Beta      Adjusted Beta        Change
# ------------------------------------------------------------
# 0.60                 0.73                 +0.13
# 1.00                 1.00                 +0.00
# 1.50                 1.33                 -0.17
# 2.00                 1.67                 -0.33
# ============================================================
\`\`\`

**Key observation:** 
- Low betas (< 1.0) adjust upward
- High betas (> 1.0) adjust downward
- Beta = 1.0 stays at 1.0

---

## Bottom-Up Beta

### When to Use

**For companies with:**
- No trading history (private companies, startups, spinoffs)
- Recent major changes (M&A, restructuring)
- Thin trading (illiquid stocks)

### Method

1. Identify comparable companies in same business
2. Calculate average unlevered beta for industry
3. Relever at target company's capital structure

\`\`\`python
"""
Bottom-Up Beta Estimation
"""

def bottom_up_beta(
    industry_businesses: dict,
    business_weights: dict,
    target_debt_equity: float,
    target_tax_rate: float
) -> float:
    """
    Calculate bottom-up beta for multi-business company.
    
    Args:
        industry_businesses: Dict of {business: industry_asset_beta}
        business_weights: Dict of {business: revenue_weight}
        target_debt_equity: Company\'s D/E ratio
        target_tax_rate: Company's tax rate
    
    Returns:
        Bottom-up levered beta
        
    Example:
        >>> industries = {'Software': 1.2, 'Hardware': 1.0, 'Services': 0.8}
        >>> weights = {'Software': 0.5, 'Hardware': 0.3, 'Services': 0.2}
        >>> beta = bottom_up_beta (industries, weights, 0.3, 0.25)
    """
    # Step 1: Calculate weighted average unlevered beta
    weighted_asset_beta = sum([
        industry_businesses[business] * business_weights[business]
        for business in industry_businesses.keys()
    ])
    
    # Step 2: Relever at company's capital structure
    levered_beta = relever_beta (weighted_asset_beta, target_debt_equity, target_tax_rate)
    
    return levered_beta


# Example: Conglomerate with multiple businesses
print("Bottom-Up Beta for Diversified Company:")
print("=" * 80)

# Define businesses and their industry betas
industries = {
    'Technology': 1.3,
    'Healthcare': 0.9,
    'Consumer Goods': 0.8,
    'Financial Services': 1.1
}

# Company\'s revenue breakdown
weights = {
    'Technology': 0.40,
    'Healthcare': 0.25,
    'Consumer Goods': 0.20,
    'Financial Services': 0.15
}

# Company\'s capital structure
de_ratio = 0.4
tax_rate = 0.25

# Calculate
beta_bottomup = bottom_up_beta (industries, weights, de_ratio, tax_rate)

print("Business Segment Analysis:")
print("-" * 80)
for business in industries.keys():
    print(f"{business:<25} Asset Beta: {industries[business]:.2f}  "
          f"Weight: {weights[business]:.0%}")

print("-" * 80)
weighted_asset_beta = sum([industries[b] * weights[b] for b in industries.keys()])
print(f"Weighted average asset beta: {weighted_asset_beta:.3f}")
print(f"Company D/E ratio: {de_ratio:.1f}")
print(f"\\nBottom-up levered beta: {beta_bottomup:.3f}")
print("=" * 80)

# Compare to single-industry company
single_industry_beta = relever_beta (industries['Technology'], de_ratio, tax_rate)
print(f"\\nComparison:")
print(f"If pure Technology company: Beta = {single_industry_beta:.3f}")
print(f"Actual (diversified): Beta = {beta_bottomup:.3f}")
print(f"Diversification reduces beta by: {single_industry_beta - beta_bottomup:.3f}")
\`\`\`

---

## CAPM Assumptions and Limitations

### CAPM Assumptions

1. **Investors are rational** and risk-averse
2. **Perfect capital markets**: No taxes, transaction costs, or restrictions
3. **Investors can borrow/lend** at risk-free rate
4. **Single-period horizon**
5. **All investors have same information** (homogeneous expectations)
6. **Investors hold market portfolio** of all risky assets
7. **Returns are normally distributed**

**Reality:** Most assumptions are violated!

### Limitations

**Problem 1: Beta Instability**

\`\`\`python
"""
Demonstrate Beta Instability Over Time
"""

def rolling_beta(
    stock_ticker: str,
    market_ticker: str = '^GSPC',
    window_months: int = 60,
    period: str = '10y'
) -> pd.DataFrame:
    """
    Calculate rolling beta over time.
    
    Shows how beta changes with different time periods.
    """
    # Download data
    stock = yf.download (stock_ticker, period=period, progress=False)
    market = yf.download (market_ticker, period=period, progress=False)
    
    # Monthly returns
    combined = pd.DataFrame({
        'stock': stock['Adj Close'],
        'market': market['Adj Close']
    }).resample('M').last().pct_change().dropna()
    
    # Calculate rolling beta
    betas = []
    dates = []
    
    for i in range (window_months, len (combined)):
        window_data = combined.iloc[i-window_months:i]
        beta = window_data['stock'].cov (window_data['market']) / window_data['market'].var()
        betas.append (beta)
        dates.append (combined.index[i])
    
    return pd.DataFrame({
        'Date': dates,
        'Beta': betas
    })


# Example usage
# rolling_betas = rolling_beta('AAPL', window_months=60)
# print(f"Beta range: {rolling_betas['Beta'].min():.2f} to {rolling_betas['Beta'].max():.2f}")
\`\`\`

**Problem 2: Single-Factor Model**

CAPM only considers market risk. In reality, other factors matter:
- Size (small vs large cap)
- Value vs growth
- Momentum
- Quality
- Volatility

**Problem 3: Market Risk Premium Uncertainty**

Historical MRP ≈ 6-7%, but varies by period:
- 1926-2023: ~8%
- 1950-2023: ~7%
- Last 20 years: ~9%

**Problem 4: Which Risk-Free Rate?**

- 3-month T-bills? (more accurate "risk-free")
- 10-year bonds? (matches project duration)

---

## Alternatives to CAPM

### 1. Fama-French Three-Factor Model

\`\`\`
E(Ri) = Rf + βM(Rm - Rf) + βS(SMB) + βV(HML)

Where:
- SMB = Small Minus Big (size premium)
- HML = High Minus Low (value premium)
\`\`\`

### 2. Arbitrage Pricing Theory (APT)

Multiple factors (GDP growth, inflation, etc.)

### 3. Build-Up Method

For private companies:
\`\`\`
Re = Rf + ERP + Size Premium + Company-Specific Risk
\`\`\`

\`\`\`python
def buildup_cost_of_equity(
    risk_free_rate: float,
    equity_risk_premium: float,
    size_premium: float = 0.03,
    company_specific_risk: float = 0.02
) -> float:
    """
    Calculate cost of equity using build-up method.
    
    Common for private companies where CAPM beta unavailable.
    
    Args:
        risk_free_rate: Risk-free rate
        equity_risk_premium: General equity risk premium (vs risk-free)
        size_premium: Small company premium (2-5%)
        company_specific_risk: Company-specific risks (2-5%)
    
    Returns:
        Cost of equity
    """
    return risk_free_rate + equity_risk_premium + size_premium + company_specific_risk


# Example: Private company cost of equity
rf = 0.04
erp = 0.065
size = 0.03  # Small company
specific = 0.025  # Moderate additional risk

cost_equity_buildup = buildup_cost_of_equity (rf, erp, size, specific)

print("Build-Up Method:")
print(f"Risk-free rate:          {rf:.2%}")
print(f"Equity risk premium:     {erp:.2%}")
print(f"Size premium:            {size:.2%}")
print(f"Company-specific risk:   {specific:.2%}")
print(f"Total cost of equity:    {cost_equity_buildup:.2%}")

# Output:
# Build-Up Method:
# Risk-free rate:          4.00%
# Equity risk premium:     6.50%
# Size premium:            3.00%
# Company-specific risk:   2.50%
# Total cost of equity:    16.00%
\`\`\`

---

## Practical Applications

### Application 1: Project Evaluation

Different projects → Different betas → Different required returns

\`\`\`python
"""
Risk-Adjusted Project Evaluation
"""

def project_wacc_with_beta(
    project_beta: float,
    company_wacc: float,
    company_beta: float,
    risk_free_rate: float,
    market_risk_premium: float,
    debt_ratio: float,
    cost_of_debt_aftertax: float
) -> float:
    """
    Calculate project-specific WACC using project beta.
    
    Process:
    1. Calculate project cost of equity using project beta
    2. Calculate project WACC
    """
    # Project cost of equity
    project_cost_equity = capm (risk_free_rate, project_beta, market_risk_premium)
    
    # Project WACC (assuming same capital structure)
    equity_weight = 1 - debt_ratio
    project_wacc = equity_weight * project_cost_equity + debt_ratio * cost_of_debt_aftertax
    
    return project_wacc


# Example: Three projects with different risk levels
company_beta = 1.0
company_wacc = 0.10
rf = 0.04
mrp = 0.065
debt_ratio = 0.30
rd_aftertax = 0.045

projects = {
    'Cost Reduction (Low Risk)': 0.6,
    'Product Line Extension (Average)': 1.0,
    'International Expansion (High Risk)': 1.5
}

print("Project-Specific Discount Rates:")
print("=" * 80)
for project_name, project_beta in projects.items():
    proj_wacc = project_wacc_with_beta(
        project_beta,
        company_wacc,
        company_beta,
        rf,
        mrp,
        debt_ratio,
        rd_aftertax
    )
    print(f"{project_name:<40} β={project_beta:.1f}  →  WACC={proj_wacc:.2%}")

print("=" * 80)
\`\`\`

---

## Key Takeaways

### CAPM Formula
\`\`\`
E(Ri) = Rf + βi × [E(Rm) - Rf]
\`\`\`

### Beta Interpretation
- **β = 0**: No market risk
- **β < 1**: Less risky than market
- **β = 1**: Same as market
- **β > 1**: Riskier than market

### Levering/Unlevering
- **Unlever** to remove financial risk
- **Relever** to add financial risk
- Use for comparable company analysis

### Practical Tips

✓ **Use 5-year monthly returns** for beta calculation  
✓ **Consider adjusted beta** (regression to 1.0)  
✓ **Use bottom-up beta** for private companies  
✓ **Different projects need different betas**  
✓ **CAPM has limitations** but remains industry standard

---

## Next Section

You now understand CAPM and beta! Next topics:
- **Capital Structure** (Section 5): Optimal debt-equity mix
- **Valuation** (Section 6): Applying CAPM to company valuation

**Next Section**: [Capital Structure & Leverage](./capital-structure-leverage) →
`,
};
