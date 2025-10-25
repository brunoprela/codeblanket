export const costOfCapitalWacc = {
  title: 'Cost of Capital (WACC)',
  id: 'cost-of-capital-wacc',
  content: `
# Cost of Capital (WACC)

## Introduction

**What discount rate should you use for NPV calculations?**

This is one of the most important questions in corporate finance. Use too high a rate, and you reject good projects. Use too low a rate, and you accept value-destroying projects.

The answer: **Weighted Average Cost of Capital (WACC)**

**WACC represents:**
- The minimum return investors expect from the company
- The average rate the company pays to finance its assets
- The hurdle rate for new investments
- The discount rate for DCF valuation

By the end of this section, you'll be able to:
- Calculate WACC for any company
- Understand each component (cost of equity, cost of debt)
- Apply WACC correctly in capital budgeting
- Adjust WACC for different risk levels
- Build a WACC calculator programmatically

### Why WACC Matters

**Example**: Tesla is considering a new Gigafactory:
- **If WACC = 8%**: NPV = $500M → **Build it!**
- **If WACC = 12%**: NPV = -$100M → **Don't build!**

**A 4% difference in WACC completely changes the decision!**

Getting WACC wrong is one of the costliest mistakes in finance.

---

## The WACC Formula

### Basic Formula

\`\`\`
WACC = (E/V) × Re + (D/V) × Rd × (1 - Tc)

Where:
- E = Market value of equity
- D = Market value of debt
- V = E + D (total firm value)
- Re = Cost of equity
- Rd = Cost of debt
- Tc = Corporate tax rate
\`\`\`

### Why the Tax Shield?

**Debt interest is tax-deductible!**

If company pays 7% interest and tax rate is 25%:
- **After-tax cost** = 7% × (1 - 0.25) = **5.25%**

This makes debt cheaper than equity and creates the "tax shield."

### Intuition

WACC is the **weighted average** of what you pay to equity and debt investors:

\`\`\`python
"""
Simple WACC Calculation
"""

def calculate_wacc(
    market_value_equity: float,
    market_value_debt: float,
    cost_of_equity: float,
    cost_of_debt: float,
    tax_rate: float
) -> float:
    """
    Calculate Weighted Average Cost of Capital.
    
    Args:
        market_value_equity: Market cap of equity
        market_value_debt: Market value of debt
        cost_of_equity: Required return on equity (Re)
        cost_of_debt: Yield on debt (Rd)
        tax_rate: Corporate tax rate (Tc)
    
    Returns:
        WACC (as decimal)
        
    Example:
        >>> wacc = calculate_wacc(
        ...     market_value_equity=600_000_000,
        ...     market_value_debt=400_000_000,
        ...     cost_of_equity=0.12,
        ...     cost_of_debt=0.06,
        ...     tax_rate=0.25
        ... )
        >>> print(f"WACC: {wacc:.2%}")
        WACC: 9.00%
    """
    total_value = market_value_equity + market_value_debt
    
    equity_weight = market_value_equity / total_value
    debt_weight = market_value_debt / total_value
    
    after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)
    
    wacc = (equity_weight * cost_of_equity) + (debt_weight * after_tax_cost_of_debt)
    
    return wacc


# Example: Tech company
equity_value = 600_000_000  # $600M market cap
debt_value = 400_000_000    # $400M debt
cost_equity = 0.12          # 12% cost of equity
cost_debt = 0.06            # 6% cost of debt  
tax_rate = 0.25             # 25% tax rate

wacc = calculate_wacc (equity_value, debt_value, cost_equity, cost_debt, tax_rate)

print("WACC Calculation:")
print(f"Market value of equity: \${equity_value / 1e6:.0f}M")
print(f"Market value of debt: \${debt_value/1e6:.0f}M")
print(f"Total firm value: \${(equity_value + debt_value)/1e6:.0f}M")
print(f"\\nEquity weight: {equity_value/(equity_value+debt_value):.1%}")
print(f"Debt weight: {debt_value/(equity_value+debt_value):.1%}")
print(f"\\nCost of equity: {cost_equity:.1%}")
print(f"Cost of debt (pre-tax): {cost_debt:.1%}")
print(f"Cost of debt (after-tax): {cost_debt*(1-tax_rate):.2%}")
print(f"\\n✓ WACC: {wacc:.2%}")

# Output:
# WACC Calculation:
# Market value of equity: $600M
# Market value of debt: $400M
# Total firm value: $1000M
#
# Equity weight: 60.0 %
# Debt weight: 40.0 %
#
# Cost of equity: 12.0 %
# Cost of debt (pre - tax): 6.0 %
# Cost of debt (after - tax): 4.50 %
#
# ✓ WACC: 9.00 %
\`\`\`

**Interpretation**: This company must earn at least 9% on its investments to satisfy both equity and debt investors.

---

## Cost of Equity (Re)

### Methods to Calculate

**Three primary approaches:**

1. **Capital Asset Pricing Model (CAPM)** ← Most common
2. **Dividend Discount Model (DDM)**
3. **Bond Yield Plus Risk Premium**

### Method 1: CAPM (Most Common)

\`\`\`
Re = Rf + β × (Rm - Rf)

Where:
- Rf = Risk-free rate (typically 10-year Treasury yield)
- β (beta) = Stock\'s systematic risk
- Rm = Expected market return
- (Rm - Rf) = Market risk premium
\`\`\`

**Example calculation:**

\`\`\`python
"""
Cost of Equity using CAPM
"""

def cost_of_equity_capm(
    risk_free_rate: float,
    beta: float,
    market_risk_premium: float
) -> float:
    """
    Calculate cost of equity using CAPM.
    
    Args:
        risk_free_rate: Risk-free rate (e.g., 10-year Treasury)
        beta: Stock's beta (systematic risk)
        market_risk_premium: Expected market return - risk-free rate
    
    Returns:
        Cost of equity
        
    Example:
        >>> re = cost_of_equity_capm(0.04, 1.2, 0.065)
        >>> print(f"Cost of equity: {re:.2%}")
        Cost of equity: 11.80%
    """
    return risk_free_rate + beta * market_risk_premium


# Example: Tesla
rf = 0.04      # 4% Treasury yield
beta = 1.8     # Tesla's beta (high volatility)
mrp = 0.065    # 6.5% market risk premium

re_tesla = cost_of_equity_capm (rf, beta, mrp)

print("Cost of Equity (CAPM):")
print(f"Risk-free rate: {rf:.2%}")
print(f"Beta: {beta:.2f}")
print(f"Market risk premium: {mrp:.2%}")
print(f"\\nCost of equity: {re_tesla:.2%}")

# Interpretation
print(f"\\nInterpretation:")
print(f"Tesla is {beta:.1f}x more volatile than market")
print(f"Investors require {re_tesla:.1%} return to hold Tesla stock")

# Output:
# Cost of Equity (CAPM):
# Risk-free rate: 4.00%
# Beta: 1.80
# Market risk premium: 6.50%
#
# Cost of equity: 15.70%
#
# Interpretation:
# Tesla is 1.8x more volatile than market
# Investors require 15.7% return to hold Tesla stock
\`\`\`

### Choosing Inputs for CAPM

**1. Risk-Free Rate (Rf)**

Use 10-year Treasury yield (matches typical project duration):

\`\`\`python
import yfinance as yf

# Get current 10-year Treasury yield
treasury = yf.Ticker("^TNX")
current_yield = treasury.history (period="1d")['Close'].iloc[-1] / 100

print(f"Current 10-year Treasury yield: {current_yield:.2%}")
\`\`\`

**2. Beta (β)**

Download from Bloomberg, Yahoo Finance, or calculate yourself:

\`\`\`python
import pandas as pd
import numpy as np

def calculate_beta(
    stock_returns: pd.Series,
    market_returns: pd.Series
) -> dict:
    """
    Calculate stock beta from historical returns.
    
    Args:
        stock_returns: Stock daily/monthly returns
        market_returns: Market (S&P 500) returns for same period
    
    Returns:
        Dictionary with beta and related statistics
    """
    # Align returns
    combined = pd.DataFrame({
        'stock': stock_returns,
        'market': market_returns
    }).dropna()
    
    # Calculate beta (covariance / variance)
    covariance = combined['stock'].cov (combined['market'])
    market_variance = combined['market'].var()
    beta = covariance / market_variance
    
    # Calculate R-squared
    correlation = combined['stock'].corr (combined['market'])
    r_squared = correlation ** 2
    
    # Regression for alpha
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        combined['market'],
        combined['stock']
    )
    
    return {
        'beta': beta,
        'alpha': intercept,  # Jensen\'s alpha
        'r_squared': r_squared,
        'correlation': correlation,
        'std_error': std_err
    }


# Example: Calculate beta for a stock
# (Assume we have returns data)
# stock_returns = pd.Series([...])
# market_returns = pd.Series([...])
# beta_analysis = calculate_beta (stock_returns, market_returns)

# Simulate for demonstration
np.random.seed(42)
market_ret = np.random.normal(0.001, 0.02, 252)  # 252 trading days
stock_ret = 1.5 * market_ret + np.random.normal(0, 0.01, 252)  # Beta ≈ 1.5

beta_analysis = calculate_beta(
    pd.Series (stock_ret),
    pd.Series (market_ret)
)

print("Beta Analysis:")
print(f"Beta: {beta_analysis['beta']:.2f}")
print(f"Alpha: {beta_analysis['alpha']:.4f}")
print(f"R-squared: {beta_analysis['r_squared']:.2%}")
print(f"Correlation: {beta_analysis['correlation']:.2%}")
\`\`\`

**3. Market Risk Premium (Rm - Rf)**

Historical average: **~6-7%**

\`\`\`python
"""
Estimate market risk premium from historical data
"""

def estimate_market_risk_premium(
    years_of_history: int = 50
) -> float:
    """
    Estimate MRP from historical S&P 500 returns.
    
    Typical ranges:
    - Short-term (5-10 years): Volatile, 2-10%
    - Long-term (50+ years): Stable, 6-7%
    """
    # Download S&P 500 and Treasury data
    sp500 = yf.Ticker("^GSPC")
    hist = sp500.history (period=f"{years_of_history}y")
    
    # Calculate annualized return
    total_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) ** (1/years_of_history) - 1
    
    # Subtract average risk-free rate (assume 4%)
    risk_free = 0.04
    mrp = total_return - risk_free
    
    return mrp

# For calculation purposes, use standard estimate
MRP_STANDARD = 0.065  # 6.5%
print(f"Standard market risk premium: {MRP_STANDARD:.1%}")
\`\`\`

### Method 2: Dividend Discount Model

For dividend-paying stocks:

\`\`\`
Re = (D1 / P0) + g

Where:
- D1 = Next year's expected dividend
- P0 = Current stock price
- g = Dividend growth rate
\`\`\`

\`\`\`python
def cost_of_equity_ddm(
    next_dividend: float,
    current_price: float,
    growth_rate: float
) -> float:
    """
    Cost of equity using Dividend Discount Model.
    
    Best for mature, dividend-paying companies.
    """
    return (next_dividend / current_price) + growth_rate


# Example: Utility company
next_div = 2.50      # $2.50 per share
price = 50.00        # $50 stock price
growth = 0.03        # 3% annual growth

re_ddm = cost_of_equity_ddm (next_div, price, growth)

print(f"Cost of Equity (DDM):")
print(f"Next dividend: \${next_div}")
print(f"Current price: \${price}")
print(f"Growth rate: {growth:.1%}")
print(f"Cost of equity: {re_ddm:.2%}")

# Output:
# Cost of Equity (DDM):
# Next dividend: $2.5
# Current price: $50
# Growth rate: 3.0%
# Cost of equity: 8.00%
\`\`\`

**When to use DDM:**
- ✓ Stable, mature companies
- ✓ Regular dividend payments
- ✗ Growth companies with no dividends (use CAPM)
- ✗ Volatile dividend patterns

### Method 3: Bond Yield Plus Risk Premium

\`\`\`
Re = Rd + Risk Premium

Where Risk Premium ≈ 3-5% for typical companies
\`\`\`

\`\`\`python
def cost_of_equity_bond_yield_plus(
    company_bond_yield: float,
    equity_risk_premium: float = 0.04
) -> float:
    """
    Cost of equity = Company\'s bond yield + risk premium.
    
    Rule of thumb: Equity holders require 3-5% more than debt holders.
    """
    return company_bond_yield + equity_risk_premium


# Example
bond_yield = 0.06    # Company's bonds yield 6%
risk_prem = 0.04     # 4% equity risk premium

re_byp = cost_of_equity_bond_yield_plus (bond_yield, risk_prem)

print(f"Cost of equity (Bond Yield Plus): {re_byp:.1%}")
# Output: Cost of equity (Bond Yield Plus): 10.0%
\`\`\`

---

## Cost of Debt (Rd)

### Definition

**Cost of debt = Yield to Maturity (YTM) on company's bonds**

Or: The interest rate the company would pay on new debt today

### Calculating Cost of Debt

**Method 1: From Bond Prices**

If bonds are publicly traded:

\`\`\`python
def yield_to_maturity(
    bond_price: float,
    face_value: float,
    coupon_rate: float,
    years_to_maturity: int,
    payments_per_year: int = 2
) -> float:
    """
    Calculate YTM (cost of debt) from bond price.
    
    Uses iterative method (Newton-Raphson).
    """
    from scipy.optimize import newton
    
    coupon_payment = (coupon_rate * face_value) / payments_per_year
    periods = years_to_maturity * payments_per_year
    
    def bond_price_func (ytm):
        """Calculate bond price given YTM"""
        pv_coupons = sum([
            coupon_payment / (1 + ytm/payments_per_year)**t
            for t in range(1, periods + 1)
        ])
        pv_face = face_value / (1 + ytm/payments_per_year)**periods
        return pv_coupons + pv_face
    
    def price_diff (ytm):
        return bond_price_func (ytm) - bond_price
    
    # Solve for YTM
    ytm = newton (price_diff, x0=0.05, maxiter=100)
    
    return ytm


# Example: Company bond
bond_price = 950      # Trading at $950
face_value = 1000     # $1,000 face value
coupon_rate = 0.06    # 6% annual coupon
years = 10            # 10 years to maturity

ytm = yield_to_maturity (bond_price, face_value, coupon_rate, years)

print("Cost of Debt Calculation:")
print(f"Bond price: \${bond_price}")
print(f"Face value: \${face_value}")
print(f"Coupon rate: {coupon_rate:.1%}")
print(f"Years to maturity: {years}")
print(f"\\nYTM (Cost of debt): {ytm:.2%}")

# Output:
# Cost of Debt Calculation:
# Bond price: $950
# Face value: $1000
# Coupon rate: 6.0%
# Years to maturity: 10
#
# YTM (Cost of debt): 6.60%
\`\`\`

**Method 2: From Credit Rating**

If no bonds are traded, use typical yield for the company's credit rating:

\`\`\`python
# Typical corporate bond yields by rating (as of example date)
CREDIT_SPREAD_TABLE = {
    'AAA': 0.015,  # +1.5% over Treasuries
    'AA': 0.020,   # +2.0%
    'A': 0.025,    # +2.5%
    'BBB': 0.035,  # +3.5%
    'BB': 0.055,   # +5.5%
    'B': 0.080,    # +8.0%
    'CCC': 0.120,  # +12.0%
}

def cost_of_debt_from_rating(
    credit_rating: str,
    risk_free_rate: float = 0.04
) -> float:
    """
    Estimate cost of debt from credit rating.
    
    Args:
        credit_rating: S&P rating (AAA, AA, A, BBB, BB, B, CCC)
        risk_free_rate: 10-year Treasury yield
    
    Returns:
        Estimated cost of debt
    """
    spread = CREDIT_SPREAD_TABLE.get (credit_rating.upper(), 0.05)
    return risk_free_rate + spread


# Example
rating = 'BBB'
rf = 0.04

cost_debt = cost_of_debt_from_rating (rating, rf)

print(f"Credit rating: {rating}")
print(f"Risk-free rate: {rf:.1%}")
print(f"Credit spread: {CREDIT_SPREAD_TABLE[rating]:.1%}")
print(f"Cost of debt: {cost_debt:.2%}")

# Output:
# Credit rating: BBB
# Risk-free rate: 4.0%
# Credit spread: 3.5%
# Cost of debt: 7.50%
\`\`\`

**Method 3: From Interest Expense**

\`\`\`
Cost of Debt ≈ Interest Expense / Total Debt
\`\`\`

\`\`\`python
def cost_of_debt_from_financials(
    interest_expense: float,
    total_debt: float
) -> float:
    """
    Estimate cost of debt from financial statements.
    
    Use average debt if debt changed significantly during year.
    """
    return interest_expense / total_debt


# Example from 10-K
interest_exp = 50_000_000   # $50M interest expense
total_debt = 800_000_000    # $800M total debt

rd = cost_of_debt_from_financials (interest_exp, total_debt)

print(f"Interest expense: \${interest_exp / 1e6:.0f}M")
print(f"Total debt: \${total_debt/1e6:.0f}M")
print(f"Cost of debt: {rd:.2%}")

# Output:
# Interest expense: $50M
# Total debt: $800M
# Cost of debt: 6.25 %
\`\`\`

---

## Market Value vs Book Value

**Critical: Always use MARKET values, not book values!**

### Market Value of Equity

Easy: **Market Capitalization**

\`\`\`
Market Value of Equity = Share Price × Shares Outstanding
\`\`\`

\`\`\`python
def market_value_equity(
    share_price: float,
    shares_outstanding: float
) -> float:
    """Calculate market value of equity."""
    return share_price * shares_outstanding


# Example: Apple
price = 180.00               # $180 per share
shares = 15_500_000_000      # 15.5 billion shares

mv_equity = market_value_equity (price, shares)

print(f"Share price: \${price:.2f}")
print(f"Shares outstanding: {shares/1e9:.1f}B")
print(f"Market cap: \${mv_equity/1e12:.2f}T")

# Output:
# Share price: $180.00
# Shares outstanding: 15.5B
# Market cap: $2.79T
\`\`\`

### Market Value of Debt

**More tricky**: Debt doesn't always trade publicly

**Option 1**: If bonds trade, use market prices

\`\`\`python
def market_value_debt_from_bonds(
    face_value: float,
    market_price_pct: float
) -> float:
    """
    Calculate market value from bond prices.
    
    Args:
        face_value: Face value of debt
        market_price_pct: Market price as % of face value
    
    Returns:
        Market value of debt
    """
    return face_value * (market_price_pct / 100)


# Example
face = 500_000_000    # $500M face value
price_pct = 95        # Trading at 95% of face value

mv_debt = market_value_debt_from_bonds (face, price_pct)

print(f"Face value: \${face / 1e6:.0f}M")
print(f"Market price: {price_pct}% of face")
print(f"Market value: \${mv_debt/1e6:.0f}M")

# Output:
# Face value: $500M
# Market price: 95 % of face
# Market value: $475M
\`\`\`

**Option 2**: Approximate market value

If debt is not publicly traded:
- **Investment-grade companies**: Market value ≈ Book value
- **High-yield / distressed**: Market value < Book value (estimate from credit spread)

\`\`\`python
def approximate_market_value_debt(
    book_value_debt: float,
    credit_rating: str
) -> float:
    """
    Approximate market value of debt.
    
    For investment grade: MV ≈ BV
    For high yield: Adjust for distress
    """
    adjustment_factors = {
        'AAA': 1.00,
        'AA': 1.00,
        'A': 0.98,
        'BBB': 0.95,
        'BB': 0.85,
        'B': 0.70,
        'CCC': 0.50,
    }
    
    factor = adjustment_factors.get (credit_rating.upper(), 0.90)
    return book_value_debt * factor


# Example
book_debt = 1_000_000_000
rating = 'BBB'

mv_debt_approx = approximate_market_value_debt (book_debt, rating)

print(f"Book value of debt: \${book_debt / 1e9:.2f}B")
print(f"Credit rating: {rating}")
print(f"Approximate market value: \${mv_debt_approx/1e9:.2f}B")

# Output:
# Book value of debt: $1.00B
# Credit rating: BBB
# Approximate market value: $0.95B
\`\`\`

---

## Complete WACC Calculation

### Real-World Example: Apple

\`\`\`python
"""
Complete WACC Calculation for Apple (Example)
"""

import pandas as pd

class WACCCalculator:
    """
    Production-grade WACC calculator.
    """
    
    def __init__(
        self,
        company_name: str,
        market_cap: float,
        total_debt: float,
        risk_free_rate: float,
        beta: float,
        market_risk_premium: float,
        cost_of_debt: float,
        tax_rate: float
    ):
        self.company_name = company_name
        self.market_cap = market_cap
        self.total_debt = total_debt
        self.risk_free_rate = risk_free_rate
        self.beta = beta
        self.market_risk_premium = market_risk_premium
        self.cost_of_debt = cost_of_debt
        self.tax_rate = tax_rate
        
        # Calculate components
        self._calculate()
    
    def _calculate (self):
        """Perform all calculations."""
        # Firm value
        self.firm_value = self.market_cap + self.total_debt
        
        # Weights
        self.equity_weight = self.market_cap / self.firm_value
        self.debt_weight = self.total_debt / self.firm_value
        
        # Cost of equity (CAPM)
        self.cost_of_equity = (
            self.risk_free_rate + 
            self.beta * self.market_risk_premium
        )
        
        # After-tax cost of debt
        self.after_tax_cost_of_debt = self.cost_of_debt * (1 - self.tax_rate)
        
        # WACC
        self.wacc = (
            self.equity_weight * self.cost_of_equity +
            self.debt_weight * self.after_tax_cost_of_debt
        )
    
    def summary (self) -> pd.DataFrame:
        """Generate summary table."""
        data = {
            'Component': [
                'Market Value of Equity',
                'Market Value of Debt',
                'Total Firm Value',
                '',
                'Equity Weight',
                'Debt Weight',
                '',
                'Risk-Free Rate',
                'Beta',
                'Market Risk Premium',
                'Cost of Equity',
                '',
                'Cost of Debt (pre-tax)',
                'Tax Rate',
                'Cost of Debt (after-tax)',
                '',
                'WACC'
            ],
            'Value': [
                f"\${self.market_cap / 1e9:.2f}B",
                f"\${self.total_debt/1e9:.2f}B",
    f"\${self.firm_value/1e9:.2f}B",
        '',
        f"{self.equity_weight:.1%}",
            f"{self.debt_weight:.1%}",
                '',
                f"{self.risk_free_rate:.2%}",
                    f"{self.beta:.2f}",
                        f"{self.market_risk_premium:.2%}",
                            f"{self.cost_of_equity:.2%}",
                                '',
                                f"{self.cost_of_debt:.2%}",
                                    f"{self.tax_rate:.1%}",
                                        f"{self.after_tax_cost_of_debt:.2%}",
                                            '',
                                            f"{self.wacc:.2%}"
            ]
        }

return pd.DataFrame (data)
    
    def sensitivity_analysis(
    self,
    parameter: str,
    range_pct: float = 0.20
) -> pd.DataFrame:
"""
        Sensitivity of WACC to parameter changes.

    Args:
parameter: 'beta', 'debt', 'tax_rate', etc.
    range_pct: +/- percentage to vary
"""
base_value = getattr (self, parameter)
values = np.linspace(
    base_value * (1 - range_pct),
    base_value * (1 + range_pct),
    20
)

waccs = []
for val in values:
            # Create temporary calculator with modified value
kwargs = {
    'company_name': self.company_name,
    'market_cap': self.market_cap,
    'total_debt': self.total_debt,
    'risk_free_rate': self.risk_free_rate,
    'beta': self.beta,
    'market_risk_premium': self.market_risk_premium,
    'cost_of_debt': self.cost_of_debt,
    'tax_rate': self.tax_rate
}
kwargs[parameter] = val

temp_calc = WACCCalculator(** kwargs)
waccs.append (temp_calc.wacc)

return pd.DataFrame({
    parameter: values,
    'WACC': waccs
})


# Calculate WACC for Apple (example values)
apple_wacc = WACCCalculator(
    company_name = 'Apple Inc.',
    market_cap = 2_800_000_000_000,   # $2.8T
    total_debt = 120_000_000_000,      # $120B
    risk_free_rate = 0.04,             # 4 %
beta=1.2,                        # 1.2
    market_risk_premium = 0.065,       # 6.5 %
cost_of_debt=0.03,               # 3 %
tax_rate=0.15,                   # 15 % (effective)
)

print("=" * 60)
print(f"WACC Analysis: {apple_wacc.company_name}")
print("=" * 60)
print(apple_wacc.summary().to_string (index = False))
print("=" * 60)

# Output:
# ============================================================
# WACC Analysis: Apple Inc.
# ============================================================
#                  Component       Value
#     Market Value of Equity   $2800.00B
#       Market Value of Debt    $120.00B
#          Total Firm Value   $2920.00B
#                                       
#              Equity Weight      95.9 %
#                Debt Weight       4.1 %
#                                       
#            Risk - Free Rate       4.00 %
#                      Beta        1.20
#      Market Risk Premium       6.50 %
#           Cost of Equity      11.80 %
#                                       
#  Cost of Debt (pre - tax)       3.00 %
#                 Tax Rate      15.0 %
# Cost of Debt (after - tax)       2.55 %
#                                       
#                     WACC      11.43 %
# ============================================================
\`\`\`

---

## Industry WACC Benchmarks

Different industries have different typical WACCs:

\`\`\`python
"""
Typical WACC by Industry
"""

INDUSTRY_WACC_BENCHMARKS = {
    'Technology': {
        'median_wacc': 0.110,
        'debt_to_equity': 0.10,
        'typical_beta': 1.1,
    },
    'Utilities': {
        'median_wacc': 0.065,
        'debt_to_equity': 1.00,
        'typical_beta': 0.6,
    },
    'Healthcare': {
        'median_wacc': 0.095,
        'debt_to_equity': 0.30,
        'typical_beta': 0.9,
    },
    'Financial Services': {
        'median_wacc': 0.090,
        'debt_to_equity': 2.00,  # Banks have high leverage
        'typical_beta': 1.0,
    },
    'Consumer Staples': {
        'median_wacc': 0.075,
        'debt_to_equity': 0.50,
        'typical_beta': 0.7,
    },
    'Energy': {
        'median_wacc': 0.085,
        'debt_to_equity': 0.60,
        'typical_beta': 1.1,
    },
    'Real Estate': {
        'median_wacc': 0.070,
        'debt_to_equity': 1.50,
        'typical_beta': 0.8,
    },
}

def get_industry_benchmark (industry: str) -> dict:
    """Get WACC benchmark for industry."""
    return INDUSTRY_WACC_BENCHMARKS.get(
        industry,
        {'median_wacc': 0.10, 'debt_to_equity': 0.50, 'typical_beta': 1.0}
    )


# Display benchmarks
print("Industry WACC Benchmarks:")
print("-" * 70)
for industry, metrics in INDUSTRY_WACC_BENCHMARKS.items():
    print(f"{industry:.<25} {metrics['median_wacc']:.1%}  "
          f"(D/E: {metrics['debt_to_equity']:.2f}, "
          f"β: {metrics['typical_beta']:.2f})")

# Output:
# Industry WACC Benchmarks:
# ----------------------------------------------------------------------
# Technology............... 11.0%  (D/E: 0.10, β: 1.10)
# Utilities................ 6.5%  (D/E: 1.00, β: 0.60)
# Healthcare............... 9.5%  (D/E: 0.30, β: 0.90)
# Financial Services....... 9.0%  (D/E: 2.00, β: 1.00)
# Consumer Staples......... 7.5%  (D/E: 0.50, β: 0.70)
# Energy................... 8.5%  (D/E: 0.60, β: 1.10)
# Real Estate.............. 7.0%  (D/E: 1.50, β: 0.80)
\`\`\`

---

## Adjusting WACC for Project Risk

**Company WACC ≠ Project WACC**

Projects riskier than company average → Use higher WACC
Projects safer than average → Use lower WACC

### Risk-Adjusted WACC

\`\`\`python
def risk_adjusted_wacc(
    company_wacc: float,
    project_risk: str
) -> float:
    """
    Adjust WACC for project-specific risk.
    
    Args:
        company_wacc: Company\'s overall WACC
        project_risk: 'low', 'average', 'high', 'very high'
    
    Returns:
        Risk-adjusted WACC
    """
    risk_adjustments = {
        'very low': -0.03,      # -3%
        'low': -0.01,           # -1%
        'average': 0.00,        # No adjustment
        'high': +0.02,          # +2%
        'very high': +0.04,     # +4%
        'speculative': +0.06    # +6%
    }
    
    adjustment = risk_adjustments.get (project_risk.lower(), 0.00)
    return company_wacc + adjustment


# Example: Different projects
company_wacc = 0.10

projects = [
    ('Expand existing factory', 'average'),
    ('New product line', 'high'),
    ('International expansion', 'very high'),
    ('Cost reduction initiative', 'low'),
]

print("Risk-Adjusted WACC by Project:")
print("-" * 70)
for project_name, risk_level in projects:
    adjusted_wacc = risk_adjusted_wacc (company_wacc, risk_level)
    print(f"{project_name:.<40} {risk_level:.<15} {adjusted_wacc:.1%}")

# Output:
# Risk-Adjusted WACC by Project:
# ----------------------------------------------------------------------
# Expand existing factory................. average         10.0%
# New product line........................ high            12.0%
# International expansion................. very high       14.0%
# Cost reduction initiative............... low              9.0%
\`\`\`

---

## Common Mistakes

### Mistake 1: Using Book Values

❌ **Wrong:**
\`\`\`python
# Using book values from balance sheet
equity_book = 500_000_000
debt_book = 300_000_000
wacc = calculate_wacc (equity_book, debt_book, ...)  # WRONG!
\`\`\`

✅ **Correct:**
\`\`\`python
# Using market values
market_cap = share_price * shares_outstanding
market_debt = ... # From bond prices or approximation
wacc = calculate_wacc (market_cap, market_debt, ...)
\`\`\`

### Mistake 2: Forgetting Tax Shield

❌ **Wrong:**
\`\`\`python
wacc = equity_weight * cost_equity + debt_weight * cost_debt  # WRONG!
\`\`\`

✅ **Correct:**
\`\`\`python
wacc = equity_weight * cost_equity + debt_weight * cost_debt * (1 - tax_rate)
\`\`\`

### Mistake 3: Using Company WACC for All Projects

Different risk = different WACC!

---

## Key Takeaways

### WACC Formula
\`\`\`
WACC = (E/V) × Re + (D/V) × Rd × (1 - Tc)
\`\`\`

### Cost of Equity (CAPM)
\`\`\`
Re = Rf + β × (Rm - Rf)
\`\`\`

### Critical Points

✓ **Always use market values**, not book values  
✓ **Tax shield makes debt cheaper** than equity  
✓ **Different projects need different WACCs**  
✓ **WACC is the minimum required return** for projects  
✓ **Beta measures systematic risk** (can't diversify away)

### Typical Ranges

- **Technology**: 9-12% WACC
- **Utilities**: 5-7% WACC
- **Mature companies**: 7-10% WACC
- **Startups**: 15-30% WACC (high risk)

---

## Next Section

Now you know what discount rate to use! Next:
- **CAPM & Beta** (Section 4): Deep dive into cost of equity
- **Capital Structure** (Section 5): Optimal debt-equity mix

**Next Section**: [CAPM & Beta](./capm-beta) →
`,
};
