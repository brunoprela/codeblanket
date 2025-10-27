export const dividendDiscountModel = {
    title: 'Dividend Discount Model (DDM)',
    id: 'dividend-discount-model',
    content: `
# Dividend Discount Model (DDM)

## Introduction

The **Dividend Discount Model (DDM)** values a stock based on the present value of all future dividends.

**Core Principle**: A stock is worth the sum of all dividends it will pay, discounted to present value.

**Formula:**
\`\`\`
Stock Price = Σ [Dividend_t / (1 + r)^t]

Where:
- Dividend_t = Dividend in year t
- r = Required return (cost of equity)
- t = Time period
\`\`\`

**When to Use DDM:**
- **Mature companies**: Stable, predictable dividends (utilities, consumer staples)
- **Banks**: High dividend payout ratios (>60%)
- **REITs**: Required to pay 90% of income as dividends
- **Dividend aristocrats**: Companies with 25+ years of dividend increases

**When NOT to Use DDM:**
- **Growth companies**: Pay no dividends (tech, biotech)
- **Cyclical companies**: Dividends fluctuate wildly
- **Startups**: No earnings, no dividends

**By the end of this section, you'll be able to:**
- Apply Gordon Growth Model for stable dividend stocks
- Use multi-stage DDM for growing dividend companies
- Calculate required return using CAPM
- Value dividend-paying stocks with Python
- Compare DDM to DCF and multiples approaches

---

## Gordon Growth Model

### Constant Growth DDM

**Assumption**: Dividends grow at constant rate (g) forever.

**Formula:**
\`\`\`
P0 = D1 / (r - g)

Where:
- P0 = Current stock price
- D1 = Next year's dividend
- r = Required return (cost of equity)
- g = Perpetual dividend growth rate
\`\`\`

**Constraints:**
- g must be < r (otherwise infinite value!)
- g typically 2-5% (GDP + inflation)

\`\`\`python
"""
Gordon Growth Model Implementation
"""

def gordon_growth_model(
    current_dividend: float,
    growth_rate: float,
    required_return: float
) -> float:
    """
    Calculate stock value using Gordon Growth Model.
    
    Args:
        current_dividend: Most recent dividend (D0)
        growth_rate: Perpetual growth rate (g)
        required_return: Cost of equity (r)
    
    Returns:
        Stock price
    
    Example:
        >>> gordon_growth_model(2.00, 0.05, 0.10)
        42.0
    """
    
    if growth_rate >= required_return:
        raise ValueError(f"Growth rate ({growth_rate:.1%}) must be < required return ({required_return:.1%})")
    
    # Next year's dividend
    d1 = current_dividend * (1 + growth_rate)
    
    # Stock price
    price = d1 / (required_return - growth_rate)
    
    return price

# Example: Utility company
current_div = 2.50  # $2.50 current dividend
growth = 0.03  # 3% perpetual growth
required_return = 0.09  # 9% required return

price = gordon_growth_model(current_div, growth, required_return)

print(f"Stock Value (Gordon Growth Model):")
print(f"  Current Dividend (D0):    \${current_div:.2f}")
print(f"  Growth Rate:              {growth:.1%}")
print(f"  Required Return:          {required_return:.1%}")
print(f"  Next Year Dividend (D1):  \${current_div * (1 + growth):.2f}")
print(f"  Stock Price:              \${price:.2f}")
print(f"  Dividend Yield:           {(current_div * (1 + growth)) / price:.2%}")
\`\`\`

---

## Multi-Stage DDM

### Two-Stage Model

**Assumption**: High growth for N years, then stable growth forever.

**Formula:**
\`\`\`
P0 = Σ [D_t / (1 + r)^t]  (years 1 to N)
   + [P_N / (1 + r)^N]     (terminal value)

Where P_N = D_N+1 / (r - g_stable)
\`\`\`

\`\`\`python
"""
Two-Stage Dividend Discount Model
"""

def two_stage_ddm(
    current_dividend: float,
    high_growth_rate: float,
    high_growth_years: int,
    stable_growth_rate: float,
    required_return: float
) -> dict:
    """
    Two-stage DDM: High growth then stable growth.
    
    Args:
        current_dividend: D0
        high_growth_rate: Growth during high-growth phase
        high_growth_years: Years of high growth
        stable_growth_rate: Perpetual growth after high-growth phase
        required_return: Cost of equity
    
    Returns:
        Dict with valuation details
    """
    
    if stable_growth_rate >= required_return:
        raise ValueError("Stable growth must be < required return")
    
    # Phase 1: High growth dividends
    pv_high_growth = 0
    dividend = current_dividend
    
    for year in range(1, high_growth_years + 1):
        dividend = dividend * (1 + high_growth_rate)
        pv = dividend / (1 + required_return) ** year
        pv_high_growth += pv
    
    # Terminal dividend (first year of stable growth)
    terminal_dividend = dividend * (1 + stable_growth_rate)
    
    # Terminal value (Gordon Growth at end of high growth)
    terminal_value = terminal_dividend / (required_return - stable_growth_rate)
    
    # PV of terminal value
    pv_terminal_value = terminal_value / (1 + required_return) ** high_growth_years
    
    # Total stock price
    stock_price = pv_high_growth + pv_terminal_value
    
    return {
        'PV High Growth Dividends': pv_high_growth,
        'Terminal Value': terminal_value,
        'PV Terminal Value': pv_terminal_value,
        'Stock Price': stock_price,
        'Terminal Value %': pv_terminal_value / stock_price
    }

# Example: Growing bank
result = two_stage_ddm(
    current_dividend=1.50,
    high_growth_rate=0.10,  # 10% growth for 5 years
    high_growth_years=5,
    stable_growth_rate=0.04,  # 4% stable growth
    required_return=0.11
)

print("\\nTwo-Stage DDM Valuation:")
for key, value in result.items():
    if '%' in key:
        print(f"  {key:.<35} {value:.1%}")
    else:
        print(f"  {key:.<35} ${value: .2f
}")
\`\`\`

---

## Calculating Required Return (Cost of Equity)

### CAPM Approach

\`\`\`
r = Rf + β × (Rm - Rf)

Where:
- Rf = Risk-free rate (10-year Treasury)
- β = Stock beta vs market
- Rm - Rf = Market risk premium (6-8%)
\`\`\`

\`\`\`python
"""
Cost of Equity Calculation
"""

def cost_of_equity_capm(
    risk_free_rate: float,
    beta: float,
    market_risk_premium: float
) -> float:
    """Calculate cost of equity using CAPM"""
    return risk_free_rate + beta * market_risk_premium

# Example
rf = 0.045  # 4.5%
beta = 1.2
mrp = 0.065  # 6.5%

coe = cost_of_equity_capm(rf, beta, mrp)
print(f"\\nCost of Equity (CAPM): {coe:.2%}")
\`\`\`

---

## Key Takeaways

### When DDM Works Best

✅ Mature companies with stable dividends
✅ Utilities, banks, REITs (high payout ratios)
✅ Dividend aristocrats (consistent dividend growth)
✅ Benchmarking dividend sustainability

### Limitations

❌ Doesn't work for non-dividend-paying stocks (growth companies)
❌ Assumes dividends = value (ignores buybacks)
❌ Sensitive to growth rate assumption (small change = big valuation swing)
❌ Ignores balance sheet (debt, assets)

### Best Practices

1. **Use for mature, dividend-paying stocks only**
2. **Cross-check with P/E and DCF**
3. **Conservative growth assumptions** (cap at GDP + inflation)
4. **Sensitivity analysis** on growth rate
5. **Consider total shareholder yield** (dividends + buybacks)

---

**Next Section**: [Sum-of-the-Parts Valuation](./sum-of-parts-valuation) →
\`,
};
