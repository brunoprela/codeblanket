export const realOptionsValuation = {
    title: 'Real Options Valuation',
    id: 'real-options-valuation',
    content: `
# Real Options Valuation

## Introduction

**Real options** value managerial flexibility and strategic choices—the ability to expand, abandon, delay, or switch strategies based on future information.

**Traditional DCF Problem**: Assumes fixed strategy (build plant, launch product, continue forever). Ignores flexibility value.

**Real Options Solution**: Value the option to adapt strategy as uncertainty resolves.

**Types of Real Options:**
1. **Option to Expand**: Invest more if market grows
2. **Option to Abandon**: Exit if market declines
3. **Option to Delay**: Wait for more information
4. **Option to Switch**: Change inputs/outputs
5. **Growth Options**: Platform investments enabling future projects

**By the end of this section:**
- Understand option thinking in corporate finance
- Value expansion and abandonment options
- Apply Black-Scholes to real assets
- Model staged investments (decision trees)
- Implement real options in Python

---

## Option Thinking Framework

### Traditional NPV vs Real Options

**Example**: Mining project

**Traditional NPV:**
- Invest $100M today
- Mine for 10 years
- PV of cash flows = $90M
- NPV = -$10M → **REJECT**

**Real Options View:**
- Invest $10M for exploration rights (1 year)
- After 1 year, **CHOOSE**: Develop mine ($90M) if commodity prices high, OR Abandon if prices low
- Value = $10M exploration + Option value
- If option value > $0, project may be attractive despite negative NPV

**Key Insight**: Flexibility has value. The ability to wait, expand, or abandon is worth money.

### Black-Scholes for Real Assets

Financial option: Right (not obligation) to buy stock at strike price.
Real option: Right (not obligation) to invest in project at investment cost.

| Financial Option | Real Option |
|-----------------|-------------|
| Stock price (S) | PV of cash flows |
| Strike price (K) | Investment cost |
| Time to maturity (T) | Time until decision |
| Volatility (σ) | Project uncertainty |
| Risk-free rate (r) | Treasury rate |

\`\`\`python
"""
Real Options Valuation using Black-Scholes
"""

import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    """
    Black-Scholes formula for call option.
    
    Args:
        S: Current asset value (PV of project cash flows)
        K: Strike price (investment cost)
        T: Time to expiration (years until decision)
        r: Risk-free rate
        sigma: Volatility of returns
    
    Returns:
        Call option value
    """
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    call_value = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    return call_value

# Example: Option to delay investment
S = 90_000_000  # PV of cash flows = $90M
K = 100_000_000  # Investment cost = $100M
T = 1  # Can wait 1 year
r = 0.04  # 4% risk-free rate
sigma = 0.40  # 40% volatility

option_value = black_scholes_call(S, K, T, r, sigma)

print(f"Traditional NPV: \${S - K:,.0f} (REJECT)")
print(f"Option to delay value: \${option_value:,.0f}")
print(f"Total project value: \${option_value:,.0f} (may be worth waiting)")
\`\`\`

---

## Key Takeaways

- Real options value managerial flexibility (expand, abandon, delay, switch)
- Traditional NPV assumes fixed strategy—undervalues flexible projects
- Black-Scholes can value real options (with appropriate assumptions)
- Volatility is friend, not enemy—uncertainty creates option value
- Common in: R&D, natural resources, infrastructure, tech platforms

**Next Section**: [Dividend Discount Model](./dividend-discount-model) →
\`,
};
