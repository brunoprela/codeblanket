export const creditDefaultSwaps = {
    title: 'Credit Default Swaps (CDS)',
    id: 'credit-default-swaps',
    content: `
# Credit Default Swaps (CDS)

## Introduction

A **Credit Default Swap (CDS)** is insurance against default. The protection buyer pays periodic premiums to the protection seller, who compensates if a credit event occurs.

**Why critical for engineers**:
- $3+ trillion CDS market (credit risk transfer mechanism)
- Pricing requires survival probability modeling
- 2008 crisis centered on CDS (AIG, Lehman)
- Complex curve bootstrapping from CDS spreads

**What you'll build**: CDS pricer, hazard rate curve builder, CDS index tracker, counterparty risk calculator.

---

## CDS Structure

### Basic Mechanics

**Parties**:
- **Protection buyer**: Pays premium, receives payoff if default
- **Protection seller**: Receives premium, pays if default

**Terms**:
- **Reference entity**: Company/sovereign being insured
- **Notional**: Face value of debt being insured
- **Tenor**: 1, 3, 5, 10 years (5yr most liquid)
- **Spread**: Annual premium in basis points

**Example**:
\`\`\`
Reference entity: Tesla Inc.
Notional: $10 million
Tenor: 5 years
CDS spread: 200 basis points (2%)

Buyer pays: $200,000 per year (quarterly: $50K)
If Tesla defaults: Seller pays ~$10M (depends on recovery)
\`\`\`

### Credit Events

**Triggers** (ISDA definitions):
1. **Bankruptcy**: Chapter 11 filing
2. **Failure to pay**: Missed interest/principal payment
3. **Restructuring**: Debt terms worsened (extended maturity, reduced rate)

**Settlement**:
- **Physical**: Buyer delivers bonds, receives par
- **Cash**: Auction determines recovery rate, seller pays (100% - Recovery) × Notional

---

## CDS Pricing

### Theoretical Framework

**CDS spread** compensates for expected loss from default.

\`\`\`
CDS Spread ≈ PD × LGD

Where:
- PD = Probability of Default (annual)
- LGD = Loss Given Default (1 - Recovery Rate)
\`\`\`

**Example**:
\`\`\`
Annual PD: 2%
Recovery rate: 40%
LGD: 60%

Fair CDS spread: 2% × 60% = 1.2% = 120 basis points
\`\`\`

### Hazard Rate Model

**Hazard rate (λ)** = instantaneous default probability.

**Survival probability**:
\`\`\`
S(t) = e^(-λt)

For constant λ
\`\`\`

**Default probability**:
\`\`\`
PD(0,t) = 1 - S(t) = 1 - e^(-λt)
\`\`\`

### Premium Leg vs Protection Leg

**Premium leg** (present value):
\`\`\`
PV_premium = Spread × Notional × Σ[S(t_i) × DF(t_i) × Δt_i]

Sum over payment dates
\`\`\`

**Protection leg** (present value):
\`\`\`
PV_protection = Notional × LGD × Σ[PD(t_{i-1}, t_i) × DF(t_i)]

Sum over periods, weighted by default probability in each period
\`\`\`

**Fair CDS spread**: PV_premium = PV_protection

---

## Python: CDS Pricer

\`\`\`python
"""
Credit Default Swap Pricing
"""
from typing import List, Tuple
from dataclasses import dataclass
from datetime import date, timedelta
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class CDSContract:
    """
    Credit Default Swap
    
    Example:
        >>> cds = CDSContract(
        ...     notional=10_000_000,
        ...     spread=0.02,  # 200bp
        ...     tenor_years=5,
        ...     recovery_rate=0.40
        ... )
        >>> fair_spread = cds.fair_spread(hazard_rate=0.03)
        >>> mtm = cds.mark_to_market(market_spread=0.025, hazard_rate=0.03)
    """
    notional: float
    spread: float  # Annual spread (e.g., 0.02 = 200bp)
    tenor_years: int
    recovery_rate: float = 0.40
    frequency: int = 4  # Quarterly payments
    
    def survival_probability(self, hazard_rate: float, time: float) -> float:
        """
        Calculate survival probability
        
        S(t) = exp(-λt)
        """
        return np.exp(-hazard_rate * time)
    
    def default_probability(
        self,
        hazard_rate: float,
        t1: float,
        t2: float
    ) -> float:
        """
        Probability of default between t1 and t2
        
        PD(t1,t2) = S(t1) - S(t2)
        """
        s1 = self.survival_probability(hazard_rate, t1)
        s2 = self.survival_probability(hazard_rate, t2)
        return s1 - s2
    
    def premium_leg_pv(
        self,
        hazard_rate: float,
        discount_rate: float = 0.05
    ) -> float:
        """
        Present value of premium payments
        
        Accounts for possibility of default (no payment if defaulted)
        """
        pv = 0.0
        dt = 1.0 / self.frequency  # Period length
        periods = self.tenor_years * self.frequency
        
        premium_per_period = self.spread * self.notional * dt
        
        for i in range(1, periods + 1):
            t = i * dt
            
            # Survival probability (pay premium only if no default)
            survival = self.survival_probability(hazard_rate, t)
            
            # Discount factor
            df = np.exp(-discount_rate * t)
            
            pv += premium_per_period * survival * df
        
        return pv
    
    def protection_leg_pv(
        self,
        hazard_rate: float,
        discount_rate: float = 0.05
    ) -> float:
        """
        Present value of protection payments
        
        Payoff if default occurs: (1 - Recovery) × Notional
        """
        lgd = 1.0 - self.recovery_rate
        payoff = lgd * self.notional
        
        pv = 0.0
        dt = 1.0 / self.frequency
        periods = self.tenor_years * self.frequency
        
        for i in range(1, periods + 1):
            t_start = (i - 1) * dt
            t_end = i * dt
            
            # Probability of default in this period
            pd = self.default_probability(hazard_rate, t_start, t_end)
            
            # Discount factor (assume default mid-period)
            t_mid = (t_start + t_end) / 2
            df = np.exp(-discount_rate * t_mid)
            
            pv += payoff * pd * df
        
        return pv
    
    def fair_spread(
        self,
        hazard_rate: float,
        discount_rate: float = 0.05
    ) -> float:
        """
        Calculate fair CDS spread
        
        Spread such that PV_premium = PV_protection
        """
        # Temporarily set spread to 1.0 to get "spread sensitivity"
        original_spread = self.spread
        self.spread = 1.0
        
        premium_per_unit = self.premium_leg_pv(hazard_rate, discount_rate)
        protection = self.protection_leg_pv(hazard_rate, discount_rate)
        
        # Fair spread
        fair = protection / premium_per_unit
        
        # Restore original spread
        self.spread = original_spread
        
        logger.info(f"Fair CDS spread: {fair*10000:.0f} bp")
        
        return fair
    
    def mark_to_market(
        self,
        market_spread: float,
        hazard_rate: float,
        discount_rate: float = 0.05
    ) -> float:
        """
        Calculate current mark-to-market value
        
        MTM = PV(protection) - PV(premium at market spread)
        
        From protection buyer's perspective
        """
        # Protection leg value (what we receive if default)
        pv_protection = self.protection_leg_pv(hazard_rate, discount_rate)
        
        # Premium leg at MARKET spread (what we pay)
        original_spread = self.spread
        self.spread = market_spread
        pv_premium_market = self.premium_leg_pv(hazard_rate, discount_rate)
        self.spread = original_spread
        
        # MTM for protection buyer
        mtm = pv_protection - pv_premium_market
        
        logger.info(f"CDS MTM: \${mtm:,.2f}")
        
        return mtm


# Example usage
if __name__ == "__main__":
    print("=== CDS Pricing ===\\n")

cds = CDSContract(
    notional = 10_000_000,
    spread = 0.02,  # 200bp
        tenor_years = 5,
    recovery_rate = 0.40
)

hazard_rate = 0.03  # 3 % annual default intensity
    
    # Fair spread
fair = cds.fair_spread(hazard_rate)
print(f"Contract spread: {cds.spread*10000:.0f} bp")
print(f"Fair spread: {fair*10000:.0f} bp")

if fair > cds.spread:
    print(f"→ Contract CHEAP (fair spread {(fair-cds.spread)*10000:.0f}bp higher)")
else:
print(f"→ Contract RICH (fair spread {(cds.spread-fair)*10000:.0f}bp lower)")
    
    # Mark - to - market
print("\\n=== Mark-to-Market ===\\n")

market_spread = 0.025  # Market now trading at 250bp

mtm = cds.mark_to_market(market_spread, hazard_rate)

print(f"Market spread: {market_spread*10000:.0f} bp")
print(f"Contract spread: {cds.spread*10000:.0f} bp")
print(f"MTM (buyer): \${mtm:,.2f}
}")

if mtm > 0:
    print("→ Protection buyer has GAIN (spreads widened)")
else:
print("→ Protection buyer has LOSS (spreads tightened)")
\`\`\`

---

## CDS Index

**CDS Index** = Basket of single-name CDS (standardized, liquid).

### Major Indices

**CDX (North America)**:
- **CDX.IG**: 125 investment-grade companies
- **CDX.HY**: 100 high-yield companies

**iTraxx (Europe)**:
- **iTraxx Europe**: 125 IG European companies
- **iTraxx Crossover**: 75 HY European companies

### Index Trading

**Spread**: Weighted average of constituents.

**Example**:
\`\`\`
CDX.IG trades at 60bp
Buying protection on $10M: Pay $60K annually
If any constituent defaults: Payoff on that name, index continues
\`\`\`

---

## 2008 Financial Crisis

**AIG CDS exposure**:
- Sold $500+ billion CDS protection
- Counterparties: Banks holding mortgage bonds
- Crisis: Mortgage defaults surged → CDS payouts required
- Problem: AIG didn't have reserves (no collateral posted)
- Bailout: US government $182 billion to prevent systemic collapse

**Lesson**: CDS counterparty risk can be systemic.

---

## Key Takeaways

1. **CDS = credit insurance**: Pay premium, receive payoff if default
2. **Pricing**: CDS spread ≈ PD × LGD, fair spread makes PV_premium = PV_protection
3. **Hazard rate**: Instantaneous default probability λ, S(t) = e^(-λt)
4. **Settlement**: Physical (deliver bonds) or cash (auction determines recovery)
5. **CDS Index**: Basket of single names (CDX.IG, iTraxx), liquid trading
6. **Crisis lesson**: Massive CDS exposure without collateral = systemic risk

**Next Section**: Exotic Derivatives - barriers, digitals, structured products.
`,
};

