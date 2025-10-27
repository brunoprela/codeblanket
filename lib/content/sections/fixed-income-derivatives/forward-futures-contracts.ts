export const forwardFuturesContracts = {
    title: 'Forward and Futures Contracts',
    id: 'forward-futures-contracts',
    content: `
# Forward and Futures Contracts

## Introduction

Forwards and futures are agreements to buy/sell an asset at a future date for a predetermined price. The primary difference: forwards are custom OTC contracts, while futures are standardized and exchange-traded.

**Why critical for engineers**:
- $30+ trillion futures market (highly liquid hedging instruments)
- Basis trading = major arbitrage strategy ($billions in hedge funds)
- Interest rate risk management (Treasury futures most liquid)
- Pricing requires yield curve modeling

**What you'll build**: Futures pricer with basis calculation, cheapest-to-deliver calculator, hedging ratio optimizer.

---

## Forward Contracts

### Structure

**Forward contract** = OTC agreement to transact at future date.

**Key terms**:
- **Underlying**: Asset to be delivered (bond, currency, commodity)
- **Forward price**: Agreed transaction price
- **Maturity/Settlement date**: When delivery occurs
- **Notional**: Size of contract

**Example**:
\`\`\`
Company needs €10M in 6 months for payment
Forward contract: Buy €10M at $1.10/€ in 6 months
Cost: $11M fixed (hedged FX risk)
\`\`\`

### Forward Pricing

**Theoretical fair forward price**:
\`\`\`
F = S × e^((r - y) × T)

Where:
- S = Current spot price
- r = Risk-free rate
- y = Yield/carry (dividends, coupons, storage costs)
- T = Time to maturity
\`\`\`

**Bond forward example**:
\`\`\`
Bond price: $110
Coupon: 4% (paid semi-annually = 2% in 6 months)
Risk-free rate: 3%
Time: 6 months = 0.5 years

F = 110 × e^((0.03 - 0.04) × 0.5)
  = 110 × e^(-0.005)
  = 110 × 0.995
  = $109.45

Lower than spot because bond pays coupon during holding period
\`\`\`

---

## Futures Contracts

### Exchange-Traded Features

**Standardization**:
- **Contract size**: Fixed (e.g., $100,000 for Treasury futures)
- **Delivery months**: March, June, September, December (quarterly)
- **Tick size**: Minimum price movement (1/32 for Treasuries)
- **Tick value**: Dollar value per tick

**Example: 10-Year Treasury Note Futures (ZN)**:
\`\`\`
Contract size: $100,000 face value
Tick size: 1/64 (half of 1/32) = 0.015625
Tick value: $15.625
Margin: ~$1,500 initial (varies)

Price quote: 112-16 (112 and 16/32 = 112.50)
\`\`\`

### Daily Settlement

**Mark-to-market**: Settled every day at close.

**Example**:
\`\`\`
Day 1: Buy 10 contracts at 112-00 (112.00)
Day 2: Settlement at 112-16 (112.50)
Gain: 16/32 = 0.50 × $1,000/point × 10 contracts = $5,000
Cash credited to margin account
\`\`\`

---

## Basis Trading

**Basis** = Spot Price - Futures Price

### Convergence

At futures expiration, basis → 0 (prices converge).

**Normal market**: Basis negative (futures > spot) = Contango
**Inverted market**: Basis positive (spot > futures) = Backwardation

**Example**:
\`\`\`
Today:
- 10-year Treasury bond: $110 (spot)
- Futures (3 months to expiry): $110.75
- Basis: 110 - 110.75 = -$0.75 (contango)

At expiration:
- Bond: $111 (assume)
- Futures: $111 (converges)
- Basis: $0
\`\`\`

### Basis Risk

**Basis risk** = Hedge isn't perfect (basis changes unpredictably).

**Example hedging with futures**:
\`\`\`
Hold $10M 10-year Treasury bonds
Hedge: Short equivalent Treasury futures

If rates rise:
- Bond value falls: -$500K
- Futures profit (short gains): +$480K
- Basis widened: Net loss $20K (basis risk)
\`\`\`

---

## Cheapest-to-Deliver (CTD)

Treasury futures allow delivery of multiple eligible bonds. Seller delivers **cheapest-to-deliver**.

### Conversion Factor

**Conversion factor** = Price bond would have if yielding 6%.

**Purpose**: Normalize bonds with different coupons/maturities.

**Delivery value**:
\`\`\`
Amount received = (Futures Price × Conversion Factor) + Accrued Interest
\`\`\`

### CTD Calculation

**Implied repo rate** determines CTD:
\`\`\`
Repo Rate = (Forward Price - Spot Price + Coupon) / Spot Price × (360/days)

Highest repo rate = CTD
\`\`\`

**Example**:
\`\`\`
Bond A: Price $105, Conversion Factor 0.95, Coupon $2
Bond B: Price $110, Conversion Factor 1.00, Coupon $2.50

Futures Price: 112

Bond A delivery proceeds: 112 × 0.95 = $106.40
Cost: $105
Carry: $2 coupon
Repo: (106.40 - 105 + 2) / 105 = 3.24%

Bond B delivery proceeds: 112 × 1.00 = $112
Cost: $110
Carry: $2.50
Repo: (112 - 110 + 2.50) / 110 = 4.09%

Bond B has higher repo → CTD
\`\`\`

---

## Python: Futures Pricing and CTD

\`\`\`python
"""
Treasury Futures Pricing and Cheapest-to-Deliver
"""
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import date, timedelta
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class Bond:
    """Treasury bond deliverable into futures"""
    cusip: str
    coupon_rate: float
    maturity_date: date
    price: float
    accrued_interest: float = 0.0
    
    def __post_init__(self):
        if self.accrued_interest == 0.0:
            # Simplified: assume mid-period
            self.accrued_interest = (self.coupon_rate / 2) * 0.5


def conversion_factor(
    coupon_rate: float,
    maturity_years: float,
    frequency: int = 2
) -> float:
    """
    Calculate conversion factor for Treasury futures
    
    CF = Price of bond if yielding 6% (standardized)
    
    Args:
        coupon_rate: Annual coupon rate
        maturity_years: Years to maturity
        frequency: Payments per year
    
    Returns:
        Conversion factor
    """
    # Standard yield for conversion factor
    standard_yield = 0.06
    
    coupon_payment = coupon_rate / frequency
    periods = int(maturity_years * frequency)
    y_per_period = standard_yield / frequency
    
    # PV of coupons
    if periods == 0:
        pv_coupons = 0
    else:
        pv_coupons = coupon_payment * (
            (1 - (1 + y_per_period) ** -periods) / y_per_period
        )
    
    # PV of principal
    pv_principal = 1.0 / ((1 + y_per_period) ** periods)
    
    cf = (pv_coupons + pv_principal)
    
    # Round to 4 decimal places (standard)
    return round(cf, 4)


class TreasuryFutures:
    """
    Treasury futures contract with CTD analysis
    
    Example:
        >>> bonds = [Bond(...), Bond(...), Bond(...)]
        >>> futures = TreasuryFutures(
        ...     contract_size=100000,
        ...     futures_price=112.0,
        ...     settlement_date=date(2024, 6, 30),
        ...     deliverable_bonds=bonds
        ... )
        >>> ctd = futures.cheapest_to_deliver()
        >>> print(f"CTD: {ctd['cusip']}, Implied Repo: {ctd['implied_repo']:.2f}%")
    """
    
    def __init__(
        self,
        contract_size: float,
        futures_price: float,
        settlement_date: date,
        deliverable_bonds: List[Bond],
        risk_free_rate: float = 0.05
    ):
        self.contract_size = contract_size
        self.futures_price = futures_price
        self.settlement_date = settlement_date
        self.deliverable_bonds = deliverable_bonds
        self.risk_free_rate = risk_free_rate
        
        logger.info(
            f"Initialized Treasury futures: \${contract_size:,.0f} face, "
            f"price {futures_price}, {len(deliverable_bonds)} deliverable bonds"
        )
    
    def implied_repo_rate(
        self,
        bond: Bond,
        days_to_settlement: int = 90
    ) -> float:
        """
        Calculate implied repo rate for bond
        
        Repo = (Forward_price - Spot_price + Carry) / Spot_price × (360/days)
        
        Where:
        - Forward_price = Futures × Conversion_factor
        - Carry = Coupon received during holding
        """
        # Conversion factor
        years_to_maturity = (
            (bond.maturity_date - self.settlement_date).days / 365.0
        )
        cf = conversion_factor(bond.coupon_rate, years_to_maturity)
        
        # Forward price (what received for delivering this bond)
        forward_price = (self.futures_price / 100) * self.contract_size * cf
        
        # Spot cost (what paid to buy bond now)
        spot_cost = bond.price + bond.accrued_interest
        
        # Carry (coupon income during holding period)
        # Simplified: prorated coupon
        carry = (bond.coupon_rate * self.contract_size / 365) * days_to_settlement
        
        # Implied repo rate (annualized)
        repo_numerator = forward_price - spot_cost + carry
        repo_denominator = spot_cost
        
        if repo_denominator == 0:
            return 0.0
        
        implied_repo = (repo_numerator / repo_denominator) * (360 / days_to_settlement)
        
        return implied_repo * 100  # Return as percentage
    
    def cheapest_to_deliver(self) -> Dict:
        """
        Identify cheapest-to-deliver bond
        
        CTD = bond with highest implied repo rate
        
        Returns:
            Dict with CTD bond info and metrics
        """
        ctd_analysis = []
        
        for bond in self.deliverable_bonds:
            repo = self.implied_repo_rate(bond)
            
            years_to_maturity = (
                (bond.maturity_date - self.settlement_date).days / 365.0
            )
            cf = conversion_factor(bond.coupon_rate, years_to_maturity)
            
            ctd_analysis.append({
                'cusip': bond.cusip,
                'coupon': bond.coupon_rate,
                'maturity': bond.maturity_date,
                'price': bond.price,
                'conversion_factor': cf,
                'implied_repo': repo
            })
        
        # Sort by implied repo (highest = CTD)
        ctd_analysis.sort(key=lambda x: x['implied_repo'], reverse=True)
        
        ctd = ctd_analysis[0]
        
        logger.info(
            f"CTD: {ctd['cusip']}, {ctd['coupon']*100:.2f}% coupon, "
            f"implied repo {ctd['implied_repo']:.2f}%"
        )
        
        return ctd
    
    def hedge_ratio(self, bond_to_hedge: Bond) -> float:
        """
        Calculate hedge ratio for hedging bond with futures
        
        Hedge Ratio = (DV01_bond / DV01_futures) × (1 / Conversion_factor)
        
        Simplified: Use duration approximation
        """
        # CTD bond (drives futures pricing)
        ctd = self.cheapest_to_deliver()
        
        # Find CTD bond object
        ctd_bond = next(
            (b for b in self.deliverable_bonds if b.cusip == ctd['cusip']),
            self.deliverable_bonds[0]
        )
        
        # Simplified duration (use bond pricing formula derivative)
        # In practice: calculate modified duration precisely
        
        # Approximation: Equal duration hedge ratio ≈ 1.0
        # For different durations: bond_duration / futures_duration
        
        hedge_ratio = 1.0  # Simplified
        
        logger.debug(f"Hedge ratio: {hedge_ratio:.4f}")
        
        return hedge_ratio


# Example usage
if __name__ == "__main__":
    print("=== Treasury Futures CTD Analysis ===\\n")
    
    # Define deliverable bonds
    bonds = [
        Bond(
            cusip="912828YK0",
            coupon_rate=0.0375,
            maturity_date=date(2033, 11, 15),
            price=102.5
        ),
        Bond(
            cusip="912828YM6",
            coupon_rate=0.0400,
            maturity_date=date(2033, 5, 15),
            price=104.8
        ),
        Bond(
            cusip="912828XZ1",
            coupon_rate=0.0350,
            maturity_date=date(2033, 2, 15),
            price=100.2
        ),
    ]
    
    # 10-Year Treasury Note Futures
    futures = TreasuryFutures(
        contract_size=100000,
        futures_price=112.0,
        settlement_date=date(2024, 6, 30),
        deliverable_bonds=bonds
    )
    
    # Analyze all bonds
    print("Deliverable Bonds Analysis:\\n")
    
    for bond in bonds:
        repo = futures.implied_repo_rate(bond)
        years_to_mat = (bond.maturity_date - futures.settlement_date).days / 365
        cf = conversion_factor(bond.coupon_rate, years_to_mat)
        
        print(f"{bond.cusip}:")
        print(f"  Coupon: {bond.coupon_rate*100:.2f}%")
        print(f"  Price: ${bond.price:.2f}")
print(f"  Conversion Factor: {cf:.4f}")
print(f"  Implied Repo: {repo:.2f}%")
print()
    
    # Identify CTD
ctd = futures.cheapest_to_deliver()

print("\\n=== Cheapest-to-Deliver ===")
print(f"CUSIP: {ctd['cusip']}")
print(f"Coupon: {ctd['coupon']*100:.2f}%")
print(f"Implied Repo: {ctd['implied_repo']:.2f}% (highest)")
\`\`\`

---

## Hedging with Futures

### Duration-Based Hedge Ratio

**Hedge ratio** = Number of futures contracts to hedge bond position.

\`\`\`
Hedge Ratio = (Bond Value × Bond Duration) / (Futures Value × Futures Duration) × (1 / CF)

Where:
- CF = Conversion factor of CTD bond
\`\`\`

**Example**:
\`\`\`
Bond portfolio: $10M, duration 8
Futures: $100K per contract, duration 7 (CTD), CF 0.95

Hedge Ratio = ($10,000,000 × 8) / ($100,000 × 7 × 0.95)
            = 80,000,000 / 665,000
            = 120.3 contracts

Short 120 futures contracts to hedge
\`\`\`

---

## Key Takeaways

1. **Forward pricing**: F = S × e^((r-y)T), accounts for carry costs/yields
2. **Futures vs forwards**: Standardized, exchange-traded, daily settlement, no counterparty risk
3. **Basis**: Spot - Futures, converges to zero at expiration
4. **CTD**: Bond with highest implied repo rate, seller chooses in Treasury futures
5. **Conversion factor**: Normalizes bonds to 6% yield for fair delivery
6. **Hedge ratio**: (Bond_DV01 / Futures_DV01) × (1/CF) for duration matching
7. **Basis risk**: Imperfect hedge from changing basis

**Next Section**: Swaps - interest rate swaps, currency swaps, valuation and applications.
`,
};

