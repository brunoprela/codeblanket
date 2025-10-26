export const yieldCurvesTermStructure = {
  title: 'Yield Curves and Term Structure',
  id: 'yield-curves-term-structure',
  content: `
# Yield Curves and Term Structure

## Introduction

The **yield curve** is one of the most important graphs in finance - showing the relationship between bond yields and time to maturity. It's used to:
- Predict recessions (inverted curve = recession within 6-24 months, historically accurate)
- Price all fixed income securities (every bond, swap, derivative)
- Assess monetary policy stance (Fed influences short end)
- Value companies (discount rate for DCF models)

**As an engineer**, you'll build systems that:
- Bootstrap yield curves from bond prices
- Interpolate rates for any maturity
- Calculate forward rates
- Detect curve inversions (trading signals)

---

## Spot Rates vs Yield to Maturity

**Spot Rate (Zero Rate)**: Interest rate for a zero-coupon bond of maturity T.

**Yield to Maturity (YTM)**: Internal rate of return for a coupon bond (complex because multiple cash flows).

### Key Difference

- **Spot rate**: Pure rate for time T (one cash flow)
- **YTM**: Average of spot rates across all cash flows (multiple periods)

**Example**:
\`\`\`
2-year bond with 5% coupon:
- Year 1 cash flow discounted at 1-year spot rate (3%)
- Year 2 cash flows discounted at 2-year spot rate (4%)
- YTM ≈ 3.98% (blend of 3% and 4%)
\`\`\`

**Why it matters**: Spot rates are the fundamental building blocks. All securities must be priced using spot rates for no-arbitrage.

---

## Bootstrapping the Yield Curve

**Bootstrapping**: Extracting spot rates from coupon bond prices.

### Algorithm

Given bond prices, solve for spot rates iteratively:

1. Start with shortest maturity (T-Bill gives 3-month spot rate directly)
2. Use known spot rates to solve for next maturity's spot rate
3. Repeat until full curve constructed

### Mathematical Formula

For bond with price P, coupon C, and maturity n:

\`\`\`
P = C/(1+s₁) + C/(1+s₂)² + ... + (C+FV)/(1+sₙ)ⁿ
\`\`\`

Solve for sₙ given s₁, s₂, ..., sₙ₋₁

---

## Python: Yield Curve Bootstrapping

\`\`\`python
"""
Yield Curve Bootstrapping Engine
"""
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import date
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d
import logging

logger = logging.getLogger(__name__)


@dataclass
class BondQuote:
    """Market quote for a bond"""
    maturity_years: float
    coupon_rate: float  # Annual coupon rate
    price: float  # Clean price
    face_value: float = 1000.0
    frequency: int = 2  # Semi-annual


class YieldCurve:
    """
    Yield curve construction via bootstrapping
    
    Example:
        >>> bonds = [
        ...     BondQuote(0.25, 0.0, 99.50),  # 3-month T-Bill
        ...     BondQuote(0.50, 0.0, 99.00),  # 6-month T-Bill
        ...     BondQuote(1.0, 0.03, 99.80),   # 1-year note
        ...     BondQuote(2.0, 0.035, 100.50), # 2-year note
        ... ]
        >>> curve = YieldCurve.bootstrap(bonds)
        >>> rate_1yr = curve.spot_rate(1.0)
        >>> print(f"1-year spot: {rate_1yr*100:.2f}%")
    """
    
    def __init__(self, maturities: np.ndarray, spot_rates: np.ndarray):
        """
        Initialize yield curve
        
        Args:
            maturities: Array of maturities in years
            spot_rates: Array of spot rates (annual, continuous compounding)
        """
        if len(maturities) != len(spot_rates):
            raise ValueError("Maturities and spot rates must have same length")
        
        if not np.all(maturities[:-1] < maturities[1:]):
            raise ValueError("Maturities must be in ascending order")
        
        self.maturities = maturities
        self.spot_rates = spot_rates
        self.interpolator = CubicSpline(maturities, spot_rates)
        
        logger.info(f"Created yield curve with {len(maturities)} points")
    
    @classmethod
    def bootstrap(cls, bonds: List[BondQuote]) -> 'YieldCurve':
        """
        Bootstrap spot rates from bond prices
        
        Args:
            bonds: List of bond quotes (sorted by maturity)
        
        Returns:
            YieldCurve object with spot rates
        """
        # Sort bonds by maturity
        bonds = sorted(bonds, key=lambda b: b.maturity_years)
        
        maturities = []
        spot_rates = []
        
        for bond in bonds:
            logger.debug(f"Bootstrapping {bond.maturity_years}yr bond...")
            
            # Special case: Zero-coupon bonds (T-Bills)
            if bond.coupon_rate == 0:
                # Price = FV / (1 + r)^t
                # r = (FV / Price)^(1/t) - 1
                spot = (bond.face_value / bond.price) ** (1 / bond.maturity_years) - 1
                
                maturities.append(bond.maturity_years)
                spot_rates.append(spot)
                
                logger.debug(f"  Zero-coupon: spot = {spot*100:.4f}%")
                continue
            
            # Coupon bonds: Bootstrap spot rate
            spot = cls._bootstrap_spot_rate(
                bond,
                maturities,
                spot_rates
            )
            
            maturities.append(bond.maturity_years)
            spot_rates.append(spot)
            
            logger.debug(f"  Coupon bond: spot = {spot*100:.4f}%")
        
        return cls(np.array(maturities), np.array(spot_rates))
    
    @staticmethod
    def _bootstrap_spot_rate(
        bond: BondQuote,
        known_maturities: List[float],
        known_spots: List[float]
    ) -> float:
        """
        Solve for spot rate given known shorter-term spots
        
        Uses Newton-Raphson to solve:
        Price = Σ CF_t / (1 + s_t)^t
        """
        # Generate cash flows
        periods_per_year = bond.frequency
        num_periods = int(bond.maturity_years * periods_per_year)
        coupon_payment = (bond.coupon_rate * bond.face_value) / periods_per_year
        
        # Create temporary interpolator for known spots
        if len(known_spots) > 0:
            known_interpolator = interp1d(
                known_maturities,
                known_spots,
                kind='linear',
                fill_value='extrapolate'
            )
        else:
            known_interpolator = None
        
        # Calculate PV of known cash flows (before maturity)
        pv_known = 0.0
        for i in range(1, num_periods):
            t = i / periods_per_year
            
            if t < bond.maturity_years:
                # Interpolate spot rate for this maturity
                if known_interpolator is not None:
                    s_t = float(known_interpolator(t))
                else:
                    s_t = 0.03  # Default guess
                
                pv = coupon_payment / ((1 + s_t) ** t)
                pv_known += pv
        
        # Solve for spot rate at maturity
        # Price = pv_known + (Coupon + Face) / (1 + s_T)^T
        # s_T = [(Coupon + Face) / (Price - pv_known)]^(1/T) - 1
        
        final_cash_flow = coupon_payment + bond.face_value
        remaining_pv_needed = bond.price - pv_known
        
        if remaining_pv_needed <= 0:
            logger.warning(f"Negative remaining PV for {bond.maturity_years}yr bond")
            remaining_pv_needed = bond.price * 0.5  # Fallback
        
        spot = (final_cash_flow / remaining_pv_needed) ** (1 / bond.maturity_years) - 1
        
        return max(spot, 0.0001)  # Floor at 0.01%
    
    def spot_rate(self, maturity: float) -> float:
        """
        Get spot rate for any maturity (interpolated)
        
        Args:
            maturity: Time to maturity in years
        
        Returns:
            Spot rate (annual)
        """
        if maturity < self.maturities[0]:
            # Extrapolate for very short maturities
            return float(self.spot_rates[0])
        
        if maturity > self.maturities[-1]:
            # Flat extrapolation for long maturities
            return float(self.spot_rates[-1])
        
        return float(self.interpolator(maturity))
    
    def forward_rate(self, start: float, end: float) -> float:
        """
        Calculate forward rate between two periods
        
        Forward rate f(t1, t2) is the rate implied for borrowing
        from time t1 to time t2.
        
        Formula: (1 + s2)^t2 = (1 + s1)^t1 × (1 + f)^(t2-t1)
        Solving: f = [(1 + s2)^t2 / (1 + s1)^t1]^(1/(t2-t1)) - 1
        
        Args:
            start: Start time in years
            end: End time in years
        
        Returns:
            Forward rate (annual)
        """
        if end <= start:
            raise ValueError("End time must be after start time")
        
        s1 = self.spot_rate(start)
        s2 = self.spot_rate(end)
        
        # Calculate forward rate
        forward = ((1 + s2) ** end / (1 + s1) ** start) ** (1 / (end - start)) - 1
        
        return forward
    
    def discount_factor(self, maturity: float) -> float:
        """
        Calculate discount factor for maturity T
        
        DF(T) = 1 / (1 + s(T))^T
        """
        s = self.spot_rate(maturity)
        return 1 / ((1 + s) ** maturity)
    
    def shape(self) -> str:
        """
        Classify yield curve shape
        
        Returns:
            'normal', 'inverted', 'flat', or 'humped'
        """
        short_rate = self.spot_rate(0.25)  # 3-month
        medium_rate = self.spot_rate(2.0)  # 2-year
        long_rate = self.spot_rate(10.0)   # 10-year
        
        # Inverted: short > long
        if short_rate > long_rate + 0.002:  # 20 bps threshold
            return 'inverted'
        
        # Flat: similar rates
        if abs(long_rate - short_rate) < 0.002:
            return 'flat'
        
        # Humped: medium highest
        if medium_rate > short_rate and medium_rate > long_rate:
            return 'humped'
        
        # Normal: upward sloping
        return 'normal'
    
    def to_dataframe(self) -> pd.DataFrame:
        """Export curve to pandas DataFrame"""
        return pd.DataFrame({
            'Maturity': self.maturities,
            'Spot_Rate': self.spot_rates * 100,  # Convert to percentage
        })
    
    def plot(self):
        """Plot yield curve (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed")
            return
        
        # Plot actual points
        plt.figure(figsize=(10, 6))
        plt.scatter(self.maturities, self.spot_rates * 100, 
                   label='Bootstrapped Points', s=100, zorder=3)
        
        # Plot interpolated curve
        fine_maturities = np.linspace(
            self.maturities[0],
            self.maturities[-1],
            100
        )
        fine_rates = [self.spot_rate(m) * 100 for m in fine_maturities]
        plt.plot(fine_maturities, fine_rates, 
                label='Interpolated Curve', linewidth=2)
        
        plt.xlabel('Maturity (Years)')
        plt.ylabel('Spot Rate (%)')
        plt.title(f'Yield Curve - {self.shape().title()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


# Example usage
if __name__ == "__main__":
    # Market bond quotes (example)
    bonds = [
        # T-Bills (zero-coupon)
        BondQuote(maturity_years=0.25, coupon_rate=0.0, price=99.00),
        BondQuote(maturity_years=0.50, coupon_rate=0.0, price=98.00),
        BondQuote(maturity_years=1.0, coupon_rate=0.0, price=96.20),
        
        # T-Notes (coupon bonds)
        BondQuote(maturity_years=2.0, coupon_rate=0.04, price=100.15),
        BondQuote(maturity_years=3.0, coupon_rate=0.045, price=101.20),
        BondQuote(maturity_years=5.0, coupon_rate=0.05, price=102.50),
        BondQuote(maturity_years=7.0, coupon_rate=0.055, price=103.80),
        BondQuote(maturity_years=10.0, coupon_rate=0.06, price=105.00),
    ]
    
    print("=== Yield Curve Bootstrapping ===\\n")
    
    # Bootstrap the curve
    curve = YieldCurve.bootstrap(bonds)
    
    # Display spot rates
    print("Spot Rates:")
    for mat in [0.25, 0.5, 1, 2, 3, 5, 7, 10]:
        rate = curve.spot_rate(mat)
        print(f"  {mat:5.2f}yr: {rate*100:6.3f}%")
    
    print(f"\\nCurve Shape: {curve.shape().upper()}")
    
    # Forward rates
    print("\\nForward Rates:")
    print(f"  1yr-2yr: {curve.forward_rate(1, 2)*100:.3f}%")
    print(f"  2yr-5yr: {curve.forward_rate(2, 5)*100:.3f}%")
    print(f"  5yr-10yr: {curve.forward_rate(5, 10)*100:.3f}%")
    
    # Discount factors
    print("\\nDiscount Factors:")
    for mat in [1, 5, 10]:
        df = curve.discount_factor(mat)
        print(f"  {mat}yr: {df:.6f}")
\`\`\`

---

## Yield Curve Shapes

### 1. Normal (Upward Sloping)

**Characteristics**:
- Long-term rates > short-term rates
- Most common shape (70% of time historically)
- Reflects healthy economy with growth expectations

**Interpretation**:
- Investors demand higher returns for longer-term risk
- Fed not currently restricting (accommodative policy)
- Economy expected to grow

### 2. Inverted (Downward Sloping)

**Characteristics**:
- Short-term rates > long-term rates
- Rare but important (recession predictor)
- Often caused by Fed raising rates aggressively

**Interpretation**:
- **Recession signal**: Every US recession since 1955 preceded by inversion
- Timeline: Inversion → Recession in 6-24 months
- Fed fighting inflation by raising short rates
- Long rates fall as investors expect Fed to cut rates later

**Trading Strategy**: When curve inverts, consider defensive positioning (bonds, utilities, gold).

### 3. Flat

**Characteristics**:
- Similar yields across maturities
- Transition state (normal ↔ inverted)

**Interpretation**:
- Uncertainty about economic direction
- Fed pause (waiting to see data)

### 4. Humped

**Characteristics**:
- Medium-term rates highest
- Rare shape

**Interpretation**:
- Mixed expectations (short-term: tight policy, long-term: easing)

---

## Term Structure Theories

### 1. Expectations Hypothesis

**Theory**: Forward rates = expected future spot rates.

**Implication**: Yield curve shape reflects market's expectations for future interest rates.

**Example**:
- If 1-year rate = 3% and 2-year rate = 4%
- Implied 1yr-2yr forward = 5%
- Market expects rates to rise from 3% to 5%

### 2. Liquidity Preference Theory

**Theory**: Investors demand **liquidity premium** for longer maturities.

**Implication**: Yield curve naturally slopes upward (even if rates expected flat).

**Rationale**:
- Longer bonds have more interest rate risk
- Investors need compensation for tying up money
- Explains why curve typically upward-sloping

### 3. Market Segmentation Theory

**Theory**: Different investors operate in different maturity segments.

**Examples**:
- Banks: Short-term (matching deposits)
- Pension funds: Long-term (matching liabilities)
- Hedge funds: Opportunistic across maturities

**Implication**: Supply/demand in each segment drives rates independently.

---

## Forward Rates

**Forward rate**: Implied future interest rate derived from spot rates.

### Formula

\`\`\`
(1 + s₂)² = (1 + s₁) × (1 + f₁,₂)

Solving for forward:
f₁,₂ = [(1 + s₂)² / (1 + s₁)] - 1
\`\`\`

### Interpretation

If 1-year spot = 3% and 2-year spot = 4%:
\`\`\`
f₁,₂ = [(1.04)² / 1.03] - 1 = 1.0816 / 1.03 - 1 = 5.01%
\`\`\`

**Meaning**: Market implies 1-year rate will be ~5% one year from now.

### Trading Application

**Forward Rate Agreement (FRA)**: Lock in forward rate today.

Example: Borrow $1M in 1 year for 1 year at 5.01% (forward rate), regardless of actual spot rate then.

---

## Real-World: Recession Prediction

**Historical Accuracy**: 2yr-10yr yield curve inversion has predicted every US recession:

- **1989**: Inverted → 1990-91 recession
- **2000**: Inverted → 2001 recession  
- **2006**: Inverted → 2008-09 recession
- **2019**: Inverted → 2020 recession (COVID accelerated)
- **2022-2023**: Inverted → ? (as of writing)

**Why it works**:
- Short rates ↑: Fed fighting inflation
- Long rates ↓: Expectations of future rate cuts (recession = Fed easing)

---

## Key Takeaways

1. **Yield curve = relationship between maturity and yield**
2. **Bootstrapping**: Extract spot rates from bond prices iteratively
3. **Forward rates**: Implied future rates from spot curve
4. **Inverted curve**: Powerful recession predictor (6-24 month lead time)
5. **Three theories**: Expectations, liquidity preference, market segmentation

**Next Section**: Duration and Convexity - measuring interest rate risk.
`,
};

