export const governmentSecurities = {
  title: 'Government Securities',
  id: 'government-securities',
  content: `
# Government Securities

## Introduction

Government securities are debt instruments issued by sovereign governments, considered the **risk-free benchmark** for all other fixed income securities.

**Why critical for engineers**:
- $25+ trillion US Treasury market (largest, most liquid bond market globally)
- Risk-free rate = foundation for all asset pricing (CAPM, DCF, derivatives)
- Fed policy transmitted through Treasury market
- Benchmark for corporate spreads

**What you'll build**: Treasury data fetcher (FRED API), TIPS real yield calculator, auction analyzer.

---

## US Treasury Securities Overview

The US Treasury issues debt in three main categories:

### 1. Treasury Bills (T-Bills)

**Maturity**: ≤ 1 year (4-week, 8-week, 13-week, 26-week, 52-week)

**Structure**: Zero-coupon (sold at discount, pay face value at maturity)

**Example**:
\`\`\`
52-week T-Bill: $980 purchase price, $1,000 face value
Yield = ($1,000 - $980) / $980 = 2.04%
\`\`\`

**Use**: Cash management, short-term parking, liquidity

### 2. Treasury Notes (T-Notes)

**Maturity**: 2, 3, 5, 7, 10 years

**Structure**: Fixed coupon, paid semi-annually

**Example**:
\`\`\`
10-year T-Note: 4.5% coupon, $1,000 face
Pays $22.50 every 6 months for 10 years + $1,000 at maturity
\`\`\`

**Use**: Portfolio duration matching, benchmarking

### 3. Treasury Bonds (T-Bonds)

**Maturity**: 20, 30 years

**Structure**: Fixed coupon, semi-annual payments

**Example**:
\`\`\`
30-year T-Bond: 4.75% coupon
Longest duration, highest interest rate risk
\`\`\`

**Use**: Pension liability matching, duration extension

---

## Treasury Inflation-Protected Securities (TIPS)

**TIPS** = Principal adjusts for inflation, protecting purchasing power.

### Mechanics

**Principal Adjustment**:
\`\`\`
Adjusted Principal = Original Principal × (CPI_today / CPI_issue)

Example:
- Issued at $1,000 when CPI = 260
- Today CPI = 280 (7.7% inflation)
- Adjusted Principal = $1,000 × (280/260) = $1,077
\`\`\`

**Coupon Payment**:
\`\`\`
Coupon = Fixed Rate × Adjusted Principal

Example:
- Fixed rate = 2.5%
- Adjusted principal = $1,077
- Semi-annual coupon = 2.5% × $1,077 / 2 = $13.46
\`\`\`

**At Maturity**: Receive greater of adjusted principal or original principal (deflation floor).

### Real Yield vs Nominal Yield

**Breakeven Inflation Rate**:
\`\`\`
Breakeven = Nominal Treasury Yield - TIPS Real Yield

Example:
- 10-year Treasury: 4.5% (nominal)
- 10-year TIPS: 2.0% (real)
- Breakeven = 4.5% - 2.0% = 2.5%

Interpretation: Market expects 2.5% average inflation over 10 years
\`\`\`

**Trading Strategy**:
- If expect inflation > 2.5%: Buy TIPS (outperform nominal)
- If expect inflation < 2.5%: Buy nominal Treasuries

---

## Python: Treasury Data and TIPS Calculator

\`\`\`python
"""
Treasury Market Data and TIPS Valuation
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import date, datetime
import requests
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TreasuryDataFetcher:
    """
    Fetch Treasury data from FRED (Federal Reserve Economic Data)
    
    Example:
        >>> fetcher = TreasuryDataFetcher(api_key="your_fred_key")
        >>> rates = fetcher.get_treasury_rates()
        >>> print(f"10-year yield: {rates['10yr']:.2f}%")
    """
    
    def __init__(self, api_key: str):
        """
        Initialize with FRED API key
        
        Get free key from: https://fred.stlouisfed.org/docs/api/api_key.html
        """
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        logger.info("Initialized Treasury data fetcher")
    
    def get_treasury_rates(
        self,
        as_of_date: Optional[date] = None
    ) -> Dict[str, float]:
        """
        Get current Treasury yield curve rates
        
        Returns:
            Dict with maturities: {'3mo': 4.5, '2yr': 4.3, '10yr': 4.1, ...}
        """
        # FRED series IDs for Treasury yields
        series_ids = {
            '3mo': 'DGS3MO',
            '6mo': 'DGS6MO',
            '1yr': 'DGS1',
            '2yr': 'DGS2',
            '3yr': 'DGS3',
            '5yr': 'DGS5',
            '7yr': 'DGS7',
            '10yr': 'DGS10',
            '20yr': 'DGS20',
            '30yr': 'DGS30',
        }
        
        rates = {}
        
        for maturity, series_id in series_ids.items():
            try:
                # Fetch from FRED
                params = {
                    'series_id': series_id,
                    'api_key': self.api_key,
                    'file_type': 'json',
                    'sort_order': 'desc',
                    'limit': 1  # Most recent
                }
                
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if data['observations']:
                    rate = float(data['observations'][0]['value'])
                    rates[maturity] = rate
                    
                    logger.debug(f"{maturity} Treasury: {rate:.2f}%")
            
            except Exception as e:
                logger.warning(f"Failed to fetch {maturity}: {e}")
                rates[maturity] = None
        
        return rates
    
    def get_tips_rates(self) -> Dict[str, float]:
        """Get TIPS real yields"""
        # FRED series for TIPS
        series_ids = {
            '5yr': 'DFII5',
            '10yr': 'DFII10',
            '30yr': 'DFII30',
        }
        
        rates = {}
        
        for maturity, series_id in series_ids.items():
            try:
                params = {
                    'series_id': series_id,
                    'api_key': self.api_key,
                    'file_type': 'json',
                    'sort_order': 'desc',
                    'limit': 1
                }
                
                response = requests.get(self.base_url, params=params)
                data = response.json()
                
                if data['observations']:
                    rate = float(data['observations'][0]['value'])
                    rates[maturity] = rate
            
            except Exception as e:
                logger.warning(f"Failed to fetch TIPS {maturity}: {e}")
        
        return rates
    
    def calculate_breakeven_inflation(self) -> Dict[str, float]:
        """
        Calculate breakeven inflation rates
        
        Breakeven = Nominal Treasury - TIPS Real Yield
        """
        nominal_rates = self.get_treasury_rates()
        tips_rates = self.get_tips_rates()
        
        breakevens = {}
        
        for maturity in ['5yr', '10yr', '30yr']:
            if nominal_rates.get(maturity) and tips_rates.get(maturity):
                breakeven = nominal_rates[maturity] - tips_rates[maturity]
                breakevens[maturity] = breakeven
                
                logger.info(
                    f"{maturity} breakeven inflation: {breakeven:.2f}%"
                )
        
        return breakevens


@dataclass
class TIPSBond:
    """
    Treasury Inflation-Protected Security
    
    Example:
        >>> tips = TIPSBond(
        ...     original_principal=1000,
        ...     coupon_rate=0.025,
        ...     issue_cpi=260.0,
        ...     maturity_years=10
        ... )
        >>> value = tips.value(current_cpi=280.0, real_yield=0.02)
        >>> print(f"TIPS value: \\$\{value:.2f}")
    """
    original_principal: float
    coupon_rate: float  # Real coupon rate (fixed)
    issue_cpi: float  # CPI at issuance
    maturity_years: float
    frequency: int = 2  # Semi-annual
    
    def adjusted_principal(self, current_cpi: float) -> float:
        """
        Calculate inflation-adjusted principal
        
        Principal adjusts with CPI
        """
        inflation_ratio = current_cpi / self.issue_cpi
        adjusted = self.original_principal * inflation_ratio
        
        # Deflation floor: never below original principal
        return max(adjusted, self.original_principal)
    
    def coupon_payment(self, current_cpi: float) -> float:
        """
        Calculate semi-annual coupon payment
        
        Coupon = Fixed Rate × Adjusted Principal / Frequency
        """
        adj_principal = self.adjusted_principal(current_cpi)
        return (self.coupon_rate * adj_principal) / self.frequency
    
    def value(
        self,
        current_cpi: float,
        real_yield: float
    ) -> float:
        """
        Price TIPS given current CPI and real yield
        
        Args:
            current_cpi: Current CPI level
            real_yield: Real yield to maturity (annual)
        
        Returns:
            TIPS price
        """
        adj_principal = self.adjusted_principal(current_cpi)
        coupon = self.coupon_payment(current_cpi)
        
        periods = int(self.maturity_years * self.frequency)
        y_per_period = real_yield / self.frequency
        
        # PV of coupons
        if y_per_period == 0:
            pv_coupons = coupon * periods
        else:
            pv_coupons = coupon * (
                (1 - (1 + y_per_period) ** -periods) / y_per_period
            )
        
        # PV of adjusted principal
        pv_principal = adj_principal / ((1 + y_per_period) ** periods)
        
        return pv_coupons + pv_principal
    
    def accrued_inflation_compensation(
        self,
        current_cpi: float
    ) -> float:
        """
        Calculate accrued inflation compensation
        
        This is taxable even though not paid until maturity
        """
        return self.adjusted_principal(current_cpi) - self.original_principal


# Example usage
if __name__ == "__main__":
    print("=== Treasury Market Data ===\\n")
    
    # Fetch current rates (requires FRED API key)
    # fetcher = TreasuryDataFetcher(api_key="your_key_here")
    # rates = fetcher.get_treasury_rates()
    
    # Simulated data
    rates = {
        '3mo': 5.20,
        '2yr': 4.50,
        '5yr': 4.30,
        '10yr': 4.25,
        '30yr': 4.40
    }
    
    print("Treasury Yield Curve:")
    for maturity, rate in rates.items():
        if rate:
            print(f"  {maturity:4}: {rate:.2f}%")
    
    # Check curve shape
    if rates['2yr'] > rates['10yr']:
        print("\\n⚠ INVERTED CURVE (recession signal)")
    else:
        print("\\n✓ Normal curve (upward sloping)")
    
    # TIPS example
    print("\\n=== TIPS Analysis ===\\n")
    
    tips = TIPSBond(
        original_principal=1000,
        coupon_rate=0.025,  # 2.5% real
        issue_cpi=260.0,
        maturity_years=10
    )
    
    # Scenario: 7.7% inflation since issuance
    current_cpi = 280.0
    
    adj_principal = tips.adjusted_principal(current_cpi)
    coupon = tips.coupon_payment(current_cpi)
    
    print(f"Original Principal: \\$\{tips.original_principal:,.2f}")
    print(f"Issue CPI: {tips.issue_cpi}")
    print(f"Current CPI: {current_cpi}")
    print(f"Inflation: {(current_cpi/tips.issue_cpi - 1)*100:.2f}%")
    print(f"\\nAdjusted Principal: \\$\{adj_principal:,.2f}")
    print(f"Semi-annual Coupon: \\$\{coupon:.2f}")
    
    # Value at different real yields
    print("\\n=== TIPS Pricing ===\\n")
    
    for real_yield in [0.015, 0.020, 0.025, 0.030]:
        price = tips.value(current_cpi, real_yield)
        print(f"Real Yield {real_yield*100:.2f}%: Price \\$\{price:.2f}")
    
    # Breakeven inflation
    print("\\n=== Breakeven Inflation ===\\n")
    
    nominal_10yr = 4.25
    tips_10yr = 2.00
    breakeven = nominal_10yr - tips_10yr
    
    print(f"10-year Treasury (nominal): {nominal_10yr:.2f}%")
    print(f"10-year TIPS (real): {tips_10yr:.2f}%")
    print(f"Breakeven Inflation: {breakeven:.2f}%")
    print(f"\\nInterpretation: Market expects {breakeven:.2f}% average inflation")
\`\`\`

---

## Treasury Auctions

The US Treasury conducts regular auctions to issue new securities.

### Auction Process

**1. Announcement**: 1 week before auction
- Amount to be sold
- Maturity date
- Auction date

**2. Bidding**: Two types
- **Competitive bids**: Specify yield (large institutions)
- **Non-competitive bids**: Accept yield (retail investors, up to $10M)

**3. Determination**: Dutch auction
- Bids ranked from lowest to highest yield
- Accept bids until amount sold
- **Stop-out yield**: Highest yield accepted (all pay this)

**4. Settlement**: T+1 or T+2

### Example Auction

**10-Year T-Note Auction**:
- Amount: $38 billion
- Bids received: $92 billion (bid-to-cover ratio = 2.42x)
- Yield range: 4.15% to 4.35%
- Stop-out yield: 4.28%

**Result**: All successful bidders pay 4.28% yield, including those who bid 4.15%.

---

## Risk-Free Rate

**Risk-Free Rate** = Treasury yield (no default risk).

### Use in Finance

**1. CAPM (Capital Asset Pricing Model)**:
\`\`\`
Expected Return = Rf + β × (Market Return - Rf)

Rf = 10-year Treasury (typically)
\`\`\`

**2. DCF (Discounted Cash Flow)**:
\`\`\`
Discount Rate = Rf + Risk Premium
\`\`\`

**3. Option Pricing**:
\`\`\`
Black-Scholes uses risk-free rate for discounting
\`\`\`

**4. Corporate Bond Spreads**:
\`\`\`
Corporate Yield = Treasury Yield + Credit Spread
\`\`\`

---

## Fed Influence on Treasuries

**Federal Reserve** controls short-term rates, influencing entire curve.

### Monetary Policy Transmission

**Rate Hikes**:
- Fed raises Fed Funds rate → T-Bill yields rise
- Forward expectations → long-term yields may rise less (curve flattening)
- Inverted curve if Fed very aggressive

**Quantitative Easing (QE)**:
- Fed buys Treasuries → demand ↑ → prices ↑ → yields ↓
- Typically targets long end (10-year, 30-year)
- Flattens curve

**Quantitative Tightening (QT)**:
- Fed stops reinvesting → supply ↑ → yields ↑

---

## Real-World: 2022-2023 Treasury Market

**Case Study**: Historic yield spike

**Timeline**:
- **Jan 2022**: 10-year yield = 1.50%
- **Mar 2022**: Fed starts hiking (0% → 0.25%)
- **Oct 2022**: 10-year yield = 4.25% (275bp rise in 10 months!)
- **2023**: Curve inverted (2yr > 10yr) for 12+ months

**Impact**:
- Bond portfolios: -13% returns (worst year since 1788)
- 60/40 portfolio: -18% (bonds failed to diversify)
- Banks: $620B unrealized losses on Treasury holdings

**Lesson**: Even "risk-free" Treasuries have interest rate risk.

---

## Key Takeaways

1. **Treasury types**: Bills (≤1yr, zero-coupon), Notes (2-10yr), Bonds (20-30yr)
2. **TIPS**: Principal adjusts for inflation, providing real return protection
3. **Breakeven inflation** = Nominal yield - TIPS real yield
4. **Treasury auctions**: Dutch auction, stop-out yield, bid-to-cover ratio
5. **Risk-free rate**: Foundation for all asset pricing (CAPM, DCF, options)
6. **Fed influence**: Controls short end directly, long end through expectations + QE
7. **2022 lesson**: Treasuries have rate risk even without default risk

**Next Section**: Derivatives Overview - forwards, futures, options, swaps fundamentals.
`,
};
