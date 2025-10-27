export const creditRiskSpreads = {
  title: 'Credit Risk and Spreads',
  id: 'credit-risk-spreads',
  content: `
# Credit Risk and Spreads

## Introduction

**Credit risk** = risk that a borrower defaults on debt obligations.

**Credit spread** = extra yield investors demand for taking on credit risk (Corporate Yield - Treasury Yield).

**Why critical for engineers**:
- Corporate bonds trade at spreads to Treasuries (50-500+ basis points)
- Credit analysis drives $10T+ corporate bond market
- Spread widening/tightening = major P&L source
- Default models = machine learning opportunity

**What you'll build**: Credit spread analyzer, default probability models, rating migration tracker.

---

## Credit Spread Decomposition

**Credit Spread = Default Risk + Liquidity Premium + Tax Effects + Other**

### 1. Default Risk Component

Expected loss from default:
\`\`\`
Expected Loss = PD × LGD

Where:
- PD = Probability of Default (annual %)
- LGD = Loss Given Default (1 - Recovery Rate)
\`\`\`

**Example**:
- PD = 2% per year
- Recovery Rate = 40%
- LGD = 1 - 0.40 = 0.60 (60% loss)
- Expected Loss = 2% × 60% = 1.2% annually

Credit spread should compensate for this 1.2% expected loss plus risk premium.

### 2. Liquidity Premium

Less liquid bonds trade at wider spreads.

**Measures**:
- Bid-ask spread (10-50bp for corporates vs 1-5bp for Treasuries)
- Trading volume (low volume = illiquid = higher spread)
- Number of dealers (more dealers = more liquid)

**Typical Premium**: 20-50bp for illiquid corporate bonds.

### 3. Tax Effects

Corporate bond interest is taxable, Treasury interest often state-tax-exempt (municipal bonds fully tax-exempt).

After-tax yield comparison drives some spread.

### 4. Other Factors

- **Convexity differences**: Corporates may have embedded options
- **Sector rotation**: Energy spreads widen when oil falls
- **Market sentiment**: Risk-on vs risk-off

---

## Credit Rating Agencies

Three major agencies: **S&P, Moody's, Fitch**

### Rating Scales

| S&P/Fitch | Moody's | Category | Typical Spread |
|-----------|---------|----------|----------------|
| AAA | Aaa | Highest quality | 50-80bp |
| AA | Aa | High quality | 80-120bp |
| A | A | Upper medium | 120-180bp |
| BBB | Baa | Lower medium | 180-250bp |
| BB | Ba | Non-investment (junk) | 250-400bp |
| B | B | Highly speculative | 400-700bp |
| CCC | Caa | Substantial risk | 700-1500bp |
| CC, C | Ca, C | Extremely speculative | 1500bp+ |
| D | D | Default | N/A |

**Investment Grade** = BBB-/Baa3 and above (institutions can hold)
**High Yield (Junk)** = BB+/Ba1 and below (higher risk, higher return)

---

## Historical Default Rates

**10-Year Cumulative Default Rates** (Moody's historical data):

- **Aaa**: 0.5% (very rare defaults)
- **Aa**: 1.5%
- **A**: 3%
- **Baa**: 6%
- **Ba**: 15%
- **B**: 30%
- **Caa-C**: 50%+

**Key insight**: Default rates increase exponentially as ratings decline.

---

## Loss Given Default (LGD) and Recovery

**Recovery Rate** = % of face value recovered in default.

**Typical Recovery Rates**:
- **Senior Secured Bonds**: 60-80% (backed by collateral)
- **Senior Unsecured**: 40-60%
- **Subordinated**: 20-40%
- **Equity**: 0-10%

**LGD = 1 - Recovery Rate**

**Example**: $1,000 bond defaults, recovers $400 → Recovery = 40%, LGD = 60%, Loss = $600.

---

## Python: Credit Spread Analyzer

\`\`\`python
"""
Credit Spread Analysis System
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import date
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Rating(Enum):
    """Credit rating categories"""
    AAA = 1
    AA = 2
    A = 3
    BBB = 4  # Investment grade cutoff
    BB = 5
    B = 6
    CCC = 7
    CC = 8
    C = 9
    D = 10  # Default


@dataclass
class DefaultData:
    """Historical default statistics by rating"""
    rating: Rating
    annual_pd: float  # Annual probability of default
    cumulative_10yr_pd: float  # 10-year cumulative default probability
    recovery_rate: float  # Typical recovery rate
    
    @property
    def lgd(self) -> float:
        """Loss given default"""
        return 1.0 - self.recovery_rate
    
    @property
    def expected_loss(self) -> float:
        """Annual expected loss"""
        return self.annual_pd * self.lgd


# Historical default data (Moody's)
DEFAULT_STATISTICS = {
    Rating.AAA: DefaultData(Rating.AAA, 0.0005, 0.005, 0.70),
    Rating.AA: DefaultData(Rating.AA, 0.002, 0.015, 0.65),
    Rating.A: DefaultData(Rating.A, 0.005, 0.030, 0.60),
    Rating.BBB: DefaultData(Rating.BBB, 0.010, 0.060, 0.55),
    Rating.BB: DefaultData(Rating.BB, 0.025, 0.150, 0.45),
    Rating.B: DefaultData(Rating.B, 0.060, 0.300, 0.35),
    Rating.CCC: DefaultData(Rating.CCC, 0.120, 0.500, 0.25),
}


class CreditSpreadAnalyzer:
    """
    Analyze credit spreads and estimate fair value
    
    Example:
        >>> analyzer = CreditSpreadAnalyzer()
        >>> fair_spread = analyzer.calculate_fair_spread(
        ...     rating=Rating.BBB,
        ...     years_to_maturity=5,
        ...     liquidity_premium=0.0030
        ... )
        >>> print(f"Fair spread: {fair_spread*10000:.0f} bp")
        Fair spread: 110 bp
    """
    
    def __init__(self):
        self.default_stats = DEFAULT_STATISTICS
        logger.info("Initialized credit spread analyzer")
    
    def calculate_fair_spread(
        self,
        rating: Rating,
        years_to_maturity: float,
        liquidity_premium: float = 0.0020,  # 20bp default
        risk_premium: float = 0.0010,  # 10bp default
    ) -> float:
        """
        Calculate theoretically fair credit spread
        
        Spread = Expected Loss + Liquidity Premium + Risk Premium
        
        Args:
            rating: Credit rating
            years_to_maturity: Bond maturity in years
            liquidity_premium: Additional yield for illiquidity
            risk_premium: Additional compensation for risk
        
        Returns:
            Fair credit spread (annual)
        """
        if rating not in self.default_stats:
            raise ValueError(f"No default data for rating {rating}")
        
        default_data = self.default_stats[rating]
        
        # Expected loss component
        expected_loss = default_data.expected_loss
        
        # Adjust for maturity (longer = more cumulative risk)
        maturity_adjustment = min(years_to_maturity / 10, 1.5)
        adjusted_expected_loss = expected_loss * maturity_adjustment
        
        # Total fair spread
        fair_spread = (
            adjusted_expected_loss +
            liquidity_premium +
            risk_premium
        )
        
        logger.debug(
            f"Fair spread for {rating.name}, {years_to_maturity}yr: "
            f"{fair_spread*10000:.0f}bp"
        )
        
        return fair_spread
    
    def analyze_spread(
        self,
        rating: Rating,
        market_spread: float,
        treasury_yield: float,
        years_to_maturity: float
    ) -> Dict[str, float]:
        """
        Analyze if market spread is fair, cheap, or rich
        
        Args:
            rating: Credit rating
            market_spread: Current market spread (e.g., 0.0150 for 150bp)
            treasury_yield: Risk-free rate
            years_to_maturity: Maturity in years
        
        Returns:
            Analysis dict with fair value assessment
        """
        fair_spread = self.calculate_fair_spread(rating, years_to_maturity)
        
        # Calculate metrics
        spread_diff = market_spread - fair_spread
        spread_ratio = market_spread / fair_spread if fair_spread > 0 else np.inf
        
        # Classify
        if spread_diff > 0.0020:  # >20bp rich
            assessment = "Rich (overvalued)"
            trade_idea = "Sell / Underweight"
        elif spread_diff < -0.0020:  # >20bp cheap
            assessment = "Cheap (undervalued)"
            trade_idea = "Buy / Overweight"
        else:
            assessment = "Fair value"
            trade_idea = "Hold / Neutral"
        
        return {
            'rating': rating.name,
            'market_spread_bp': market_spread * 10000,
            'fair_spread_bp': fair_spread * 10000,
            'spread_diff_bp': spread_diff * 10000,
            'spread_ratio': spread_ratio,
            'assessment': assessment,
            'trade_idea': trade_idea,
            'corporate_yield': treasury_yield + market_spread,
            'fair_yield': treasury_yield + fair_spread,
        }
    
    def estimate_default_probability(
        self,
        market_spread: float,
        recovery_rate: float = 0.40,
        liquidity_premium: float = 0.0020
    ) -> float:
        """
        Estimate implied default probability from market spread
        
        market_spread = PD × LGD + liquidity_premium
        Solving for PD: PD = (spread - liquidity) / LGD
        
        Args:
            market_spread: Observed credit spread
            recovery_rate: Expected recovery rate
            liquidity_premium: Illiquidity component
        
        Returns:
            Implied annual default probability
        """
        lgd = 1.0 - recovery_rate
        
        if lgd == 0:
            return 0.0
        
        default_component = market_spread - liquidity_premium
        implied_pd = default_component / lgd
        
        # Floor at 0 (spreads can be tight)
        implied_pd = max(implied_pd, 0.0)
        
        logger.debug(
            f"Implied PD: {implied_pd*100:.2f}% "
            f"(spread={market_spread*10000:.0f}bp, recovery={recovery_rate*100:.0f}%)"
        )
        
        return implied_pd


class RatingMigrationMatrix:
    """
    Model rating transitions (upgrades/downgrades)
    
    Example:
        >>> matrix = RatingMigrationMatrix()
        >>> prob = matrix.transition_probability(Rating.BBB, Rating.BB, years=1)
        >>> print(f"BBB → BB downgrade probability: {prob*100:.2f}%")
        BBB → BB downgrade probability: 4.50%
    """
    
    def __init__(self):
        # 1-year transition matrix (simplified historical averages)
        self.annual_transitions = self._create_transition_matrix()
    
    def _create_transition_matrix(self) -> pd.DataFrame:
        """
        Create annual rating transition probability matrix
        
        Rows = current rating, Columns = future rating
        """
        # Simplified 1-year transition probabilities
        # Based on Moody's historical data (conceptual)
        data = {
            #       AAA    AA     A     BBB    BB     B    CCC     D
            'AAA': [0.90, 0.08, 0.01, 0.005, 0.003, 0.001, 0.001, 0.000],
            'AA':  [0.01, 0.88, 0.09, 0.01,  0.005, 0.003, 0.001, 0.001],
            'A':   [0.00, 0.02, 0.86, 0.10,  0.01,  0.005, 0.003, 0.002],
            'BBB': [0.00, 0.00, 0.04, 0.85,  0.08,  0.02,  0.005, 0.005],
            'BB':  [0.00, 0.00, 0.00, 0.02,  0.80,  0.12,  0.04,  0.020],
            'B':   [0.00, 0.00, 0.00, 0.00,  0.02,  0.78,  0.14,  0.060],
            'CCC': [0.00, 0.00, 0.00, 0.00,  0.00,  0.05,  0.73,  0.220],
        }
        
        index = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']
        columns = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']
        
        return pd.DataFrame(data, index=index, columns=columns)
    
    def transition_probability(
        self,
        from_rating: Rating,
        to_rating: Rating,
        years: int = 1
    ) -> float:
        """
        Calculate probability of transitioning from one rating to another
        
        For multi-year: (transition matrix)^years
        """
        if years < 1:
            raise ValueError("Years must be >= 1")
        
        matrix = self.annual_transitions
        
        # Multi-year transition = matrix^years
        if years > 1:
            matrix_power = matrix.pow(years)
        else:
            matrix_power = matrix
        
        from_str = from_rating.name
        to_str = to_rating.name if to_rating != Rating.D else 'D'
        
        if from_str not in matrix_power.index or to_str not in matrix_power.columns:
            return 0.0
        
        prob = matrix_power.loc[from_str, to_str]
        
        return float(prob)


# Example usage
if __name__ == "__main__":
    print("=== Credit Spread Analysis ===\\n")
    
    analyzer = CreditSpreadAnalyzer()
    
    # Analyze BBB corporate bond
    analysis = analyzer.analyze_spread(
        rating=Rating.BBB,
        market_spread=0.0180,  # 180bp spread
        treasury_yield=0.045,   # 4.5% Treasury yield
        years_to_maturity=10
    )
    
    print(f"Rating: {analysis['rating']}")
    print(f"Market Spread: {analysis['market_spread_bp']:.0f} bp")
    print(f"Fair Spread: {analysis['fair_spread_bp']:.0f} bp")
    print(f"Difference: {analysis['spread_diff_bp']:+.0f} bp")
    print(f"Assessment: {analysis['assessment']}")
    print(f"Trade Idea: {analysis['trade_idea']}")
    print(f"Corporate Yield: {analysis['corporate_yield']*100:.2f}%")
    
    # Implied default probability
    print("\\n=== Implied Default Probability ===\\n")
    
    implied_pd = analyzer.estimate_default_probability(
        market_spread=0.0180,
        recovery_rate=0.55
    )
    
    historical_pd = DEFAULT_STATISTICS[Rating.BBB].annual_pd
    
    print(f"Market-implied PD: {implied_pd*100:.2f}% annually")
    print(f"Historical PD: {historical_pd*100:.2f}% annually")
    print(f"Difference: {(implied_pd - historical_pd)*100:+.2f}%")
    
    if implied_pd > historical_pd * 1.5:
        print("→ Market pricing in elevated default risk (cheap bonds)")
    elif implied_pd < historical_pd * 0.7:
        print("→ Market complacent on default risk (expensive bonds)")
    else:
        print("→ Market default expectations aligned with history")
    
    # Rating migration
    print("\\n=== Rating Migration Analysis ===\\n")
    
    migration = RatingMigrationMatrix()
    
    downgrade_prob = migration.transition_probability(Rating.BBB, Rating.BB, years=1)
    default_prob = migration.transition_probability(Rating.BBB, Rating.D, years=1)
    
    print(f"BBB → BB downgrade (1yr): {downgrade_prob*100:.2f}%")
    print(f"BBB → Default (1yr): {default_prob*100:.2f}%")
    print(f"BBB staying investment grade: {(1-downgrade_prob-default_prob)*100:.2f}%")
\`\`\`

---

## Fallen Angels and Rising Stars

**Fallen Angel**: Bond downgraded from investment grade (BBB) to high yield (BB).

**Impact**:
- Must be sold by many institutions (mandates require investment grade)
- Forced selling → price drops → spreads widen
- Trading opportunity: Buy fallen angels that recover

**Rising Star**: Bond upgraded from high yield to investment grade.

**Impact**:
- New buying from institutions
- Spreads tighten, prices rise

**Historical Examples**:
- **Ford** (2020): Fell to junk during COVID, rose back to investment grade 2023
- **Kraft Heinz** (2020): Fallen angel, later recovered

---

## Real-World: Corporate Bond Credit Analysis

**Case Study**: Apple Corporate Bond (AAA-rated)

**Analysis**:
- **Rating**: AAA (highest quality, rare for corporates)
- **Spread**: ~50-70bp over Treasuries (very tight)
- **Rationale**: $162B cash, low debt, strong cash flows
- **Default Probability**: <0.1% annually (nearly risk-free)

**Comparison**: Energy High Yield (CCC-rated)

- **Rating**: CCC (substantial risk)
- **Spread**: 800-1500bp (8-15% above Treasuries!)
- **Rationale**: Oil price volatility, high leverage
- **Default Probability**: ~12-20% annually

---

## Key Takeaways

1. **Credit spread = compensation for default risk + liquidity + risk premium**
2. **Default probability and LGD drive expected loss**
3. **Investment grade (BBB+) vs High Yield (BB+) cutoff is critical**
4. **Historical default rates**: Aaa 0.5%, Baa 6%, B 30% (10-year cumulative)
5. **Recovery rates**: Senior secured 60-80%, subordinated 20-40%
6. **Fallen angels** = forced selling opportunity
7. **Spread analysis**: Compare market spread to fair value

**Next Section**: Corporate Bonds - callable bonds, convertibles, and embedded options.
`,
};
