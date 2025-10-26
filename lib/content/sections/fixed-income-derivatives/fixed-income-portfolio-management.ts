export const fixedIncomePortfolioManagement = {
    title: 'Fixed Income Portfolio Management',
    id: 'fixed-income-portfolio-management',
    content: `
# Fixed Income Portfolio Management

## Introduction

Fixed income portfolio management involves constructing and managing bond portfolios to achieve specific objectives (income, capital preservation, total return).

**Why critical for engineers**:
- $120+ trillion global bond market
- Algorithmic portfolio construction and rebalancing
- Performance attribution analysis
- Risk budgeting and optimization

**What you'll build**: Portfolio optimizer, benchmark tracker, performance attribution system, rebalancing engine.

---

## Investment Objectives

### Income Focus

**Goal**: Generate stable cash flow from coupon payments.

**Strategy**:
- High-coupon bonds
- Longer maturities (higher yields typically)
- Investment-grade focus (reliable payments)

**Metrics**: Current yield, yield-to-maturity, coupon rate

### Total Return

**Goal**: Maximize total return (coupons + price appreciation).

**Strategy**:
- Active duration management
- Credit spread trading
- Yield curve positioning

**Metrics**: Total return, Sharpe ratio, information ratio

### Capital Preservation

**Goal**: Protect principal value.

**Strategy**:
- Short duration (minimize rate risk)
- High credit quality (AAA/AA)
- Diversification

**Metrics**: Maximum drawdown, volatility, downside deviation

---

## Active vs Passive Management

### Passive (Index Tracking)

**Approach**: Replicate benchmark index.

**Example**: Bloomberg Barclays Aggregate Bond Index
- 10,000+ bonds
- Stratified sampling (hold representative subset)
- Minimize tracking error

**Advantages**:
- Low costs (0.05-0.15% annual fee)
- Transparent
- Tax efficient

**Challenges**:
- Tracking error (can't hold all bonds)
- Rebalancing costs
- Corporate actions (calls, maturities)

### Active Management

**Approach**: Outperform benchmark through security selection and timing.

**Strategies**:
1. **Duration management**: Adjust portfolio duration vs benchmark
2. **Yield curve positioning**: Bullets, barbells, ladders
3. **Sector rotation**: Shift between Treasuries, corporates, agencies
4. **Credit selection**: Pick undervalued bonds
5. **Carry trade**: Buy high-yield, fund at low rates

**Target**: Outperform benchmark by 0.50-1.50% annually (before fees)

**Risk**: Tracking error 1-3%, potential underperformance

---

## Portfolio Strategies

### Bullet Strategy

**Structure**: Concentrate bonds around single maturity.

**Example**:
\`\`\`
All bonds mature in 7-8 years
Purpose: Match specific liability
Advantage: Precise timing, immunization
Disadvantage: Reinvestment risk at maturity
\`\`\`

### Barbell Strategy

**Structure**: Short and long maturities, nothing in middle.

**Example**:
\`\`\`
50% in 2-year bonds
50% in 30-year bonds
Average duration: 16 years

Advantage: Convexity (more price appreciation if rates fall)
Disadvantage: Lower yield than bullet of same duration
\`\`\`

### Ladder Strategy

**Structure**: Equal weights across maturities.

**Example**:
\`\`\`
10% each in 1, 2, 3, 4, 5, 6, 7, 8, 9, 10-year bonds

Advantage: Diversification, regular maturities
Disadvantage: Complex to maintain
\`\`\`

---

## Performance Attribution

**Question**: Why did portfolio outperform/underperform benchmark?

Performance attribution decomposes returns into sources of alpha to understand skill vs luck.

### Attribution Components

**1. Duration Effect**:
\`\`\`
Duration_effect = (Portfolio_duration - Benchmark_duration) × Δ_yields × -1

Example:
Portfolio duration: 7.5 years
Benchmark duration: 7.0 years
Yield change: -50bp (-0.50%)

Duration effect = (7.5 - 7.0) × (-0.005) × (-1) = +0.25%
Portfolio outperformed by 25bp due to longer duration in falling rate environment
\`\`\`

**2. Curve Effect**:
\`\`\`
Measures impact of non-parallel yield curve shifts

Example: Curve steepening (2s10s widens +30bp)
- Portfolio: Barbell strategy (2yr + 30yr)
- Benchmark: Bullet (10yr concentrated)
- Impact: -15bp (barbell underperforms in steepening)
\`\`\`

**3. Sector Effect**:
\`\`\`
Overweight/underweight sectors relative to benchmark

Sector_effect = Σ(w_p - w_b) × R_sector

Where:
w_p = portfolio weight in sector
w_b = benchmark weight
R_sector = sector return

Example:
Corporates: 40% portfolio vs 30% benchmark, +5% return
Sector effect = (0.40 - 0.30) × 0.05 = +0.50%
\`\`\`

**4. Selection Effect**:
\`\`\`
Bond picking skill within sectors

Selection_effect = Σ w_p × (R_bond - R_sector)

Example:
Within corporates sector (+5% avg):
Selected bonds returned +5.8%
Selection effect = 0.40 × (0.058 - 0.050) = +0.32%
\`\`\`

**5. Carry Effect**:
\`\`\`
Coupon income + rolldown gain

Carry = Coupon yield + Rolldown return

Rolldown: As bond ages, yield decreases (if upward-sloping curve)
Example: Buy 10yr at 5% yield, hold 1yr → now 9yr bond at 4.8% yield
Rolldown gain: Price appreciation from yield compression
\`\`\`

**Full Example**:
\`\`\`
Portfolio return: +7.20%
Benchmark return: +6.00%
Outperformance: +1.20%

Attribution:
1. Duration: +0.40% (overweight duration, rates fell 50bp)
2. Curve: -0.10% (barbell hurt by steepening)
3. Sector: +0.30% (overweight corporates, spreads tightened 20bp)
4. Selection: +0.50% (picked bonds that beat sector by 80bp)
5. Carry: +0.10% (higher coupon than benchmark)

Total: +1.20% ✓

Insight: 42% of alpha (0.50/1.20) from security selection (skill)
        33% from duration positioning (tactical bet)
        25% from sector allocation (strategic overweight)
\`\`\`

---

## Python: Performance Attribution System

\`\`\`python
"""
Fixed Income Performance Attribution Engine
"""
from typing import Dict, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Bond position in portfolio"""
    cusip: str
    sector: str
    market_value: float
    duration: float
    ytm: float
    return_pct: float  # Period return


class PerformanceAttribution:
    """
    Decompose portfolio returns vs benchmark
    
    Example:
        >>> portfolio_positions = [
        ...     Position('AAPL', 'Corporate', 1_000_000, 7.5, 0.05, 0.062),
        ...     Position('TSRY', 'Treasury', 2_000_000, 8.0, 0.045, 0.055)
        ... ]
        >>> benchmark_positions = [...]
        >>> attribution = PerformanceAttribution(portfolio_positions, benchmark_positions)
        >>> results = attribution.analyze()
        >>> print(f"Duration effect: {results['duration_effect']*100:.2f}%")
    """
    
    def __init__(
        self,
        portfolio_positions: List[Position],
        benchmark_positions: List[Position],
        yield_change: float  # Overall yield change (e.g., -0.005 for -50bp)
    ):
        self.portfolio = portfolio_positions
        self.benchmark = benchmark_positions
        self.yield_change = yield_change
        
        # Calculate totals
        self.portfolio_value = sum(p.market_value for p in portfolio_positions)
        self.benchmark_value = sum(p.market_value for p in benchmark_positions)
        
        logger.info(
            f"Initialized attribution: "
            f"Portfolio ${self.portfolio_value / 1e6: .1f
}M, "
            f"Benchmark ${self.benchmark_value/1e6:.1f}M"
        )
    
    def duration_effect(self) -> float:
"""
        Calculate alpha from duration positioning

Duration_effect = (Port_duration - Bench_duration) × Δy × -1
"""
        # Portfolio duration(value - weighted)
port_duration = sum(
    p.market_value * p.duration for p in self.portfolio
) / self.portfolio_value
        
        # Benchmark duration
bench_duration = sum(
    p.market_value * p.duration for p in self.benchmark
) / self.benchmark_value
        
        # Effect
effect = (port_duration - bench_duration) * self.yield_change * (-1)

logger.debug(
    f"Duration effect: Port={port_duration:.2f}, Bench={bench_duration:.2f}, "
            f"Effect={effect*100:.2f}%"
)

return effect
    
    def sector_effect(self) -> Dict[str, float]:
"""
        Calculate alpha from sector allocation
        
        For each sector: (w_p - w_b) × R_sector
"""
        # Group by sector
port_sectors = self._group_by_sector(self.portfolio, self.portfolio_value)
bench_sectors = self._group_by_sector(self.benchmark, self.benchmark_value)

sector_effects = {}
all_sectors = set(port_sectors.keys()) | set(bench_sectors.keys())

for sector in all_sectors:
    port_data = port_sectors.get(sector, { 'weight': 0, 'return': 0 })
bench_data = bench_sectors.get(sector, { 'weight': 0, 'return': 0 })
            
            # Sector return (use benchmark sector return for allocation effect)
    sector_return = bench_data['return']
            
            # Weight difference
weight_diff = port_data['weight'] - bench_data['weight']
            
            # Sector effect
effect = weight_diff * sector_return
sector_effects[sector] = effect

logger.debug(
    f"Sector {sector}: Weight diff {weight_diff*100:+.1f}%, "
                f"Return {sector_return*100:.2f}%, Effect {effect*100:+.2f}%"
)

return sector_effects
    
    def selection_effect(self) -> Dict[str, float]:
"""
        Calculate alpha from security selection within sectors
        
        For each sector: w_p × (R_port_sector - R_bench_sector)
"""
port_sectors = self._group_by_sector(self.portfolio, self.portfolio_value)
bench_sectors = self._group_by_sector(self.benchmark, self.benchmark_value)

selection_effects = {}

for sector in port_sectors:
    if sector not in bench_sectors:
continue

port_return = port_sectors[sector]['return']
bench_return = bench_sectors[sector]['return']
port_weight = port_sectors[sector]['weight']
            
            # Selection effect
effect = port_weight * (port_return - bench_return)
selection_effects[sector] = effect

logger.debug(
    f"Selection {sector}: Port return {port_return*100:.2f}%, "
                f"Bench return {bench_return*100:.2f}%, Effect {effect*100:+.2f}%"
)

return selection_effects
    
    def _group_by_sector(
    self,
    positions: List[Position],
    total_value: float
) -> Dict[str, Dict]:
"""Group positions by sector and calculate sector stats"""
sectors = {}

for pos in positions:
    if pos.sector not in sectors:
sectors[pos.sector] = {
    'total_value': 0,
    'weighted_return': 0,
}

sectors[pos.sector]['total_value'] += pos.market_value
sectors[pos.sector]['weighted_return'] += pos.market_value * pos.return_pct
        
        # Calculate weights and returns
for sector, data in sectors.items():
    data['weight'] = data['total_value'] / total_value
data['return'] = data['weighted_return'] / data['total_value']

return sectors
    
    def analyze(self) -> Dict:
"""
        Full attribution analysis

Returns:
            Dict with all attribution components
"""
        # Calculate returns
port_return = sum(
    p.market_value * p.return_pct for p in self.portfolio
) / self.portfolio_value

bench_return = sum(
    p.market_value * p.return_pct for p in self.benchmark
) / self.benchmark_value

active_return = port_return - bench_return
        
        # Attribution components
duration_eff = self.duration_effect()
sector_effs = self.sector_effect()
selection_effs = self.selection_effect()

total_sector_eff = sum(sector_effs.values())
total_selection_eff = sum(selection_effs.values())
        
        # Residual(unexplained)
explained = duration_eff + total_sector_eff + total_selection_eff
residual = active_return - explained

results = {
    'portfolio_return': port_return,
    'benchmark_return': bench_return,
    'active_return': active_return,
    'duration_effect': duration_eff,
    'sector_effects': sector_effs,
    'total_sector_effect': total_sector_eff,
    'selection_effects': selection_effs,
    'total_selection_effect': total_selection_eff,
    'residual': residual,
}

logger.info(
    f"Attribution: Active={active_return*100:.2f}%, "
            f"Duration={duration_eff*100:.2f}%, "
            f"Sector={total_sector_eff*100:.2f}%, "
            f"Selection={total_selection_eff*100:.2f}%"
)

return results
    
    def report(self) -> pd.DataFrame:
"""Generate attribution report as DataFrame"""
results = self.analyze()

data = {
    'Component': [
        'Portfolio Return',
        'Benchmark Return',
        '--- Active Return ---',
        'Duration Effect',
        'Sector Effect',
        'Selection Effect',
        'Residual',
    ],
    'Return (%)': [
        results['portfolio_return'] * 100,
        results['benchmark_return'] * 100,
        results['active_return'] * 100,
        results['duration_effect'] * 100,
        results['total_sector_effect'] * 100,
        results['total_selection_effect'] * 100,
        results['residual'] * 100,
    ]
}

return pd.DataFrame(data)


# Example usage
if __name__ == "__main__":
    print("=== Performance Attribution Analysis ===\\n")
    
    # Portfolio positions
portfolio = [
    Position('AAPL_2030', 'Corporate', 1_000_000, 7.5, 0.050, 0.062),
    Position('TSRY_2032', 'Treasury', 2_000_000, 8.0, 0.045, 0.055),
    Position('XOM_2028', 'Corporate', 500_000, 6.5, 0.055, 0.048),
]
    
    # Benchmark positions
benchmark = [
    Position('BENCH_CORP', 'Corporate', 1_200_000, 7.0, 0.051, 0.058),
    Position('BENCH_TSRY', 'Treasury', 1_800_000, 7.5, 0.046, 0.052),
]
    
    # Market environment: rates fell 50bp
yield_change = -0.005
    
    # Run attribution
attribution = PerformanceAttribution(portfolio, benchmark, yield_change)
results = attribution.analyze()
    
    # Display report
report = attribution.report()
print(report.to_string(index = False))

print("\\n=== Detailed Sector Effects ===\\n")
for sector, effect in results['sector_effects'].items():
    print(f"{sector:15}: {effect*100:+.2f}%")

print("\\n=== Detailed Selection Effects ===\\n")
for sector, effect in results['selection_effects'].items():
    print(f"{sector:15}: {effect*100:+.2f}%")
\`\`\`

---

## Risk Management

### Tracking Error

**Definition**: Standard deviation of portfolio returns vs benchmark returns.

\`\`\`
TE = StdDev(Portfolio_return - Benchmark_return)

Target: <1% for enhanced index, 2-4% for active
\`\`\`

### VaR (Value at Risk)

**Definition**: Maximum loss at confidence level over time horizon.

\`\`\`
95% 1-day VaR = $1M
Interpretation: 95% confident loss won't exceed $1M tomorrow
\`\`\`

### Scenario Analysis

Test portfolio under stress scenarios:
- Rates +200bp
- Credit spreads widen 100bp
- Curve inverts
- Sector rotation (flight to quality)

---

## Key Takeaways

1. **Objectives**: Income (high coupons), total return (active management), capital preservation (short duration)
2. **Active vs Passive**: Passive (low cost, track index), Active (outperform, higher fees, tracking error)
3. **Strategies**: Bullet (single maturity), Barbell (short+long), Ladder (equal weights across maturities)
4. **Performance attribution**: Duration, curve, sector, selection, carry effects explain returns
5. **Risk management**: Tracking error, VaR, scenario analysis monitor portfolio risk
6. **Benchmarks**: Bloomberg Barclays Agg most common, customize for specific mandates

**Next Section**: Derivative Risk Management - VaR, stress testing, risk limits, compliance.
`,
};

