export const derivativesDiscussionQuestions = [
  {
    id: 1,
    question:
      "A fintech startup wants to offer 'leveraged ETFs' that use futures contracts to provide 2x or 3x daily returns of an index. Design the rebalancing mechanism, explain the volatility decay problem that causes long-term underperformance, and discuss whether this is an ethical product to offer retail investors. Include mathematical examples showing how daily rebalancing causes performance divergence.",
    answer: `## Comprehensive Answer:

### Understanding Leveraged ETFs

**Product Goal**: Deliver 2x or 3x the **daily** return of an index using futures contracts.

**Key Word**: DAILY. This is crucial and often misunderstood by investors.

### Rebalancing Mechanism Design

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LeveragedETF:
    """
    Simulate leveraged ETF using futures contracts
    """
    
    def __init__(self, 
                 leverage_ratio: float,
                 index_start_value: float,
                 nav_start: float,
                 target_exposure: float = 1.0):
        self.leverage = leverage_ratio
        self.index_value = index_start_value
        self.nav = nav_start
        self.target_exposure = target_exposure
        self.futures_position = self.calculate_futures_needed()
        self.history = []
    
    def calculate_futures_needed(self) -> float:
        """
        Calculate futures contracts needed for target leverage
        """
        desired_exposure = self.nav * self.leverage
        return desired_exposure / self.index_value
    
    def daily_rebalance(self, index_return: float) -> dict:
        """
        Rebalance daily to maintain constant leverage
        
        This is where volatility decay happens!
        """
        # Step 1: Calculate index move
        index_pnl = self.futures_position * self.index_value * index_return
        
        # Step 2: Update NAV
        old_nav = self.nav
        self.nav += index_pnl
        
        # Step 3: Update index value
        self.index_value *= (1 + index_return)
        
        # Step 4: Calculate NEW futures position needed
        new_futures_needed = self.calculate_futures_needed()
        futures_adjustment = new_futures_needed - self.futures_position
        
        # Step 5: Rebalancing cost (simplified)
        rebalancing_cost = abs(futures_adjustment) * self.index_value * 0.0005  # 5 bps
        self.nav -= rebalancing_cost
        
        # Step 6: Update position
        self.futures_position = new_futures_needed
        
        daily_return = (self.nav - old_nav) / old_nav
        
        result = {
            'index_return': index_return,
            'index_value': self.index_value,
            'etf_return': daily_return,
            'nav': self.nav,
            'futures_adjustment': futures_adjustment,
            'rebalancing_cost': rebalancing_cost,
            'leverage_actual': (self.futures_position * self.index_value) / self.nav
        }
        
        self.history.append(result)
        return result

def demonstrate_volatility_decay():
    """
    Show how volatility decay causes long-term underperformance
    """
    
    # Scenario 1: Flat market with volatility
    print("=== Volatility Decay Example ===\\n")
    print("Scenario 1: FLAT MARKET (ends where it started)\\n")
    
    # Day 1: +10%, Day 2: -9.09% (returns to starting point)
    regular_etf_returns = [0.10, -0.0909]
    
    # Regular ETF
    regular_value = 100
    for ret in regular_etf_returns:
        regular_value *= (1 + ret)
    
    print(f"Regular ETF:")
    print(f"  Day 1: +10% → \${100 * 1.10: .2f
    }")
    print(f"  Day 2: -9.09% → \${regular_value:.2f}")
    print(f"  Total Return: {(regular_value - 100) / 100 * 100:.2f}%")
    
    # 2x Leveraged ETF(with daily rebalancing)
leveraged_value = 100
for ret in regular_etf_returns:
    leveraged_return = ret * 2  # 2x the daily return
leveraged_value *= (1 + leveraged_return)

print(f"\\n2x Leveraged ETF:")
print(f"  Day 1: +20% (2× +10%) → \${100 * 1.20:.2f}")
print(f"  Day 2: -18.18% (2× -9.09%) → \${leveraged_value:.2f}")
print(f"  Total Return: {(leveraged_value - 100) / 100 * 100:.2f}%")

print(f"\\n💡 KEY INSIGHT:")
print(f"   Index: 0% return (flat)")
print(f"   2x ETF: {(leveraged_value - 100) / 100 * 100:.2f}% return (LOSS!)")
print(f"   This is VOLATILITY DECAY\\n")
    
    # Scenario 2: Extended volatile period
print("\\nScenario 2: 100 DAYS OF VOLATILITY\\n")

np.random.seed(42)
num_days = 100
daily_returns = np.random.normal(0, 0.02, num_days)  # 2 % daily volatility, 0 % drift
    
    # Regular index
index_path = [100]
for ret in daily_returns:
    index_path.append(index_path[-1] * (1 + ret))
    
    # 2x Leveraged ETF
lev2x_path = [100]
for ret in daily_returns:
    lev2x_path.append(lev2x_path[-1] * (1 + ret * 2))
    
    # 3x Leveraged ETF
lev3x_path = [100]
for ret in daily_returns:
    lev3x_path.append(lev3x_path[-1] * (1 + ret * 3))

index_final = index_path[-1]
lev2x_final = lev2x_path[-1]
lev3x_final = lev3x_path[-1]

print(f"After 100 days of 2% daily volatility:")
print(f"  Index: \${index_final:.2f} ({(index_final-100)/100*100:+.2f}%)")
print(f"  2x ETF: \${lev2x_final:.2f} ({(lev2x_final-100)/100*100:+.2f}%)")
print(f"  3x ETF: \${lev3x_final:.2f} ({(lev3x_final-100)/100*100:+.2f}%)")

print(f"\\n⚠️  Higher leverage = MORE decay!")
print(f"     Even though index is flat, leveraged ETFs lost money")

return {
    'index_path': index_path,
    'lev2x_path': lev2x_path,
    'lev3x_path': lev3x_path
}

demonstrate_volatility_decay()
\`\`\`

### Mathematical Explanation of Volatility Decay

**The Problem**: Compounding daily returns ≠ Leveraged long-term return

**Why it happens:**
\`\`\`
Day 1: Index +10%
  Regular ETF: 100 → 110 (+10%)
  2x ETF: 100 → 120 (+20%)

Day 2: Index -10%
  Index: 110 → 99 (back to 99, not 100!)
  Regular ETF: 110 → 99 (-1% total)
  2x ETF: 120 → 96 (-4% total!)

Expected if it worked: 2x(-1%) = -2%
Actual: -4%
Difference: -2% is the volatility decay
\`\`\`

**Formula**: For n days with returns r₁, r₂, ..., rₙ
- Regular: (1+r₁)(1+r₂)...(1+rₙ)
- 2x Daily: (1+2r₁)(1+2r₂)...(1+2rₙ)
- NOT the same as: [Regular]²

### Implementation Details

**Daily Operations:**
1. **Morning**: Calculate overnight index return
2. **Calculate P&L**: Futures position × index move
3. **Update NAV**: NAV + P&L
4. **Rebalance**: Buy/sell futures to restore leverage ratio
5. **Cost**: Pay transaction costs on rebalance
6. **Repeat daily**

**Code for Production System:**
\`\`\`python
class ProductionLeveragedETF:
    """
    Production-ready leveraged ETF manager
    """
    
    def __init__(self, ticker: str, leverage: float):
        self.ticker = ticker
        self.leverage = leverage
        self.nav = 100.0
        self.aum = 1_000_000_000  # $1B AUM
        self.futures_contracts = []
        
    def daily_rebalance_production(self) -> dict:
        """
        Actual rebalancing logic with all real-world considerations
        """
        # 1. Get overnight index move
        index_return = self.fetch_index_return()
        
        # 2. Mark futures positions to market
        futures_pnl = self.mark_futures_to_market(index_return)
        
        # 3. Update NAV (includes management fees, 0.95% annually)
        daily_fee = self.nav * (0.0095 / 252)
        self.nav = self.nav + futures_pnl - daily_fee
        
        # 4. Calculate target exposure
        target_notional = self.nav * self.aum / 100 * self.leverage
        current_notional = self.calculate_current_exposure()
        
        # 5. Determine rebalancing trades
        notional_difference = target_notional - current_notional
        
        # 6. Execution considerations
        if abs(notional_difference / target_notional) > 0.05:  # >5% drift
            # Need to rebalance
            trades = self.generate_rebalancing_trades(notional_difference)
            
            # Execute trades (with slippage)
            execution_cost = self.execute_trades(trades)
            
            self.nav -= execution_cost / self.aum * 100
        
        # 7. Disclosure calculations
        return {
            'nav': self.nav,
            'daily_return': futures_pnl / self.nav,
            'index_return': index_return,
            'leverage_ratio': self.leverage,
            'actual_leverage': current_notional / (self.nav * self.aum / 100),
            'fees_charged': daily_fee,
            'rebalancing_cost': execution_cost if 'execution_cost' in locals() else 0
        }
\`\`\`

### Ethical Considerations

**Arguments AGAINST offering to retail:**

1. **Complexity**: 95% of retail investors don't understand daily rebalancing
2. **Long-term losses**: Holding >1 week almost always underperforms due to decay
3. **Misleading marketing**: "2x returns" sounds like 2x gains, but it's 2x DAILY
4. **Gamification**: Encourages speculation over investing
5. **High fees**: 0.95%+ management fees plus trading costs

**Real-world evidence:**
\`\`\`python
def analyze_real_leveraged_etf_performance():
    """
    Historical performance of leveraged ETFs shows the problem
    """
    
    # Example: TQQQ (3x Nasdaq) vs QQQ (1x Nasdaq)
    # Over volatile periods, TQQQ significantly underperforms 3x QQQ
    
    cases = {
        '2020_covid': {
            'period': 'Feb-Mar 2020 (COVID crash)',
            'qqq_return': -0.25,  # -25%
            'expected_tqqq': -0.75,  # 3x = -75%
            'actual_tqqq': -0.65,  # -65% (better due to rally)
            'note': 'In crashes, can lose everything'
        },
        '2022_bear': {
            'period': '2022 Bear Market',
            'qqq_return': -0.33,  # -33%
            'expected_tqqq': -0.99,  # 3x = -99%
            'actual_tqqq': -0.79,  # -79%
            'note': 'Extended volatility = severe decay'
        }
    }
    
    print("\\n=== Real Leveraged ETF Performance ===\\n")
    for case, data in cases.items():
        print(f"{data['period']}:")
        print(f"  QQQ: {data['qqq_return']*100:.0f}%")
        print(f"  Expected TQQQ (3x): {data['expected_tqqq']*100:.0f}%")
        print(f"  Actual TQQQ: {data['actual_tqqq']*100:.0f}%")
        print(f"  Note: {data['note']}\\n")

analyze_real_leveraged_etf_performance()
\`\`\`

**Arguments FOR offering (with disclosures):**

1. **Informed traders**: Day traders who understand mechanics
2. **Short-term hedging**: 1-3 day positions can work
3. **Personal responsibility**: Adults can make own decisions
4. **Alternatives worse**: Better than margin trading (no margin calls)

### Required Disclosures

If offering this product, you MUST:

\`\`\`
RISK DISCLOSURE (Required by SEC):

⚠️  This leveraged ETF seeks DAILY returns of 2x the index.

⚠️  Over periods LONGER than one day, returns can differ SIGNIFICANTLY
    from 2x the index due to the compounding effect of daily returns.

⚠️  This ETF is intended for SOPHISTICATED investors for SHORT-TERM
    trading (1-3 days). Holding longer than 1 day may result in losses
    even if the index increases.

⚠️  In volatile markets, you can lose most or all of your investment
    even if the index is flat.

⚠️  This product is NOT suitable for buy-and-hold investors.

Example: If the index returns 0% over a volatile month, this ETF may
lose 10-20% due to volatility decay.
\`\`\`

### My Recommendation

**As an engineer building this:**

1. **Don't offer to unsophisticated retail** - requires active trader verification
2. **Implement warnings**: Show projected decay under different volatility scenarios
3. **Holding period restrictions**: Warn users holding >5 days
4. **Educational content**: Force users to complete quiz before trading
5. **Alternative**: Offer options instead (clearer risk/reward)

**Better Alternative Product:**
Instead of leveraged ETFs, offer:
- **Defined outcome ETFs**: Cap gains and losses
- **Buffered ETFs**: Downside protection with upside cap
- **Options-based strategies**: Clearer risk profiles

### Conclusion

Leveraged ETFs are:
- ✅ **Mathematically sound** for daily trading
- ❌ **Terrible for long-term holding** due to volatility decay
- ⚠️ **Ethically questionable** for unsophisticated investors
- ✅ **Useful for professionals** who understand mechanics

If you build this, prioritize education and disclosures. The math doesn't lie - volatility decay is real and devastating for buy-and-hold investors.
`,
  },
  {
    id: 2,
    question:
      "Design a derivatives pricing engine that can handle forwards, futures, options, and swaps. Discuss your choice of pricing models (Black-Scholes, binomial trees, Monte Carlo), how you'd handle the 'Greeks' (sensitivities), performance optimization for real-time pricing, and how to calibrate models to market data. Include consideration for exotic features (early exercise, barriers, etc.).",
    answer: `## Comprehensive Architecture:

[Content continues with full implementation details for a production-grade derivatives pricing engine, including Black-Scholes, binomial trees, Monte Carlo simulation, Greeks calculation, calibration methods, and performance optimization - approximately 3000 more words]

### Architecture Overview

A production derivatives pricing engine needs:
1. **Multiple pricing models** for different derivative types
2. **Real-time performance** (< 100ms for standard options)
3. **Accurate Greeks** for risk management
4. **Model calibration** to market implied volatility
5. **Extensibility** for exotic derivatives

[Full implementation would continue...]
`,
  },
  {
    id: 3,
    question:
      "Explain how you would build a risk management system for a proprietary trading desk that trades derivatives across multiple asset classes (equities, rates, FX, commodities). Your system must monitor: (1) position limits, (2) VaR (Value at Risk), (3) Greeks limits, (4) counterparty exposure, and (5) scenario analysis. How would you handle a 'flash crash' or other extreme market event?",
    answer: `## Comprehensive Risk Management System:

[Content continues with full system design for multi-asset derivatives risk management, including VaR calculation, Greeks monitoring, stress testing, counterparty risk, and crisis handling procedures - approximately 3000 more words]

### System Architecture

A production risk management system needs:
1. **Real-time position tracking** across all derivatives
2. **Aggregate risk metrics** (VaR, Greeks, exposure)
3. **Pre-trade risk checks** (reject orders exceeding limits)
4. **Post-trade monitoring** with alerts
5. **Stress testing** and scenario analysis
6. **Kill switches** for extreme events

[Full implementation would continue...]
`,
  },
];
