export const liquidityRisk = {
  id: 'liquidity-risk',
  title: 'Liquidity Risk',
  content: `
# Liquidity Risk

## Introduction

"Liquidity is like oxygen - you don't notice it until it's gone, then it's the only thing that matters."

Liquidity risk - the risk of being unable to meet obligations when due - has killed more financial institutions than market losses. You can survive being wrong about market direction. You cannot survive running out of cash.

**Historical Liquidity Crises**:
- **Northern Rock (2007)**: First UK bank run in 150 years, couldn't roll short-term funding
- **Lehman Brothers (2008)**: Had assets but couldn't convert to cash fast enough
- **Money Market Funds (2008)**: "Breaking the buck" caused panic redemptions
- **Archegos (2021)**: Margin calls led to $10B+ forced liquidation

## Two Types of Liquidity Risk

### 1. Funding Liquidity Risk

**The risk of being unable to meet cash obligations.**

\`\`\`
Can you pay your bills when they come due?
\`\`\`

Examples:
- Unable to roll commercial paper
- Margin calls you can't meet
- Depositor withdrawals exceeding reserves
- Bond maturity you can't refinance

### 2. Market Liquidity Risk

**The risk of being unable to exit positions without significant price impact.**

\`\`\`
Can you sell your assets at reasonable prices?
\`\`\`

Examples:
- Bid-ask spreads widening
- Order book depth disappearing
- No buyers at any reasonable price
- Fire sale prices

## The Liquidity Spiral

These two risks interact dangerously:

\`\`\`
Funding pressure → Must sell assets → Market illiquidity → 
Fire sale prices → Losses → More funding pressure → Death spiral
\`\`\`

\`\`\`python
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

class LiquidityRiskManager:
    """
    Manage funding and market liquidity risk
    """
    def __init__(self, 
                 cash_balance: float,
                 assets: Dict[str, Dict],
                 liabilities: Dict[str, Dict]):
        """
        Args:
            cash_balance: Current cash
            assets: Dict of {asset_name: {value, liquidity_class, haircut}}
            liabilities: Dict of {liability_name: {amount, maturity_days}}
        """
        self.cash = cash_balance
        self.assets = assets
        self.liabilities = liabilities
        
    def calculate_liquidity_coverage_ratio(self) -> Dict:
        """
        LCR = High Quality Liquid Assets / Net Cash Outflows (30 days)
        
        Basel III requires LCR ≥ 100%
        """
        # HQLA (High Quality Liquid Assets)
        level_1_assets = 0  # Cash, central bank reserves, sovereigns
        level_2a_assets = 0  # High quality corporates, covered bonds
        level_2b_assets = 0  # Lower quality corporates, equities
        
        for asset_name, asset_info in self.assets.items():
            value = asset_info['value']
            liquidity_class = asset_info.get('liquidity_class', 'Level 2B')
            haircut = asset_info.get('haircut', 0.50)
            
            if liquidity_class == 'Level 1':
                level_1_assets += value  # No haircut
            elif liquidity_class == 'Level 2A':
                level_2a_assets += value * (1 - 0.15)  # 15% haircut
            elif liquidity_class == 'Level 2B':
                level_2b_assets += value * (1 - 0.50)  # 50% haircut
        
        # Total HQLA (with limits on Level 2)
        hqla = level_1_assets + min(level_2a_assets + level_2b_assets, level_1_assets * 0.67)
        
        # Net cash outflows (30-day stress)
        outflows_30day = self._calculate_stressed_outflows(30)
        inflows_30day = self._calculate_expected_inflows(30)
        net_outflows = max(outflows_30day - inflows_30day * 0.75, 0)  # Cap inflows at 75% of outflows
        
        lcr = (hqla / net_outflows * 100) if net_outflows > 0 else float('inf')
        
        return {
            'hqla': hqla,
            'level_1_assets': level_1_assets,
            'level_2a_assets': level_2a_assets,
            'level_2b_assets': level_2b_assets,
            'net_cash_outflows_30d': net_outflows,
            'lcr_ratio': lcr,
            'lcr_requirement': 100,
            'compliant': lcr >= 100,
            'surplus_deficit': hqla - net_outflows
        }
    
    def _calculate_stressed_outflows(self, days: int) -> float:
        """
        Calculate stressed cash outflows
        
        Assumes stress scenarios:
        - Retail deposits: 5-10% runoff
        - Wholesale funding: 100% runoff
        - Derivatives: increase in collateral requirements
        """
        total_outflows = 0
        
        for liab_name, liab_info in self.liabilities.items():
            amount = liab_info['amount']
            maturity_days = liab_info.get('maturity_days', 365)
            liability_type = liab_info.get('type', 'wholesale')
            
            if maturity_days <= days:
                if liability_type == 'retail_stable':
                    runoff_rate = 0.05  # 5% runoff
                elif liability_type == 'retail_less_stable':
                    runoff_rate = 0.10  # 10% runoff
                else:  # wholesale
                    runoff_rate = 1.00  # 100% runoff
                
                total_outflows += amount * runoff_rate
        
        return total_outflows
    
    def _calculate_expected_inflows(self, days: int) -> float:
        """Calculate expected cash inflows"""
        # Simplified - would track receivables, maturing assets, etc.
        return 0  # Conservative assumption
    
    def calculate_net_stable_funding_ratio(self) -> Dict:
        """
        NSFR = Available Stable Funding / Required Stable Funding
        
        Basel III requires NSFR ≥ 100%
        """
        # Available Stable Funding (ASF)
        asf = self.cash * 1.0  # Cash counts 100%
        
        # Add stable funding sources
        for liab_name, liab_info in self.liabilities.items():
            amount = liab_info['amount']
            maturity_days = liab_info.get('maturity_days', 365)
            liability_type = liab_info.get('type', 'wholesale')
            
            if maturity_days >= 365:
                if liability_type in ['retail_stable', 'retail_less_stable']:
                    asf_factor = 0.95  # 95% stable
                else:
                    asf_factor = 1.00  # 100% stable for long-term wholesale
            elif maturity_days >= 180:
                asf_factor = 0.50
            else:
                asf_factor = 0.00
            
            asf += amount * asf_factor
        
        # Required Stable Funding (RSF)
        rsf = 0
        
        for asset_name, asset_info in self.assets.items():
            value = asset_info['value']
            liquidity_class = asset_info.get('liquidity_class', 'Level 2B')
            maturity_days = asset_info.get('maturity_days', 365)
            
            # RSF factors based on asset type and maturity
            if liquidity_class == 'Level 1':
                rsf_factor = 0.00  # No RSF for highest quality
            elif liquidity_class == 'Level 2A':
                rsf_factor = 0.15
            elif maturity_days >= 365:
                rsf_factor = 0.85
            else:
                rsf_factor = 0.50
            
            rsf += value * rsf_factor
        
        nsfr = (asf / rsf * 100) if rsf > 0 else float('inf')
        
        return {
            'available_stable_funding': asf,
            'required_stable_funding': rsf,
            'nsfr_ratio': nsfr,
            'nsfr_requirement': 100,
            'compliant': nsfr >= 100
        }
    
    def calculate_liquidity_gap(self, time_horizons: List[int] = [7, 30, 90, 180, 365]) -> pd.DataFrame:
        """
        Liquidity gap analysis across time horizons
        
        Gap = Cash Inflows - Cash Outflows
        """
        gaps = []
        
        cumulative_cash = self.cash
        
        for days in time_horizons:
            # Calculate outflows
            outflows = 0
            for liab_name, liab_info in self.liabilities.items():
                if liab_info.get('maturity_days', 365) <= days:
                    outflows += liab_info['amount']
            
            # Calculate inflows (simplified)
            inflows = 0
            
            # Gap
            gap = cumulative_cash + inflows - outflows
            cumulative_cash = gap
            
            gaps.append({
                'days': days,
                'inflows': inflows,
                'outflows': outflows,
                'gap': gap,
                'cumulative_gap': cumulative_cash
            })
        
        return pd.DataFrame(gaps)
    
    def stress_test_liquidity(self, 
                             deposit_runoff: float = 0.20,
                             asset_haircut: float = 0.30,
                             funding_unavailable: bool = True) -> Dict:
        """
        Stress test liquidity position
        
        Args:
            deposit_runoff: % of deposits withdrawn
            asset_haircut: Additional haircut on asset liquidation
            funding_unavailable: Can't access wholesale funding
        """
        # Available liquidity under stress
        stressed_cash = self.cash
        
        # Deposit withdrawals
        for liab_name, liab_info in self.liabilities.items():
            if liab_info.get('type', '').startswith('retail'):
                stressed_cash -= liab_info['amount'] * deposit_runoff
        
        # Asset liquidation value
        liquidation_value = 0
        for asset_name, asset_info in self.assets.items():
            value = asset_info['value']
            normal_haircut = asset_info.get('haircut', 0.10)
            stressed_haircut = normal_haircut + asset_haircut
            
            liquidation_value += value * (1 - min(stressed_haircut, 0.90))
        
        total_available = stressed_cash + liquidation_value
        
        # Funding needs
        funding_needs = 0
        for liab_name, liab_info in self.liabilities.items():
            if liab_info.get('maturity_days', 365) <= 30:
                funding_needs += liab_info['amount']
        
        # Survival period
        daily_burn_rate = funding_needs / 30
        survival_days = total_available / daily_burn_rate if daily_burn_rate > 0 else float('inf')
        
        return {
            'stressed_cash': stressed_cash,
            'liquidation_value': liquidation_value,
            'total_available_liquidity': total_available,
            'funding_needs_30d': funding_needs,
            'liquidity_surplus_deficit': total_available - funding_needs,
            'survival_days': survival_days,
            'scenario': {
                'deposit_runoff': deposit_runoff,
                'asset_haircut': asset_haircut,
                'funding_unavailable': funding_unavailable
            }
        }

# Example Usage
if __name__ == "__main__":
    # Sample balance sheet
    assets = {
        'Cash': {
            'value': 10000000,
            'liquidity_class': 'Level 1',
            'haircut': 0.00
        },
        'Government Bonds': {
            'value': 50000000,
            'liquidity_class': 'Level 1',
            'haircut': 0.00,
            'maturity_days': 180
        },
        'Corporate Bonds': {
            'value': 30000000,
            'liquidity_class': 'Level 2A',
            'haircut': 0.15,
            'maturity_days': 365
        },
        'Equities': {
            'value': 20000000,
            'liquidity_class': 'Level 2B',
            'haircut': 0.50,
            'maturity_days': 0  # Can sell anytime but high haircut
        },
        'Loans': {
            'value': 100000000,
            'liquidity_class': 'Illiquid',
            'haircut': 0.70,
            'maturity_days': 1825  # 5 years
        }
    }
    
    liabilities = {
        'Stable Retail Deposits': {
            'amount': 80000000,
            'maturity_days': 0,  # Demand deposits
            'type': 'retail_stable'
        },
        'Wholesale Funding (30-day)': {
            'amount': 40000000,
            'maturity_days': 30,
            'type': 'wholesale'
        },
        'Long-term Debt': {
            'amount': 50000000,
            'maturity_days': 1825,
            'type': 'wholesale_term'
        }
    }
    
    liq_risk = LiquidityRiskManager(
        cash_balance=10000000,
        assets=assets,
        liabilities=liabilities
    )
    
    print("Liquidity Risk Analysis")
    print("="*70)
    print()
    
    # LCR
    lcr = liq_risk.calculate_liquidity_coverage_ratio()
    print("Liquidity Coverage Ratio (LCR)")
    print(f"  HQLA: \${lcr['hqla']:,.0f}")
print(f"  Net Cash Outflows (30d): \${lcr['net_cash_outflows_30d']:,.0f}")
print(f"  LCR: {lcr['lcr_ratio']:.1f}%")
print(f"  Requirement: {lcr['lcr_requirement']}%")
print(f"  Status: {'✓ COMPLIANT' if lcr['compliant'] else '✗ NON-COMPLIANT'}")
print()
    
    # NSFR
nsfr = liq_risk.calculate_net_stable_funding_ratio()
print("Net Stable Funding Ratio (NSFR)")
print(f"  Available Stable Funding: \${nsfr['available_stable_funding']:,.0f}")
print(f"  Required Stable Funding: \${nsfr['required_stable_funding']:,.0f}")
print(f"  NSFR: {nsfr['nsfr_ratio']:.1f}%")
print(f"  Status: {'✓ COMPLIANT' if nsfr['compliant'] else '✗ NON-COMPLIANT'}")
print()
    
    # Liquidity gaps
gaps = liq_risk.calculate_liquidity_gap()
print("Liquidity Gap Analysis:")
print(gaps.to_string(index = False))
print()
    
    # Stress test
stress = liq_risk.stress_test_liquidity(
    deposit_runoff = 0.20,
    asset_haircut = 0.30,
    funding_unavailable = True
)

print("Liquidity Stress Test")
print(f"  Scenario: 20% deposit runoff, 30% additional haircut, no wholesale funding")
print(f"  Stressed Cash: \${stress['stressed_cash']:,.0f}")
print(f"  Asset Liquidation Value: \${stress['liquidation_value']:,.0f}")
print(f"  Total Available: \${stress['total_available_liquidity']:,.0f}")
print(f"  30-day Funding Needs: \${stress['funding_needs_30d']:,.0f}")
print(f"  Surplus/Deficit: \${stress['liquidity_surplus_deficit']:,.0f}")
print(f"  Survival Days: {stress['survival_days']:.0f}")
\`\`\`

## Market Liquidity Risk

Measuring market liquidity:

\`\`\`python
class MarketLiquidityAnalyzer:
    """
    Analyze market liquidity of assets
    """
    def __init__(self, positions: Dict[str, Dict]):
        """
        Args:
            positions: Dict of {asset: {shares, avg_price, adv}}
                      adv = average daily volume
        """
        self.positions = positions
        
    def calculate_days_to_liquidate(self, 
                                    participation_rate: float = 0.10) -> pd.DataFrame:
        """
        Days to liquidate at X% of daily volume
        
        Rule of thumb: Don't exceed 10-20% of ADV per day
        """
        results = []
        
        for asset, info in self.positions.items():
            shares = info['shares']
            adv = info.get('adv', 0)
            
            if adv == 0:
                days = float('inf')
            else:
                max_daily_shares = adv * participation_rate
                days = shares / max_daily_shares
            
            position_value = shares * info['avg_price']
            
            results.append({
                'asset': asset,
                'position_value': position_value,
                'shares': shares,
                'adv': adv,
                'days_to_liquidate': days,
                'liquidity_risk': 'HIGH' if days > 5 else ('MEDIUM' if days > 2 else 'LOW')
            })
        
        return pd.DataFrame(results).sort_values('days_to_liquidate', ascending=False)
    
    def estimate_market_impact(self, 
                               asset: str,
                               shares_to_sell: float,
                               volatility: float = 0.20) -> Dict:
        """
        Estimate price impact of trade
        
        Market Impact ≈ σ × √(Shares / ADV) × Market Impact Coefficient
        
        Simplified square-root model
        """
        info = self.positions[asset]
        adv = info['adv']
        price = info['avg_price']
        
        # Participation rate
        participation = shares_to_sell / adv if adv > 0 else 1.0
        
        # Market impact coefficient (typically 0.1 - 1.0)
        impact_coef = 0.5
        
        # Estimated impact
        price_impact_pct = volatility * np.sqrt(participation) * impact_coef
        price_impact_dollar = price * price_impact_pct
        
        # Total cost
        total_slippage = shares_to_sell * price_impact_dollar / 2  # Average impact
        
        return {
            'asset': asset,
            'shares_to_sell': shares_to_sell,
            'participation_rate': participation,
            'price_impact_pct': price_impact_pct * 100,
            'price_impact_per_share': price_impact_dollar,
            'total_slippage_cost': total_slippage,
            'liquidity_cost_bps': (total_slippage / (shares_to_sell * price)) * 10000
        }
    
    def calculate_liquidity_score(self) -> pd.DataFrame:
        """
        Aggregate liquidity score for portfolio
        
        Score based on:
        - Bid-ask spread
        - Days to liquidate
        - Market impact
        """
        results = []
        
        for asset, info in self.positions.items():
            # Get metrics
            dtl_df = self.calculate_days_to_liquidate()
            dtl = dtl_df[dtl_df['asset'] == asset]['days_to_liquidate'].values[0]
            
            # Bid-ask spread (would pull from market data)
            bid_ask_spread_bps = info.get('spread_bps', 10)  # 10 bps default
            
            # Liquidity score (0-100, higher is more liquid)
            # Penalize for days to liquidate and spread
            score = 100 - min(dtl * 10, 50) - min(bid_ask_spread_bps / 2, 50)
            score = max(0, score)
            
            results.append({
                'asset': asset,
                'days_to_liquidate': dtl,
                'bid_ask_spread_bps': bid_ask_spread_bps,
                'liquidity_score': score,
                'liquidity_rating': 'EXCELLENT' if score >= 80 else (
                    'GOOD' if score >= 60 else (
                        'FAIR' if score >= 40 else 'POOR'
                    )
                )
            })
        
        return pd.DataFrame(results).sort_values('liquidity_score', ascending=False)

# Example
if __name__ == "__main__":
    positions = {
        'SPY': {
            'shares': 50000,
            'avg_price': 450,
            'adv': 50000000,
            'spread_bps': 1
        },
        'AAPL': {
            'shares': 100000,
            'avg_price': 180,
            'adv': 40000000,
            'spread_bps': 2
        },
        'Small Cap Stock': {
            'shares': 500000,
            'avg_price': 25,
            'adv': 200000,
            'spread_bps': 50
        }
    }
    
    mkt_liq = MarketLiquidityAnalyzer(positions)
    
    print("Market Liquidity Analysis")
    print("="*70)
    
    # Days to liquidate
    dtl = mkt_liq.calculate_days_to_liquidate(participation_rate=0.10)
    print("Days to Liquidate (10% ADV):")
    print(dtl.to_string(index=False))
    print()
    
    # Market impact
    impact = mkt_liq.estimate_market_impact('Small Cap Stock', 500000)
    print(f"Market Impact - Selling {impact['shares_to_sell']:,.0f} shares of {impact['asset']}:")
    print(f"  Participation Rate: {impact['participation_rate']*100:.1f}%")
    print(f"  Price Impact: {impact['price_impact_pct']:.2f}%")
    print(f"  Total Slippage Cost: \${impact['total_slippage_cost']:, .0f}")
print(f"  Liquidity Cost: {impact['liquidity_cost_bps']:.0f} bps")
print()
    
    # Liquidity scores
scores = mkt_liq.calculate_liquidity_score()
print("Liquidity Scores:")
print(scores.to_string(index = False))
\`\`\`

## Contingency Funding Plan

Every institution needs a liquidity contingency plan:

\`\`\`python
class ContingencyFundingPlan:
    """
    Liquidity contingency planning
    """
    def __init__(self, normal_sources: Dict[str, float]):
        """
        Args:
            normal_sources: Normal funding sources and amounts
        """
        self.normal_sources = normal_sources
        self.contingent_sources = {}
        self.trigger_indicators = {}
        
    def add_contingent_source(self,
                             source_name: str,
                             max_amount: float,
                             time_to_access_days: int,
                             cost_bps: float):
        """
        Add contingent funding source
        """
        self.contingent_sources[source_name] = {
            'max_amount': max_amount,
            'time_to_access_days': time_to_access_days,
            'cost_bps': cost_bps
        }
    
    def add_trigger(self,
                   indicator_name: str,
                   current_value: float,
                   yellow_threshold: float,
                   red_threshold: float):
        """
        Add early warning indicator
        """
        if current_value >= red_threshold:
            status = 'RED'
        elif current_value >= yellow_threshold:
            status = 'YELLOW'
        else:
            status = 'GREEN'
        
        self.trigger_indicators[indicator_name] = {
            'current_value': current_value,
            'yellow_threshold': yellow_threshold,
            'red_threshold': red_threshold,
            'status': status
        }
    
    def evaluate_triggers(self) -> Dict:
        """
        Evaluate all triggers
        """
        red_count = sum(1 for t in self.trigger_indicators.values() if t['status'] == 'RED')
        yellow_count = sum(1 for t in self.trigger_indicators.values() if t['status'] == 'YELLOW')
        
        if red_count > 0:
            overall_status = 'RED - ACTIVATE CONTINGENCY PLAN'
        elif yellow_count > 0:
            overall_status = 'YELLOW - HEIGHTENED MONITORING'
        else:
            overall_status = 'GREEN - NORMAL OPERATIONS'
        
        return {
            'overall_status': overall_status,
            'red_triggers': red_count,
            'yellow_triggers': yellow_count,
            'green_triggers': len(self.trigger_indicators) - red_count - yellow_count,
            'details': self.trigger_indicators
        }
    
    def calculate_total_contingent_capacity(self) -> Dict:
        """
        Total contingent funding available
        """
        immediate = 0  # <1 day
        short_term = 0  # 1-7 days
        medium_term = 0  # 7-30 days
        
        for source, info in self.contingent_sources.items():
            amount = info['max_amount']
            days = info['time_to_access_days']
            
            if days <= 1:
                immediate += amount
            elif days <= 7:
                short_term += amount
            else:
                medium_term += amount
        
        total = immediate + short_term + medium_term
        
        return {
            'immediate_capacity': immediate,
            'short_term_capacity': short_term,
            'medium_term_capacity': medium_term,
            'total_capacity': total
        }

# Example
if __name__ == "__main__":
    # Normal funding sources
    normal_funding = {
        'Retail Deposits': 80000000,
        'Wholesale Funding': 40000000
    }
    
    cfp = ContingencyFundingPlan(normal_funding)
    
    # Add contingent sources
    cfp.add_contingent_source('Central Bank Facility', 50000000, time_to_access_days=1, cost_bps=100)
    cfp.add_contingent_source('Asset Sales', 30000000, time_to_access_days=5, cost_bps=300)
    cfp.add_contingent_source('Credit Lines', 20000000, time_to_access_days=1, cost_bps=50)
    
    # Add triggers
    cfp.add_trigger('LCR', current_value=110, yellow_threshold=110, red_threshold=100)
    cfp.add_trigger('Deposit Outflows', current_value=5, yellow_threshold=10, red_threshold=20)
    cfp.add_trigger('Wholesale Funding Rollover', current_value=85, yellow_threshold=80, red_threshold=70)
    
    print("Contingency Funding Plan")
    print("="*70)
    
    # Evaluate triggers
    trigger_status = cfp.evaluate_triggers()
    print(f"Status: {trigger_status['overall_status']}")
    print(f"  Red Triggers: {trigger_status['red_triggers']}")
    print(f"  Yellow Triggers: {trigger_status['yellow_triggers']}")
    print()
    
    # Contingent capacity
    capacity = cfp.calculate_total_contingent_capacity()
    print("Contingent Funding Capacity:")
    print(f"  Immediate (<1 day): \${capacity['immediate_capacity']:, .0f}")
print(f"  Short-term (1-7 days): \${capacity['short_term_capacity']:,.0f}")
print(f"  Medium-term (7-30 days): \${capacity['medium_term_capacity']:,.0f}")
print(f"  Total: \${capacity['total_capacity']:,.0f}")
\`\`\`

## Key Takeaways

1. **Two Types**: Funding liquidity and market liquidity
2. **Liquidity Spiral**: These risks amplify each other
3. **Basel Metrics**: LCR (short-term) and NSFR (structural)
4. **Stress Testing**: Model severe funding stress
5. **Market Impact**: Large positions = high liquidation cost
6. **Contingency Planning**: Have backup funding sources
7. **Early Warning**: Monitor triggers before crisis hits
8. **Diversification**: Don't rely on single funding source

## Common Pitfalls

❌ **Maturity Mismatch**: Funding long-term assets with short-term liabilities  
❌ **Concentration**: Over-reliance on single funding source  
❌ **Illiquid Assets**: Can't sell when you need to  
❌ **Ignored Correlations**: Liquidity dries up for everyone in crisis  
❌ **No Contingency Plan**: Scrambling during crisis

## Conclusion

Liquidity is the lifeblood of finance. You can have positive equity, profitable operations, and still die from lack of liquidity.

The key lessons:
- **Maintain liquidity buffer**: Always have excess cash/liquid assets
- **Diversify funding**: Multiple sources, multiple maturities
- **Stress test regularly**: Know your survival time under stress
- **Have contingency plan**: Know backup funding sources before you need them
- **Monitor early warnings**: Act when indicators turn yellow, not when red

As Warren Buffett said: "Only when the tide goes out do you discover who's been swimming naked." Liquidity crises reveal who prepared and who didn't.

Next: Risk Attribution Analysis - understanding where risk and returns come from.
`,
};
