export const alternativeInvestments = {
  title: 'Alternative Investments',
  slug: 'alternative-investments',
  description:
    'Master hedge funds, private equity, real estate, and collectibles',
  content: `
# Alternative Investments

## Introduction: Beyond Stocks and Bonds

Alternative investments offer diversification and uncorrelated returns:
- 📊 **$13+ Trillion** in global alternatives AUM
- 🏦 **Hedge funds**: Absolute return strategies, long-short, arbitrage
- 💼 **Private equity**: Buy companies, improve them, sell for profit
- 🏠 **Real estate**: REITs, direct ownership, commercial properties
- 🎨 **Collectibles**: Art, wine, watches, NFTs

**What makes them "alternative":**
- Illiquid (can't sell instantly)
- High minimums (\$100K-$1M+)
- Limited transparency
- Different risk/return profile
- Low correlation with stocks/bonds

**What you'll learn:**
- Hedge fund strategies
- Private equity mechanics
- Real estate investing
- Alternative data
- Building systems for alternatives
- Due diligence frameworks

---

## Hedge Funds: Strategies and Structure

Hedge funds pursue "absolute returns" - profit regardless of market direction.

\`\`\`python
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List
import numpy as np

class HedgeFundStrategy(Enum):
    LONG_SHORT_EQUITY = "Long/Short Equity"
    MARKET_NEUTRAL = "Market Neutral"
    EVENT_DRIVEN = "Event Driven"
    GLOBAL_MACRO = "Global Macro"
    CTA_MANAGED_FUTURES = "CTA/Managed Futures"
    ARBITRAGE = "Arbitrage"
    DISTRESSED = "Distressed Securities"

@dataclass
class HedgeFund:
    """Model hedge fund characteristics"""
    
    name: str
    strategy: HedgeFundStrategy
    aum: float
    management_fee: float  # Usually 2%
    performance_fee: float  # Usually 20%
    minimum_investment: float
    lockup_period_months: int
    redemption_frequency: str
    target_return: float
    target_volatility: float
    
    def calculate_fees (self, 
                      starting_aum: float,
                      ending_aum: float,
                      benchmark_return: float = 0) -> Dict:
        """
        Calculate 2-and-20 fee structure
        
        Management fee: 2% of AUM (regardless of performance)
        Performance fee: 20% of profits above high-water mark
        """
        # Management fee (on average AUM)
        avg_aum = (starting_aum + ending_aum) / 2
        mgmt_fee = avg_aum * self.management_fee
        
        # Performance fee (on profits)
        profit = max(0, ending_aum - starting_aum)
        
        # High-water mark: Only charge on new highs
        # Hurdle rate: Some funds only charge above benchmark
        excess_return = profit - (starting_aum * benchmark_return)
        
        perf_fee = max(0, excess_return) * self.performance_fee
        
        total_fees = mgmt_fee + perf_fee
        
        # Net return to investor
        gross_return = (ending_aum - starting_aum) / starting_aum
        net_return = (ending_aum - starting_aum - total_fees) / starting_aum
        
        return {
            'management_fee': mgmt_fee,
            'performance_fee': perf_fee,
            'total_fees': total_fees,
            'gross_return': gross_return * 100,
            'net_return': net_return * 100,
            'fee_as_pct_of_profit': (total_fees / profit * 100) if profit > 0 else 0
        }
    
    @staticmethod
    def strategy_profiles() -> Dict:
        """Profile major hedge fund strategies"""
        return {
            'Long/Short Equity': {
                'description': 'Long undervalued stocks, short overvalued stocks',
                'target_return': '10-15% annually',
                'volatility': '8-12%',
                'market_correlation': '0.4-0.6 (partially hedged)',
                'example': 'Tiger Global, Lone Pine',
                'risk': 'Long and short can both lose (2020: growth shorts killed funds)'
            },
            'Market Neutral': {
                'description': 'Equal long/short, beta = 0, profit from stock selection',
                'target_return': '5-8% annually',
                'volatility': '3-6%',
                'market_correlation': '~0 (zero beta)',
                'example': 'Renaissance Technologies (Medallion Fund)',
                'risk': 'Low returns in low-vol markets, model risk'
            },
            'Event Driven': {
                'description': 'Profit from M&A, bankruptcies, restructurings',
                'target_return': '8-12%',
                'volatility': '6-10%',
                'market_correlation': '0.2-0.4 (merger arb)',
                'example': 'Elliott Management, Third Point',
                'risk': 'Deals fall through (regulatory, financing)'
            },
            'Global Macro': {
                'description': 'Trade currencies, rates, commodities based on macro views',
                'target_return': '10-20%',
                'volatility': '10-20%',
                'market_correlation': '0-0.3 (uncorrelated)',
                'example': 'Bridgewater, Soros Fund Management',
                'risk': 'Wrong macro call = big losses (Soros bet on Yen)'
            },
            'CTA/Managed Futures': {
                'description': 'Trend-following on futures (stocks, bonds, commodities)',
                'target_return': '5-15%',
                'volatility': '15-25%',
                'market_correlation': '-0.2-0.2 (often negative = crisis alpha)',
                'example': 'Winton, Man AHL',
                'risk': 'Whipsaws in choppy markets, leverage'
            }
        }

# Example fund
fund = HedgeFund(
    name="Alpha Capital",
    strategy=HedgeFundStrategy.LONG_SHORT_EQUITY,
    aum=1_000_000_000,  # $1B AUM
    management_fee=0.02,  # 2%
    performance_fee=0.20,  # 20%
    minimum_investment=1_000_000,  # $1M minimum
    lockup_period_months=12,  # 1-year lockup
    redemption_frequency="Quarterly",
    target_return=0.15,  # 15% target
    target_volatility=0.10  # 10% vol
)

# Calculate fees on $10M investment
fees = fund.calculate_fees(
    starting_aum=10_000_000,
    ending_aum=11_500_000,  # 15% return
    benchmark_return=0.10  # S&P returned 10%
)

print("=== Hedge Fund Fee Structure ===\\n")
print(f"Fund: {fund.name}")
print(f"Strategy: {fund.strategy.value}")
print(f"\\nInvestment Performance:")
print(f"  Starting: $10,000,000")
print(f"  Ending: $11,500,000")
print(f"  Gross Return: {fees['gross_return']:.2f}%")
print(f"\\nFees (2-and-20):")
print(f"  Management Fee (2%): \\$\{fees['management_fee']:,.0f}")
print(f"  Performance Fee (20%): \\$\{fees['performance_fee']:,.0f}")
print(f"  Total Fees: \\$\{fees['total_fees']:,.0f}")
print(f"  Fees as % of Profit: {fees['fee_as_pct_of_profit']:.1f}%")
print(f"\\nNet Return to Investor: {fees['net_return']:.2f}%")

# Strategy profiles
print("\\n\\n=== Hedge Fund Strategies ===\\n")
profiles = HedgeFund.strategy_profiles()

for strategy, details in list (profiles.items())[:3]:
    print(f"{strategy}:")
    print(f"  Description: {details['description']}")
    print(f"  Target Return: {details['target_return']}")
    print(f"  Volatility: {details['volatility']}")
    print(f"  Market Correlation: {details['market_correlation']}")
    print(f"  Risk: {details['risk']}\\n")
\`\`\`

**Key Insight**: 2-and-20 fees are expensive. On 15% return, ~17% of profit goes to fees!

---

## Private Equity: Leveraged Buyouts and VC

Private equity owns companies not traded on public markets.

\`\`\`python
class PrivateEquityFund:
    """
    Model PE fund structure and returns
    """
    
    def __init__(self, fund_size: float, leverage_ratio: float = 0.60):
        self.fund_size = fund_size
        self.leverage_ratio = leverage_ratio  # 60% debt, 40% equity
        self.management_fee = 0.02  # 2% on committed capital
        self.carried_interest = 0.20  # 20% of profits
        self.hurdle_rate = 0.08  # 8% preferred return to LPs
        
    def lbo_model (self,
                  purchase_price: float,
                  ebitda: float,
                  entry_multiple: float,
                  exit_multiple: float,
                  revenue_growth: float,
                  ebitda_margin_improvement: float,
                  holding_period_years: int = 5,
                  debt_paydown: float = 0.40) -> Dict:
        """
        Leveraged Buyout (LBO) model
        
        1. Buy company using 60% debt, 40% equity
        2. Improve operations (revenue growth, margin expansion)
        3. Pay down debt
        4. Sell at higher multiple
        5. Profit!
        """
        # Entry
        purchase_price = ebitda * entry_multiple
        equity_investment = purchase_price * (1 - self.leverage_ratio)
        debt = purchase_price * self.leverage_ratio
        
        # Operations (improve over holding period)
        final_revenue = (1 + revenue_growth) ** holding_period_years
        final_ebitda = ebitda * final_revenue * (1 + ebitda_margin_improvement)
        
        # Exit
        enterprise_value = final_ebitda * exit_multiple
        remaining_debt = debt * (1 - debt_paydown)
        equity_value = enterprise_value - remaining_debt
        
        # Returns
        total_return = equity_value / equity_investment
        irr = (total_return ** (1 / holding_period_years)) - 1
        moic = equity_value / equity_investment  # Multiple on Invested Capital
        
        return {
            'purchase_price': purchase_price,
            'equity_investment': equity_investment,
            'initial_debt': debt,
            'entry_multiple': entry_multiple,
            'exit_multiple': exit_multiple,
            'final_ebitda': final_ebitda,
            'enterprise_value_exit': enterprise_value,
            'remaining_debt': remaining_debt,
            'equity_value_exit': equity_value,
            'moic': moic,
            'irr': irr * 100,
            'holding_period': holding_period_years
        }
    
    def calculate_waterfall (self, 
                           total_profit: float,
                           lp_commitment: float) -> Dict:
        """
        Private equity waterfall structure
        
        1. Return capital to LPs
        2. Preferred return to LPs (8% hurdle)
        3. Catch-up to GP (until GP has 20%)
        4. 80/20 split thereafter
        """
        # Return of capital
        capital_returned = lp_commitment
        remaining = total_profit - capital_returned
        
        # Preferred return (8% annually over holding period, simplified)
        preferred_return = lp_commitment * (1.08 ** 5 - 1)  # 5-year hold
        lp_preferred = min (preferred_return, remaining)
        remaining -= lp_preferred
        
        # Catch-up (GP gets 100% until they have 20% of all profits)
        catchup = lp_preferred / 4  # Simplified
        gp_catchup = min (catchup, remaining)
        remaining -= gp_catchup
        
        # 80/20 split
        lp_split = remaining * 0.80
        gp_split = remaining * 0.20
        
        # Totals
        lp_total = capital_returned + lp_preferred + lp_split
        gp_total = gp_catchup + gp_split
        
        return {
            'total_profit': total_profit,
            'lp_total': lp_total,
            'gp_total': gp_total,
            'lp_pct': lp_total / total_profit * 100,
            'gp_pct': gp_total / total_profit * 100,
            'breakdown': {
                'capital_returned': capital_returned,
                'preferred_return': lp_preferred,
                'catchup': gp_catchup,
                '80_20_split_lp': lp_split,
                '80_20_split_gp': gp_split
            }
        }

# LBO example
pe_fund = PrivateEquityFund (fund_size=1_000_000_000)

lbo = pe_fund.lbo_model(
    purchase_price=500_000_000,  # Ignored, calculated from EBITDA
    ebitda=50_000_000,  # $50M EBITDA
    entry_multiple=10,  # Buy at 10x EBITDA
    exit_multiple=12,  # Sell at 12x (multiple expansion)
    revenue_growth=0.05,  # 5% annual revenue growth
    ebitda_margin_improvement=0.10,  # Improve margins by 10%
    holding_period_years=5,
    debt_paydown=0.40  # Pay down 40% of debt
)

print("\\n=== Leveraged Buyout Model ===\\n")
print(f"Entry:")
print(f"  Purchase Price: \\$\{lbo['purchase_price']/1e6:.0f}M")
print(f"  Entry Multiple: {lbo['entry_multiple']}x EBITDA")
print(f"  Equity Investment: \\$\{lbo['equity_investment']/1e6:.0f}M")
print(f"  Initial Debt: \\$\{lbo['initial_debt']/1e6:.0f}M")
print(f"\\nOperations (5 years):")
print(f"  Revenue Growth: 5% annually")
print(f"  Margin Improvement: 10%")
print(f"  Final EBITDA: \\$\{lbo['final_ebitda']/1e6:.0f}M")
print(f"\\nExit:")
print(f"  Enterprise Value: \\$\{lbo['enterprise_value_exit']/1e6:.0f}M")
print(f"  Exit Multiple: {lbo['exit_multiple']}x EBITDA")
print(f"  Remaining Debt: \\$\{lbo['remaining_debt']/1e6:.0f}M")
print(f"  Equity Value: \\$\{lbo['equity_value_exit']/1e6:.0f}M")
print(f"\\nReturns:")
print(f"  MOIC: {lbo['moic']:.2f}x")
print(f"  IRR: {lbo['irr']:.1f}%")
\`\`\`

**Value Creation in PE:**1. **Multiple expansion**: Buy at 10x, sell at 12x
2. **Operational improvements**: Grow revenue, improve margins
3. **Debt paydown**: Reduces debt, increases equity value
4. **Financial engineering**: Leverage amplifies returns

---

## Real Estate Investing

Real estate provides income and inflation protection.

\`\`\`python
class RealEstateInvestment:
    """
    Model real estate investment returns
    """
    
    def calculate_reit_metrics (self,
                               property_value: float,
                               annual_noi: float,  # Net Operating Income
                               debt: float,
                               interest_rate: float,
                               appreciation_rate: float,
                               holding_period: int = 10) -> Dict:
        """
        REIT and direct real estate metrics
        
        Cap Rate = NOI / Property Value
        Cash-on-Cash = Cash Flow / Equity Investment
        Total Return = Cash Flow + Appreciation
        """
        # Cap rate
        cap_rate = annual_noi / property_value
        
        # Equity investment
        equity = property_value - debt
        
        # Annual debt service
        annual_debt_service = debt * interest_rate
        
        # Cash flow
        annual_cash_flow = annual_noi - annual_debt_service
        
        # Cash-on-cash return
        cash_on_cash = annual_cash_flow / equity
        
        # Total return over holding period
        # Appreciation
        future_value = property_value * ((1 + appreciation_rate) ** holding_period)
        
        # Total cash flows
        total_cash_flows = annual_cash_flow * holding_period
        
        # Sale proceeds (after paying off debt)
        sale_proceeds = future_value - debt
        
        # Total return
        total_return = (total_cash_flows + sale_proceeds - equity) / equity
        irr = (1 + total_return) ** (1 / holding_period) - 1
        
        return {
            'property_value': property_value,
            'equity_investment': equity,
            'debt': debt,
            'annual_noi': annual_noi,
            'cap_rate': cap_rate * 100,
            'annual_cash_flow': annual_cash_flow,
            'cash_on_cash_return': cash_on_cash * 100,
            'future_value': future_value,
            'sale_proceeds': sale_proceeds,
            'total_return': total_return * 100,
            'irr': irr * 100,
            'holding_period': holding_period
        }
    
    @staticmethod
    def reit_types() -> Dict:
        """Different REIT categories"""
        return {
            'Residential': {
                'examples': 'AvalonBay (AVB), Equity Residential (EQR)',
                'focus': 'Apartments, multifamily housing',
                'typical_yield': '3-4%',
                'risk': 'Economic cycles, rent control'
            },
            'Office': {
                'examples': 'Boston Properties (BXP), SL Green (SLG)',
                'focus': 'Office buildings',
                'typical_yield': '4-6%',
                'risk': 'Work-from-home trend, urban exodus'
            },
            'Industrial': {
                'examples': 'Prologis (PLD), Duke Realty (DRE)',
                'focus': 'Warehouses, logistics',
                'typical_yield': '2-3%',
                'risk': 'E-commerce growth = strong, but cyclical'
            },
            'Retail': {
                'examples': 'Simon Property (SPG), Realty Income (O)',
                'focus': 'Shopping malls, strip centers',
                'typical_yield': '4-7%',
                'risk': 'E-commerce killing retail'
            },
            'Data Centers': {
                'examples': 'Equinix (EQIX), Digital Realty (DLR)',
                'focus': 'Server farms, cloud infrastructure',
                'typical_yield': '2-3%',
                'risk': 'High growth but expensive, tech dependent'
            }
        }

# Example: Apartment building investment
re_investment = RealEstateInvestment()

metrics = re_investment.calculate_reit_metrics(
    property_value=10_000_000,  # $10M apartment building
    annual_noi=600_000,  # $600K net operating income
    debt=6_000_000,  # 60% LTV
    interest_rate=0.05,  # 5% mortgage rate
    appreciation_rate=0.03,  # 3% annual appreciation
    holding_period=10
)

print("\\n\\n=== Real Estate Investment Analysis ===\\n")
print(f"Property: $10M Apartment Building")
print(f"Equity Investment: \\$\{metrics['equity_investment']/1e6:.1f}M")
print(f"Debt: \\$\{metrics['debt']/1e6:.1f}M (60% LTV)")
print(f"\\nIncome Metrics:")
print(f"  Annual NOI: \\$\{metrics['annual_noi']:,.0f}")
print(f"  Cap Rate: {metrics['cap_rate']:.2f}%")
print(f"  Annual Cash Flow: \\$\{metrics['annual_cash_flow']:,.0f}")
print(f"  Cash-on-Cash Return: {metrics['cash_on_cash_return']:.2f}%")
print(f"\\n10-Year Returns:")
print(f"  Future Value: \\$\{metrics['future_value']/1e6:.1f}M")
print(f"  Total Return: {metrics['total_return']:.1f}%")
print(f"  IRR: {metrics['irr']:.2f}%")
\`\`\`

---

## Building an Alternatives Analysis System

\`\`\`python
class AlternativesPortfolio:
    """
    Manage portfolio of alternative investments
    """
    
    def __init__(self):
        self.hedge_funds = []
        self.pe_funds = []
        self.real_estate = []
        self.collectibles = []
        
    def calculate_illiquidity_premium (self,
                                     liquid_return: float,
                                     illiquid_return: float,
                                     lockup_years: float) -> Dict:
        """
        Should you accept illiquidity for higher returns?
        
        Illiquidity Premium = Illiquid Return - Liquid Return
        
        Rule: Need 2-4% premium per year of lockup
        """
        premium = illiquid_return - liquid_return
        premium_per_year = premium / lockup_years
        
        acceptable = premium_per_year > 0.02  # 2%+ per year minimum
        
        return {
            'liquid_return': liquid_return * 100,
            'illiquid_return': illiquid_return * 100,
            'lockup_years': lockup_years,
            'illiquidity_premium': premium * 100,
            'premium_per_year': premium_per_year * 100,
            'acceptable': acceptable,
            'recommendation': 'INVEST' if acceptable else 'PASS'
        }
    
    def due_diligence_checklist (self) -> Dict:
        """
        Due diligence for alternative investments
        """
        return {
            'Track Record': [
                '10+ years of audited returns',
                'Consistency across market cycles',
                'Compare to relevant benchmark',
                'Check for survivorship bias'
            ],
            'Strategy': [
                'Clearly defined, repeatable process',
                'Capacity limits (can strategy scale?)',
                'Edge: Why does this work?',
                'Risk management framework'
            ],
            'Team': [
                'Experience and stability',
                'Skin in the game (managers invest own $)',
                'Alignment of interests',
                'Key person risk'
            ],
            'Terms': [
                'Fees (2-and-20 vs 1-and-10)',
                'Liquidity (lockup, redemption frequency)',
                'High-water mark (don\\'t pay twice on same gains)',
                'Hurdle rate (only pay performance above benchmark)'
            ],
            'Operational': [
                'Independent administrator',
                'Third-party custody',
                'Big 4 auditor',
                'Regulatory compliance'
            },
            'Red Flags': [
                'Consistent smooth returns (Madoff!)',
                'Proprietary administrator',
                'High turnover of partners',
                'Opaque strategy',
                'Marketing focus > substance'
            ]
        }

# Usage
portfolio = AlternativesPortfolio()

# Illiquidity analysis
analysis = portfolio.calculate_illiquidity_premium(
    liquid_return=0.08,  # Public stocks: 8%
    illiquid_return=0.15,  # PE fund: 15%
    lockup_years=5  # 5-year lockup
)

print("\\n\\n=== Illiquidity Premium Analysis ===\\n")
print(f"Liquid Investment (Public Stocks): {analysis['liquid_return']:.1f}% annually")
print(f"Illiquid Investment (PE Fund): {analysis['illiquid_return']:.1f}% annually")
print(f"Lockup Period: {analysis['lockup_years']} years")
print(f"\\nIlliquidity Premium: {analysis['illiquidity_premium']:.1f}%")
print(f"Premium per Year: {analysis['premium_per_year']:.1f}%")
print(f"Acceptable: {'YES' if analysis['acceptable'] else 'NO'}")
print(f"\\nRecommendation: {analysis['recommendation']}")

# Due diligence
dd = portfolio.due_diligence_checklist()
print("\\n\\n=== Alternative Investment Due Diligence ===\\n")
print("Track Record:")
for item in dd['Track Record']:
    print(f"  ✓ {item}")

print("\\nRed Flags to Avoid:")
for item in dd['Red Flags']:
    print(f"  ✗ {item}")
\`\`\`

---

## Summary

**Key Takeaways:**1. **Hedge Funds**: Absolute return strategies, 2-and-20 fees, higher risk
2. **Private Equity**: LBOs, operational improvements, 5-7 year lockups
3. **Real Estate**: Income + appreciation, REITs for liquidity
4. **Illiquidity Premium**: Need 2-4% extra return per year of lockup
5. **Due Diligence**: Track record, strategy, team, terms, operational
6. **Fees Matter**: 2-and-20 can eat 30%+ of profits

**For Engineers:**
- Limited data (illiquid, private)
- Long time horizons (5-10 years)
- Complex fee structures
- Due diligence > algorithms
- Diversification across strategies

**Next Steps:**
- Module 7: Portfolio optimization with alternatives
- Module 12: Alternative data sources
- Module 18: ML for alternative investment selection

You now understand alternatives - diversification beyond stocks/bonds!
`,
  exercises: [
    {
      prompt:
        'Build a private equity LBO modeling system that calculates IRR and MOIC across different scenarios (base, bull, bear). Include leverage ratios, operational improvements, multiple expansion, and debt paydown. Visualize which factors drive returns most.',
      solution:
        '// Implementation: 1) Input: Entry multiple, exit multiple, EBITDA, revenue growth, margin improvement, leverage, debt paydown, 2) Calculate purchase price, equity investment, debt, 3) Project financials over 5 years, 4) Calculate exit enterprise value and equity value, 5) Compute IRR and MOIC, 6) Run Monte Carlo with 1000 scenarios (vary all inputs), 7) Sensitivity analysis: Which variable drives returns most? (usually exit multiple), 8) Visualize tornado chart of sensitivities',
    },
    {
      prompt:
        'Create a hedge fund due diligence system that fetches hedge fund return data, calculates risk-adjusted metrics (Sharpe, Sortino, max drawdown, downside capture), detects smoothing/fraud indicators (too-consistent returns), and flags red flags based on a scoring rubric.',
      solution:
        '// Implementation: 1) Fetch monthly returns (from HFR, PivotalPath, or manual), 2) Calculate: Sharpe ratio, Sortino ratio, Calmar ratio, max drawdown, up/down capture vs S&P 500, 3) Fraud detection: Check if monthly volatility is suspiciously low (Madoff had 0.5% vol with 10%+ returns = impossible), 4) Rolling correlations with major indices, 5) Red flags: Consistent smooth returns, same administrator as fund (not independent), unclear strategy, 6) Score fund 0-100, flag if < 60',
    },
  ],
};
