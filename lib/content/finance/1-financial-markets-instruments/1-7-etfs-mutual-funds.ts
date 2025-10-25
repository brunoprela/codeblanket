export const etfsMutualFunds = {
  title: 'ETFs & Mutual Funds',
  slug: 'etfs-mutual-funds',
  description: 'Master pooled investment vehicles - from SPY to factor funds',
  content: `
# ETFs & Mutual Funds

## Introduction: Democratizing Investment

ETFs and Mutual Funds allow regular investors to own diversified portfolios:
- 📊 **$10+ Trillion** in ETF assets globally
- 🎯 **Instant diversification** - own 500 stocks with one purchase
- 💰 **Low cost** - expense ratios as low as 0.03%
- 🔄 **SPY trades $30B daily** - most liquid security in the world
- 📈 **Index funds beat 90%+ of active managers** over 10+ years

**What makes them powerful:**
- Professional management (or passive index tracking)
- Diversification (reduce unsystematic risk)
- Liquidity (trade like stocks for ETFs)
- Tax efficiency (especially ETFs)
- Low minimum investment

**What you'll learn:**
- ETFs vs Mutual Funds (key differences)
- Index funds and passive investing
- Active vs passive management
- Factor investing (value, momentum, quality)
- Building ETF trading systems
- Creation/redemption mechanism

---

## ETFs vs Mutual Funds: Key Differences

Both pool money from investors, but structure differs significantly.

\`\`\`python
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict
import numpy as np

class FundType(Enum):
    ETF = "Exchange-Traded Fund"
    MUTUAL_FUND = "Mutual Fund"
    CLOSED_END_FUND = "Closed-End Fund"

@dataclass
class InvestmentFund:
    """Compare ETFs, Mutual Funds, Closed-End Funds"""
    
    name: str
    ticker: str
    fund_type: FundType
    expense_ratio: float
    aum: float  # Assets Under Management
    inception_date: str
    trading_mechanism: str
    pricing: str
    minimum_investment: float
    tax_efficiency: str
    
    def get_characteristics (self) -> Dict:
        """Compare fund types"""
        if self.fund_type == FundType.ETF:
            return {
                'trading': 'Intraday on exchange (like stocks)',
                'pricing': 'Market price (may differ from NAV)',
                'minimum': '$0 (just buy 1 share)',
                'liquidity': 'High (trade anytime market open)',
                'tax_efficiency': 'High (in-kind redemptions)',
                'costs': 'Commission + spread',
                'management': 'Usually passive (index tracking)',
                'typical_expense_ratio': '0.03-0.20%'
            }
        elif self.fund_type == FundType.MUTUAL_FUND:
            return {
                'trading': 'Once daily after market close',
                'pricing': 'NAV (Net Asset Value) at 4pm ET',
                'minimum': '$1,000-$3,000 typically',
                'liquidity': 'Once per day',
                'tax_efficiency': 'Lower (cash redemptions)',
                'costs': 'No commission (buy direct)',
                'management': 'Can be active or passive',
                'typical_expense_ratio': '0.10-1.00%'
            }
        else:  # Closed-End Fund
            return {
                'trading': 'Intraday on exchange',
                'pricing': 'Market price (often at premium/discount to NAV)',
                'minimum': '$0 (buy shares)',
                'liquidity': 'Variable',
                'tax_efficiency': 'Moderate',
                'costs': 'Commission',
                'management': 'Usually active',
                'typical_expense_ratio': '0.50-1.50%'
            }

# Major funds
funds = [
    InvestmentFund(
        name="SPDR S&P 500 ETF",
        ticker="SPY",
        fund_type=FundType.ETF,
        expense_ratio=0.0945,  # 0.09%
        aum=450_000_000_000,  # $450B
        inception_date="1993-01-22",
        trading_mechanism="Intraday",
        pricing="Market price",
        minimum_investment=0,
        tax_efficiency="High"
    ),
    InvestmentFund(
        name="Vanguard 500 Index Fund",
        ticker="VFINX",
        fund_type=FundType.MUTUAL_FUND,
        expense_ratio=0.14,  # 0.14%
        aum=850_000_000_000,  # $850B (largest mutual fund!)
        inception_date="1976-08-31",
        trading_mechanism="Once daily",
        pricing="NAV at 4pm",
        minimum_investment=3000,
        tax_efficiency="Moderate"
    ),
    InvestmentFund(
        name="Invesco QQQ ETF",
        ticker="QQQ",
        fund_type=FundType.ETF,
        expense_ratio=0.20,
        aum=200_000_000_000,  # $200B
        inception_date="1999-03-10",
        trading_mechanism="Intraday",
        pricing="Market price",
        minimum_investment=0,
        tax_efficiency="High"
    )
]

print("=== ETF vs Mutual Fund Comparison ===\\n")

for fund in funds:
    chars = fund.get_characteristics()
    print(f"{fund.name} ({fund.ticker}) - {fund.fund_type.value}")
    print(f"  AUM: \${fund.aum/1e9:.0f}B")
    print(f"  Expense Ratio: {fund.expense_ratio:.2f}%")
    print(f"  Trading: {chars['trading']}")
    print(f"  Pricing: {chars['pricing']}")
    print(f"  Minimum: \${fund.minimum_investment:,.0f}")
    print(f"  Tax Efficiency: {chars['tax_efficiency']}\\n")
\`\`\`

**Key Insight**: ETFs revolutionized investing by combining mutual fund diversification with stock-like trading.

---

## Index Funds and Passive Investing

"Don't look for the needle in the haystack. Just buy the haystack." - Jack Bogle

\`\`\`python
class IndexFund:
    """
    Model index fund that tracks a benchmark
    """
    
    def __init__(self, 
                 name: str,
                 benchmark_index: str,
                 expense_ratio: float):
        self.name = name
        self.benchmark = benchmark_index
        self.expense_ratio = expense_ratio
        self.holdings = []
    
    def calculate_tracking_error (self,
                                 fund_returns: np.array,
                                 index_returns: np.array) -> Dict:
        """
        Tracking error = volatility of (fund returns - index returns)
        
        Good index fund: Low tracking error (<0.10%)
        """
        # Difference in returns
        active_returns = fund_returns - index_returns
        
        # Tracking error (annualized std dev of difference)
        tracking_error = np.std (active_returns) * np.sqrt(252)
        
        # Mean difference (should be close to expense ratio)
        mean_underperformance = np.mean (active_returns) * 252
        
        return {
            'tracking_error': tracking_error,
            'avg_underperformance': mean_underperformance,
            'expense_ratio': self.expense_ratio,
            'quality': 'Excellent' if tracking_error < 0.001 else 'Good' if tracking_error < 0.005 else 'Poor'
        }
    
    @staticmethod
    def passive_vs_active_returns (years: int = 10) -> Dict:
        """
        Historical performance: Passive vs Active
        
        Fact: 90%+ of active managers underperform over 10 years
        """
        # S&P 500 historical return
        passive_annual_return = 0.10  # 10% annually
        passive_expense = 0.0003  # 0.03%
        passive_net = passive_annual_return - passive_expense
        
        # Active fund average
        active_annual_return = 0.10  # Same gross return
        active_expense = 0.0075  # 0.75% (much higher fees)
        active_underperformance = 0.015  # -1.5% from bad picks
        active_net = active_annual_return - active_expense - active_underperformance
        
        # Compound over time
        passive_total = (1 + passive_net) ** years
        active_total = (1 + active_net) ** years
        
        # $10K invested
        initial = 10000
        passive_value = initial * passive_total
        active_value = initial * active_total
        
        return {
            'years': years,
            'passive_return': passive_net * 100,
            'active_return': active_net * 100,
            'passive_final': passive_value,
            'active_final': active_value,
            'difference': passive_value - active_value,
            'passive_wins': passive_value > active_value
        }

# Simulate tracking
index_fund = IndexFund(
    name="Vanguard S&P 500",
    benchmark_index="S&P 500",
    expense_ratio=0.0003
)

# Generate fake returns
np.random.seed(42)
days = 252 * 10  # 10 years
index_returns = np.random.normal(0.0004, 0.01, days)  # 0.04% daily return, 1% vol
fund_returns = index_returns - (index_fund.expense_ratio / 252) + np.random.normal(0, 0.0001, days)

tracking = index_fund.calculate_tracking_error (fund_returns, index_returns)

print("\\n=== Index Fund Tracking Analysis ===\\n")
print(f"Fund: {index_fund.name}")
print(f"Benchmark: {index_fund.benchmark}")
print(f"Expense Ratio: {index_fund.expense_ratio*100:.3f}%")
print(f"Tracking Error: {tracking['tracking_error']*100:.3f}%")
print(f"Avg Underperformance: {tracking['avg_underperformance']*100:.2f}%")
print(f"Quality: {tracking['quality']}")

# Passive vs Active
comparison = IndexFund.passive_vs_active_returns (years=10)

print(f"\\n\\n=== 10-Year Performance: Passive vs Active ===\\n")
print(f"Passive Index Fund:")
print(f"  Net Return: {comparison['passive_return']:.2f}% annually")
print(f"  $10K grows to: \${comparison['passive_final']:,.0f}")
print(f"\\nActive Management:")
print(f"  Net Return: {comparison['active_return']:.2f}% annually")
print(f"  $10K grows to: \${comparison['active_final']:,.0f}")
print(f"\\nDifference: \${comparison['difference']:,.0f} (passive wins!)")
\`\`\`

**Why Passive Wins:**
1. **Fees**: 0.03% vs 0.75%+ (0.72% difference × 10 years = 7.5%!)
2. **Trading costs**: Index funds trade rarely, active funds trade constantly
3. **Tax efficiency**: Low turnover = lower taxes
4. **Skill is rare**: Few managers beat the market consistently

---

## Factor Investing: Smart Beta

Beyond market-cap-weighted indexes - target specific characteristics.

\`\`\`python
class FactorInvesting:
    """
    Factor investing: Target stocks with specific characteristics
    """
    
    @staticmethod
    def explain_factors() -> Dict:
        """
        Major factors that historically outperform
        """
        return {
            'Value': {
                'definition': 'Cheap stocks (low P/E, P/B)',
                'rationale': 'Mean reversion - cheap stocks get repriced',
                'historical_premium': '+2-3% annually vs market',
                'etf_examples': 'VTV, IVE, VLUE',
                'risk': 'Can underperform for years (2010-2020)',
                'metric': 'P/E, P/B, P/S ratios'
            },
            'Momentum': {
                'definition': 'Stocks with strong recent performance',
                'rationale': 'Trends persist - winners keep winning',
                'historical_premium': '+3-5% annually',
                'etf_examples': 'MTUM, PDP, JMOM',
                'risk': 'Crashes hard in reversals',
                'metric': '12-month return (skip most recent month)'
            },
            'Quality': {
                'definition': 'Profitable, stable companies',
                'rationale': 'Good businesses compound over time',
                'historical_premium': '+1-2% annually',
                'etf_examples': 'QUAL, JQUA, SPHQ',
                'risk': 'Lower upside in bull markets',
                'metric': 'ROE, debt/equity, earnings stability'
            },
            'Size': {
                'definition': 'Small-cap stocks',
                'rationale': 'Small companies grow faster',
                'historical_premium': '+2% annually',
                'etf_examples': 'IWM, VB, IJR',
                'risk': 'Higher volatility, liquidity issues',
                'metric': 'Market capitalization'
            },
            'Low Volatility': {
                'definition': 'Stable, low-risk stocks',
                'rationale': 'Low-vol stocks have higher risk-adjusted returns',
                'historical_premium': '+1% (but with lower volatility)',
                'etf_examples': 'USMV, SPLV, LVHD',
                'risk': 'Underperforms in strong bull markets',
                'metric': 'Beta, standard deviation'
            }
        }
    
    def build_multifactor_portfolio (self,
                                   universe: List[str],
                                   factors: Dict[str, float]) -> Dict:
        """
        Combine multiple factors
        
        Example: 50% Value + 30% Momentum + 20% Quality
        """
        # Score each stock on each factor
        scores = {}
        for stock in universe:
            stock_score = 0
            for factor, weight in factors.items():
                factor_score = self.calculate_factor_score (stock, factor)
                stock_score += factor_score * weight
            scores[stock] = stock_score
        
        # Rank and select top stocks
        ranked_stocks = sorted (scores.items(), key=lambda x: x[1], reverse=True)
        top_stocks = ranked_stocks[:50]  # Top 50
        
        # Equal weight
        weight = 1.0 / len (top_stocks)
        
        return {
            'holdings': {stock: weight for stock, score in top_stocks},
            'factor_exposures': factors,
            'num_holdings': len (top_stocks),
            'diversification': 'Equal weighted'
        }
    
    def calculate_factor_score (self, stock: str, factor: str) -> float:
        """
        Score stock on specific factor
        
        Higher score = better fit for factor
        """
        # Simplified scoring (in production: use real data)
        if factor == 'Value':
            pe_ratio = self.get_pe_ratio (stock)
            return 1 / pe_ratio if pe_ratio > 0 else 0  # Lower P/E = higher score
        
        elif factor == 'Momentum':
            return_12m = self.get_12month_return (stock)
            return return_12m  # Higher return = higher score
        
        elif factor == 'Quality':
            roe = self.get_roe (stock)
            return roe  # Higher ROE = higher score
        
        return 0

# Explain factors
factor_system = FactorInvesting()
factors = factor_system.explain_factors()

print("\\n=== Factor Investing: Smart Beta ===\\n")

for factor_name, details in factors.items():
    print(f"{factor_name} Factor:")
    print(f"  Definition: {details['definition']}")
    print(f"  Rationale: {details['rationale']}")
    print(f"  Historical Premium: {details['historical_premium']}")
    print(f"  ETF Examples: {details['etf_examples']}")
    print(f"  Risk: {details['risk']}\\n")

# Build multi-factor portfolio
portfolio = factor_system.build_multifactor_portfolio(
    universe=['AAPL', 'MSFT', 'GOOGL'],  # Simplified
    factors={
        'Value': 0.40,
        'Momentum': 0.30,
        'Quality': 0.30
    }
)

print("\\nMulti-Factor Portfolio:")
print(f"  Holdings: {portfolio['num_holdings']} stocks")
print(f"  Factor Mix: 40% Value, 30% Momentum, 30% Quality")
\`\`\`

**Factor Investing in Practice:**
- **Fama-French 3-Factor Model**: Market + Size + Value
- **Carhart 4-Factor**: Add Momentum
- **Fama-French 5-Factor**: Add Profitability + Investment

---

## ETF Creation/Redemption Mechanism

What makes ETFs tax-efficient and keeps price close to NAV.

\`\`\`python
class ETFCreationRedemption:
    """
    Model the creation/redemption process
    
    This is what makes ETFs special!
    """
    
    def __init__(self, etf_ticker: str):
        self.etf_ticker = etf_ticker
        self.nav = 100.00  # Net Asset Value per share
        self.market_price = 100.05  # Trading price
        self.creation_unit_size = 50000  # Shares per creation unit
    
    def check_arbitrage_opportunity (self) -> Dict:
        """
        If ETF trades at premium/discount, Authorized Participants arbitrage
        
        Premium: Market Price > NAV → Create shares
        Discount: Market Price < NAV → Redeem shares
        """
        difference = self.market_price - self.nav
        difference_pct = (difference / self.nav) * 100
        
        if abs (difference_pct) > 0.1:  # >0.1% = arbitrage opportunity
            if difference > 0:
                action = "CREATE"
                trade = "Buy underlying stocks, create ETF shares, sell ETF"
                profit_per_share = difference
            else:
                action = "REDEEM"
                trade = "Buy ETF shares, redeem for stocks, sell stocks"
                profit_per_share = abs (difference)
            
            profit_per_unit = profit_per_share * self.creation_unit_size
            
            return {
                'nav': self.nav,
                'market_price': self.market_price,
                'difference': difference,
                'difference_pct': difference_pct,
                'arbitrage': True,
                'action': action,
                'trade': trade,
                'profit_per_share': profit_per_share,
                'profit_per_unit': profit_per_unit
            }
        else:
            return {
                'nav': self.nav,
                'market_price': self.market_price,
                'difference': difference,
                'difference_pct': difference_pct,
                'arbitrage': False,
                'action': 'NONE',
                'note': 'Price is fair, no arbitrage'
            }
    
    def create_shares (self, underlying_stocks: Dict[str, int]) -> Dict:
        """
        Authorized Participant creates ETF shares
        
        1. Buy underlying stocks
        2. Deliver to ETF
        3. Receive ETF shares
        4. Sell ETF shares on market
        """
        # Calculate cost of underlying basket
        basket_cost = sum(
            stock_price * quantity 
            for stock_price, quantity in underlying_stocks.items()
        )
        
        # Receive ETF shares
        etf_shares_received = self.creation_unit_size
        
        # Sell ETF shares at market price
        etf_revenue = etf_shares_received * self.market_price
        
        # Profit
        profit = etf_revenue - basket_cost
        
        return {
            'action': 'CREATE',
            'basket_cost': basket_cost,
            'etf_shares_received': etf_shares_received,
            'etf_market_price': self.market_price,
            'etf_revenue': etf_revenue,
            'profit': profit,
            'explanation': 'Bought stocks, created ETF shares, sold at premium'
        }
    
    def redeem_shares (self, etf_shares: int) -> Dict:
        """
        Authorized Participant redeems ETF shares
        
        1. Buy ETF shares on market
        2. Deliver to ETF
        3. Receive underlying stocks
        4. Sell stocks
        """
        # Buy ETF shares
        etf_cost = etf_shares * self.market_price
        
        # Receive underlying stocks
        stocks_received_value = etf_shares * self.nav
        
        # Sell stocks
        stock_revenue = stocks_received_value
        
        # Profit
        profit = stock_revenue - etf_cost
        
        return {
            'action': 'REDEEM',
            'etf_shares': etf_shares,
            'etf_cost': etf_cost,
            'stocks_received_value': stocks_received_value,
            'stock_revenue': stock_revenue,
            'profit': profit,
            'explanation': 'Bought ETF at discount, redeemed for stocks, sold stocks'
        }

# Example
etf = ETFCreationRedemption('SPY')
etf.market_price = 100.15  # Trading at premium

arb = etf.check_arbitrage_opportunity()

print("\\n=== ETF Arbitrage Mechanism ===\\n")
print(f"ETF: {etf.etf_ticker}")
print(f"NAV: \${arb['nav']:.2f}")
print(f"Market Price: \${arb['market_price']:.2f}")
print(f"Difference: \${arb['difference']:.2f} ({arb['difference_pct']:.2f}%)")
print(f"Arbitrage Opportunity: {'YES' if arb['arbitrage'] else 'NO'}")

if arb['arbitrage']:
    print(f"\\nAction: {arb['action']}")
    print(f"Trade: {arb['trade']}")
    print(f"Profit per Unit: \${arb['profit_per_unit']:,.0f}")
\`\`\`

**Why This Matters:**
1. **Keeps price close to NAV**: Arbitrageurs eliminate premiums/discounts
2. **Tax efficiency**: In-kind redemptions don't trigger capital gains
3. **Liquidity**: Even illiquid ETFs can be created/redeemed

---

## Building an ETF Trading System

\`\`\`python
class ETFTradingSystem:
    """
    Production ETF trading system
    """
    
    def __init__(self):
        self.positions = {}
        self.portfolio_value = 1000000
    
    def calculate_etf_premium_discount (self,
                                       etf_ticker: str) -> Dict:
        """
        Monitor ETF premium/discount to NAV
        
        Trade opportunity if >0.5% deviation
        """
        market_price = self.get_market_price (etf_ticker)
        nav = self.get_nav (etf_ticker)
        
        premium_discount = ((market_price - nav) / nav) * 100
        
        return {
            'ticker': etf_ticker,
            'market_price': market_price,
            'nav': nav,
            'premium_discount_pct': premium_discount,
            'action': 'BUY' if premium_discount < -0.5 else 'SELL' if premium_discount > 0.5 else 'HOLD'
        }
    
    def sector_rotation (self, economic_indicator: str) -> List[str]:
        """
        Rotate between sector ETFs based on economic cycle
        
        Expansion: Tech (XLK), Discretionary (XLY)
        Peak: Energy (XLE), Materials (XLB)
        Contraction: Staples (XLP), Healthcare (XLV)
        Trough: Financials (XLF), Industrials (XLI)
        """
        cycles = {
            'Expansion': ['XLK', 'XLY', 'XLI'],  # Growth sectors
            'Peak': ['XLE', 'XLB'],  # Commodities
            'Contraction': ['XLP', 'XLV', 'XLU'],  # Defensive
            'Trough': ['XLF', 'XLI']  # Recovery
        }
        
        return cycles.get (economic_indicator, ['SPY'])  # Default to SPY
    
    def build_core_satellite_portfolio (self) -> Dict:
        """
        Core: 70% broad market (SPY, VTI)
        Satellite: 30% tactical bets (sector ETFs, factor ETFs)
        """
        total = self.portfolio_value
        
        return {
            'core': {
                'SPY': total * 0.50,  # S&P 500
                'VTI': total * 0.20   # Total US market
            },
            'satellite': {
                'QQQ': total * 0.10,   # Nasdaq 100 (tech)
                'IWM': total * 0.05,   # Small cap
                'VTV': total * 0.05,   # Value
                'MTUM': total * 0.05,  # Momentum
                'USMV': total * 0.05   # Low volatility
            },
            'allocation': {
                'core_pct': 70,
                'satellite_pct': 30
            }
        }

# Usage
etf_system = ETFTradingSystem()

# Monitor premium/discount
spy_analysis = etf_system.calculate_etf_premium_discount('SPY')
print("\\n=== ETF Trading System ===\\n")
print(f"SPY Analysis:")
print(f"  Market Price: \${spy_analysis['market_price']:.2f}")
print(f"  NAV: \${spy_analysis['nav']:.2f}")
print(f"  Premium/Discount: {spy_analysis['premium_discount_pct']:.2f}%")
print(f"  Action: {spy_analysis['action']}")

# Core-satellite portfolio
portfolio = etf_system.build_core_satellite_portfolio()
print(f"\\n\\nCore-Satellite Portfolio (\$1M):")
print(f"\\nCore (70% - Low-cost index):")
for etf, amount in portfolio['core'].items():
    print(f"  {etf}: \${amount:,.0f}")

print(f"\\nSatellite (30% - Tactical):")
for etf, amount in portfolio['satellite'].items():
    print(f"  {etf}: \${amount:,.0f}")
\`\`\`

---

## Summary

**Key Takeaways:**

1. **ETFs vs Mutual Funds**: ETFs trade intraday, more tax-efficient, lower minimums
2. **Passive Beats Active**: 90%+ of active managers underperform over 10+ years
3. **Fees Matter**: 0.70% fee difference = 7.5%+ lost over 10 years
4. **Factor Investing**: Target value, momentum, quality for potential outperformance
5. **Creation/Redemption**: Keeps ETF price close to NAV, enables tax efficiency
6. **Diversification**: Own entire market with single purchase

**For Engineers:**
- ETF data easier than individual stocks (fewer tickers)
- Premium/discount monitoring for arbitrage
- Factor models require fundamental data
- Low turnover = easier backtesting
- Tax efficiency = more complex P&L accounting

**Next Steps:**
- Module 7: Portfolio Management (Markowitz, risk parity)
- Module 9: Factor models implementation
- Module 18: ML for ETF selection

You now understand ETFs - the building blocks of modern portfolios!
`,
  exercises: [
    {
      prompt:
        'Build an ETF premium/discount monitor that fetches real-time ETF prices and NAVs, calculates premium/discount %, alerts when >0.5% deviation occurs, and tracks historical patterns. Monitor SPY, QQQ, IWM, and sector ETFs.',
      solution:
        '// Implementation: 1) Connect to market data API (Alpha Vantage, IEX), 2) Fetch ETF market prices (real-time or 15-min delayed), 3) Scrape NAV from ETF provider websites (State Street for SPY, etc.), 4) Calculate (Price - NAV) / NAV * 100, 5) Store in database with timestamp, 6) Alert if abs (premium/discount) > 0.5%, 7) Create dashboard showing current and historical premium/discount charts, 8) Identify patterns (ETFs trading at consistent discounts = buy opportunity)',
    },
    {
      prompt:
        'Create a factor ETF backtesting system that compares Value (VTV), Momentum (MTUM), Quality (QUAL), Low Vol (USMV), and Size (IWM) ETFs over 10 years. Calculate returns, volatility, Sharpe ratios, max drawdowns, and optimal factor allocations using Markowitz optimization.',
      solution:
        '// Implementation: 1) Fetch 10 years of daily prices for factor ETFs using yfinance/Alpha Vantage, 2) Calculate daily/monthly returns, 3) Compute metrics: CAGR, annual volatility, Sharpe ratio, max drawdown, 4) Calculate correlation matrix, 5) Implement Markowitz optimization to find max Sharpe portfolio, 6) Backtest equal-weight, market-cap-weight, and optimized portfolios, 7) Visualize cumulative returns, drawdowns, rolling Sharpe ratios, 8) Compare factor timing vs buy-and-hold',
    },
  ],
};
