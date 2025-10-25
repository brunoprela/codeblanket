export const investmentVehiclesProducts = {
  title: 'Investment Vehicles & Products',
  id: 'investment-vehicles-products',
  content: `
# Investment Vehicles & Products

## Introduction

**Investment vehicles** are structures that hold and manage financial assets. Understanding these is critical for building financial systems because:

- **Different structures, different engineering**: ETFs need arbitrage mechanisms, mutual funds need NAV calculation
- **Regulatory differences**: Each vehicle has unique compliance requirements
- **Data requirements**: Different reporting standards, pricing mechanisms
- **User experience**: Different vehicles suit different user needs

This section covers all major investment products you'll encounter when building financial platforms.

---

## Individual Securities

### Stocks (Equities)

**Direct ownership** in companies. Most straightforward investment vehicle.

**Characteristics**:
- Trade on exchanges (NYSE, NASDAQ)
- Real-time pricing during market hours
- No management fees (beyond brokerage commission)
- Tax: Capital gains when sold, dividends taxed annually

**Use Cases**:
- Active investors who pick stocks
- Long-term buy-and-hold investors
- Corporate insiders (own company stock)

\`\`\`python
"""
Stock Portfolio Tracker
"""
import yfinance as yf
import pandas as pd

class StockPortfolio:
    """Track individual stock positions"""
    
    def __init__(self):
        self.positions = {}  # {ticker: {'shares': int, 'cost_basis': float}}
    
    def add_position (self, ticker: str, shares: int, price: float):
        """Add or update position"""
        if ticker in self.positions:
            # Update cost basis (weighted average)
            old_shares = self.positions[ticker]['shares']
            old_basis = self.positions[ticker]['cost_basis']
            new_shares = old_shares + shares
            new_basis = (old_shares * old_basis + shares * price) / new_shares
            
            self.positions[ticker] = {
                'shares': new_shares,
                'cost_basis': new_basis
            }
        else:
            self.positions[ticker] = {
                'shares': shares,
                'cost_basis': price
            }
    
    def get_portfolio_value (self) -> pd.DataFrame:
        """Calculate current portfolio value"""
        portfolio_data = []
        
        for ticker, position in self.positions.items():
            # Get current price
            stock = yf.Ticker (ticker)
            current_price = stock.info['currentPrice']
            
            shares = position['shares']
            cost_basis = position['cost_basis']
            current_value = shares * current_price
            cost = shares * cost_basis
            gain_loss = current_value - cost
            gain_loss_pct = (gain_loss / cost) * 100
            
            portfolio_data.append({
                'Ticker': ticker,
                'Shares': shares,
                'Cost Basis': f'\${cost_basis:.2f}',
'Current Price': f'\${current_price:.2f}',
    'Cost': f'\${cost:.2f}',
        'Current Value': f'\${current_value:.2f}',
            'Gain/Loss': f'\${gain_loss:.2f}',
                'Return': f'{gain_loss_pct:+.2f}%'
            })

df = pd.DataFrame (portfolio_data)
        
        # Calculate totals
total_cost = sum (position['shares'] * position['cost_basis'] 
                         for position in self.positions.values())
    total_value = sum (yf.Ticker (ticker).info['currentPrice'] * position['shares']
                          for ticker, position in self.positions.items())
    total_gain = total_value - total_cost

print("\\n=== Stock Portfolio ===")
print(df.to_string (index = False))
print(f"\\nTotal Cost: \${total_cost:,.2f}")
print(f"Total Value: \${total_value:,.2f}")
print(f"Total Gain/Loss: \${total_gain:,.2f} ({(total_gain/total_cost)*100:+.2f}%)")

return df


# Example usage
portfolio = StockPortfolio()
portfolio.add_position('AAPL', 100, 150.00)
portfolio.add_position('MSFT', 50, 300.00)
portfolio.add_position('GOOGL', 25, 140.00)

portfolio.get_portfolio_value()
\`\`\`

---

## Mutual Funds

### Active Management

**Mutual funds** pool money from many investors to buy diversified portfolios, managed by professional portfolio managers.

**Key Characteristics**:
- **Active management**: Manager picks stocks (vs passive index tracking)
- **End-of-day pricing**: NAV calculated once daily after market close
- **Higher fees**: Expense ratios 0.5-2% annually
- **Minimum investments**: Often $1,000-$10,000
- **Tax inefficient**: Capital gains distributed to all shareholders

**Net Asset Value (NAV) Calculation**:

\`\`\`
NAV = (Total Assets - Total Liabilities) / Number of Shares Outstanding

Example:
Assets: $100M in stocks + $5M cash = $105M
Liabilities: $2M expenses = $2M
Shares Outstanding: 10M
NAV = (\$105M - $2M) / 10M = $10.30 per share
\`\`\`

\`\`\`python
"""
Mutual Fund NAV Calculator
"""

class MutualFund:
    """Simulate mutual fund NAV calculation"""
    
    def __init__(self, name: str, ticker: str):
        self.name = name
        self.ticker = ticker
        self.holdings = {}  # {stock_ticker: shares}
        self.cash = 0
        self.shares_outstanding = 0
        self.expenses = 0
    
    def calculate_nav (self) -> float:
        """Calculate Net Asset Value"""
        # Calculate total assets
        assets_value = self.cash
        
        for ticker, shares in self.holdings.items():
            stock = yf.Ticker (ticker)
            price = stock.info.get('currentPrice', 0)
            assets_value += shares * price
        
        # Calculate liabilities (accrued expenses)
        liabilities = self.expenses
        
        # Calculate NAV
        if self.shares_outstanding > 0:
            nav = (assets_value - liabilities) / self.shares_outstanding
        else:
            nav = 10.00  # Initial NAV
        
        return nav
    
    def purchase_shares (self, investor: str, amount: float):
        """Investor purchases fund shares"""
        nav = self.calculate_nav()
        shares_purchased = amount / nav
        
        self.cash += amount
        self.shares_outstanding += shares_purchased
        
        print(f"{investor} purchased {shares_purchased:.2f} shares at \${nav:.2f} NAV")
print(f"  Investment: \${amount:.2f}")

return shares_purchased
    
    def redeem_shares (self, investor: str, shares: float):
"""Investor redeems (sells) fund shares"""
nav = self.calculate_nav()
redemption_amount = shares * nav

if redemption_amount > self.cash:
            # Need to sell holdings to raise cash
print(f"  Warning: Selling holdings to meet redemption")

self.cash -= redemption_amount
self.shares_outstanding -= shares

print(f"{investor} redeemed {shares:.2f} shares at \${nav:.2f} NAV")
print(f"  Proceeds: \${redemption_amount:.2f}")

return redemption_amount


# Example: Fidelity Contrafund simulation
fund = MutualFund("Fidelity Contrafund", "FCNTX")

# Initial setup
fund.cash = 1_000_000
fund.shares_outstanding = 100_000  # Initial NAV = $10.00

# Fund buys stocks
fund.holdings = {
    'AAPL': 1000,
    'MSFT': 500,
    'GOOGL': 300
}
fund.cash = 100_000  # Remaining cash after buying stocks

# Calculate NAV
nav = fund.calculate_nav()
print(f"\\nCurrent NAV: \${nav:.2f}")

# Investor buys into fund
fund.purchase_shares("Alice", 50_000)

# New NAV after purchase
nav = fund.calculate_nav()
print(f"\\nUpdated NAV: \${nav:.2f}")
\`\`\`

**Pros**:
- Professional management
- Diversification (own hundreds of stocks with one purchase)
- Research and analysis included

**Cons**:
- Higher fees eat into returns
- Tax inefficient (you pay taxes on fund's trading)
- Only trade at end-of-day NAV (can't exit quickly)
- Most active funds underperform index

---

## Exchange-Traded Funds (ETFs)

### Passive Index Tracking (Mostly)

**ETFs** combine best of stocks and mutual funds: diversification + real-time trading + low fees.

**Key Differences vs Mutual Funds**:

| Feature | ETF | Mutual Fund |
|---------|-----|-------------|
| **Trading** | Intraday (like stocks) | End-of-day only |
| **Pricing** | Real-time market price | NAV at 4pm |
| **Fees** | 0.03-0.50% | 0.5-2% |
| **Minimum** | Price of 1 share (~$50-500) | $1,000-10,000 |
| **Tax Efficiency** | High (in-kind redemptions) | Low (taxable distributions) |
| **Management** | Mostly passive | Mostly active |

**ETF Creation/Redemption Mechanism** (Key Engineering Concept):

\`\`\`python
"""
ETF Arbitrage Mechanism
How ETFs stay close to NAV through arbitrage
"""

class ETF:
    """Simplified ETF with creation/redemption"""
    
    def __init__(self, name: str, ticker: str, underlying_index: list):
        self.name = name
        self.ticker = ticker
        self.underlying_index = underlying_index  # List of (ticker, weight)
        self.shares_outstanding = 0
    
    def calculate_nav (self) -> float:
        """Calculate NAV based on underlying holdings"""
        nav = 0
        for ticker, weight in self.underlying_index:
            stock = yf.Ticker (ticker)
            price = stock.info.get('currentPrice', 0)
            nav += price * weight
        return nav
    
    def get_market_price (self) -> float:
        """Simulate market price (can differ from NAV)"""
        # In reality, pulled from exchange
        # For simulation, add small random premium/discount
        nav = self.calculate_nav()
        import random
        premium = random.uniform(-0.005, 0.005)  # ±0.5%
        return nav * (1 + premium)
    
    def create_shares (self, authorized_participant: str, basket_value: float):
        """
        Creation: AP delivers basket of stocks, receives ETF shares
        
        This is HOW ETFs are created. Only large institutions (APs) can do this.
        """
        nav = self.calculate_nav()
        creation_unit = 50_000  # Standard creation unit size
        shares_created = creation_unit
        
        print(f"\\n=== ETF Creation ===")
        print(f"AP: {authorized_participant}")
        print(f"Delivers stock basket worth: \${basket_value:,.0f}")
print(f"Receives ETF shares: {shares_created:,}")
print(f"NAV per share: \${nav:.2f}")

self.shares_outstanding += shares_created

return shares_created
    
    def redeem_shares (self, authorized_participant: str, shares: int):
"""
Redemption: AP delivers ETF shares, receives basket of stocks
        
        This is HOW investors can exit ETF positions at NAV.
        """
nav = self.calculate_nav()
basket_value = shares * nav

print(f"\\n=== ETF Redemption ===")
print(f"AP: {authorized_participant}")
print(f"Delivers ETF shares: {shares:,}")
print(f"Receives stock basket worth: \${basket_value:,.0f}")
print(f"NAV per share: \${nav:.2f}")

self.shares_outstanding -= shares

return basket_value
    
    def check_arbitrage_opportunity (self):
"""
        Check if market price differs from NAV(arbitrage opportunity)
        
        If ETF trades at premium: AP creates shares (profit)
        If ETF trades at discount: AP redeems shares (profit)
"""
nav = self.calculate_nav()
market_price = self.get_market_price()

diff = market_price - nav
diff_pct = (diff / nav) * 100

print(f"\\n=== Arbitrage Check ===")
print(f"NAV: \${nav:.2f}")
print(f"Market Price: \${market_price:.2f}")
print(f"Difference: \${diff:.4f} ({diff_pct:+.3f}%)")

if diff_pct > 0.1:  # Premium > 0.1 %
    print("→ PREMIUM: AP should CREATE shares (sell ETF, profit)")
        elif diff_pct < -0.1:  # Discount > 0.1 %
    print("→ DISCOUNT: AP should REDEEM shares (buy ETF, profit)")
        else:
print("→ Fair value: No arbitrage opportunity")


# Example: SPY(S & P 500 ETF)
spy = ETF("SPDR S&P 500 ETF", "SPY", [
    ('AAPL', 0.07),  # 7 % weight
        ('MSFT', 0.06),
    ('GOOGL', 0.04),
    ('AMZN', 0.03),
    ('NVDA', 0.03),
    # ... (simplified, real SPY has 500 stocks)
])

# Check NAV
nav = spy.calculate_nav()
print(f"SPY NAV: \${nav:.2f}")

# Authorized Participant creates shares (when ETF at premium)
spy.create_shares("Goldman Sachs", 10_000_000)

# Check arbitrage opportunity
spy.check_arbitrage_opportunity()
\`\`\`

**Why This Matters for Engineers**:

The creation/redemption mechanism is **critical** to ETFs working. When you build:
- **Robo-advisors**: Need to understand ETF pricing for automatic rebalancing
- **Trading platforms**: ETF orders route differently than stocks
- **Risk systems**: ETFs track indices but can temporarily diverge

**Popular ETFs**:
- **SPY**: S&P 500 (\$450B AUM, 0.09% fee)
- **QQQ**: NASDAQ-100 (\$200B AUM, 0.20% fee)
- **VTI**: Total U.S. Stock Market (\$350B AUM, 0.03% fee)
- **AGG**: U.S. Aggregate Bonds (\$90B AUM, 0.03% fee)

---

## Target-Date Funds

### "Set It and Forget It"

**Target-date funds** automatically adjust asset allocation based on retirement date.

**Glide Path** concept:
\`\`\`
Age 25 (40 years to retirement):
- 90% stocks, 10% bonds

Age 45 (20 years to retirement):
- 70% stocks, 30% bonds

Age 65 (retirement):
- 40% stocks, 60% bonds

Age 75 (10 years into retirement):
- 30% stocks, 70% bonds
\`\`\`

\`\`\`python
"""
Target-Date Fund Simulator
"""

class TargetDateFund:
    """Auto-adjusting asset allocation"""
    
    def __init__(self, target_year: int):
        self.target_year = target_year
        self.current_year = 2024
    
    def calculate_allocation (self) -> dict:
        """
        Calculate stock/bond allocation based on years to retirement
        
        Rule of thumb: stocks% = 100 - (110 - years_to_retirement)
        More sophisticated: glide path with multiple asset classes
        """
        years_to_retirement = self.target_year - self.current_year
        
        if years_to_retirement > 40:
            stocks = 0.90
            bonds = 0.10
        elif years_to_retirement > 30:
            stocks = 0.85
            bonds = 0.15
        elif years_to_retirement > 20:
            stocks = 0.75
            bonds = 0.25
        elif years_to_retirement > 10:
            stocks = 0.60
            bonds = 0.40
        elif years_to_retirement > 0:
            stocks = 0.50
            bonds = 0.50
        else:  # In retirement
            years_in_retirement = abs (years_to_retirement)
            stocks = max(0.30, 0.50 - years_in_retirement * 0.02)
            bonds = 1 - stocks
        
        return {
            'stocks': stocks,
            'bonds': bonds,
            'years_to_retirement': years_to_retirement
        }
    
    def rebalance_portfolio (self, current_portfolio_value: float):
        """Calculate rebalancing trades"""
        allocation = self.calculate_allocation()
        
        target_stocks = current_portfolio_value * allocation['stocks']
        target_bonds = current_portfolio_value * allocation['bonds']
        
        print(f"\\n=== Target-Date Fund {self.target_year} ===")
        print(f"Years to Retirement: {allocation['years_to_retirement']}")
        print(f"\\nTarget Allocation:")
        print(f"  Stocks: {allocation['stocks']:.0%} (\${target_stocks:,.0f}) ")
print(f"  Bonds: {allocation['bonds']:.0%} (\${target_bonds:,.0f})")

return allocation


# Example: Vanguard Target Retirement 2050
tdf_2050 = TargetDateFund(2050)
tdf_2050.rebalance_portfolio(100_000)

# Different target dates
tdf_2030 = TargetDateFund(2030)
tdf_2030.rebalance_portfolio(100_000)

tdf_2065 = TargetDateFund(2065)
tdf_2065.rebalance_portfolio(100_000)
\`\`\`

**Pros**:
- Automatic rebalancing (no effort)
- Age-appropriate risk (less stocks as you age)
- One-fund solution (simple)

**Cons**:
- One-size-fits-all (ignores individual circumstances)
- Higher fees (0.12-0.50% vs 0.03% for index ETF)
- Less control (can't adjust allocation)

---

## Robo-Advisors

### Automated Portfolio Management

**Robo-advisors** (Betterment, Wealthfront, Schwab Intelligent Portfolios) use algorithms to manage portfolios.

**How They Work**:
1. **Risk assessment**: Questionnaire determines risk tolerance
2. **Asset allocation**: Algorithm assigns target allocation
3. **Auto-rebalancing**: Maintains target allocation
4. **Tax-loss harvesting**: Automatically harvest losses for tax savings
5. **Dividend reinvestment**: Auto-reinvest dividends

\`\`\`python
"""
Robo-Advisor Portfolio Builder
"""

class RoboAdvisor:
    """Simplified robo-advisor logic"""
    
    def __init__(self):
        self.risk_profiles = {
            'conservative': {'stocks': 0.30, 'bonds': 0.70},
            'moderate': {'stocks': 0.60, 'bonds': 0.40},
            'aggressive': {'stocks': 0.90, 'bonds': 0.10}
        }
    
    def assess_risk_tolerance (self, age: int, risk_preference: str, 
                              time_horizon: int) -> str:
        """
        Determine risk profile based on inputs
        
        Factors:
        - Age: Younger = more risk capacity
        - Preference: Self-reported risk tolerance
        - Time horizon: Longer = more risk capacity
        """
        if risk_preference == 'conservative':
            return 'conservative'
        elif risk_preference == 'aggressive' and age < 40 and time_horizon > 20:
            return 'aggressive'
        else:
            return 'moderate'
    
    def build_portfolio (self, risk_profile: str) -> dict:
        """
        Build diversified ETF portfolio
        
        Asset classes:
        - U.S. Large Cap (VTI)
        - International (VXUS)
        - Bonds (AGG)
        - Real Estate (VNQ)
        """
        allocation = self.risk_profiles[risk_profile]
        stock_pct = allocation['stocks']
        bond_pct = allocation['bonds']
        
        portfolio = {
            'VTI': stock_pct * 0.60,   # 60% of stocks = U.S. Large Cap
            'VXUS': stock_pct * 0.30,  # 30% of stocks = International
            'VNQ': stock_pct * 0.10,   # 10% of stocks = Real Estate
            'AGG': bond_pct             # 100% of bonds = U.S. Aggregate
        }
        
        return portfolio
    
    def rebalance_check (self, current_weights: dict, target_weights: dict,
                        threshold: float = 0.05) -> dict:
        """
        Check if rebalancing needed
        
        Only rebalance if drift > threshold (typically 5%)
        """
        needs_rebalance = False
        trades = {}
        
        for ticker in target_weights:
            current = current_weights.get (ticker, 0)
            target = target_weights[ticker]
            drift = abs (current - target)
            
            if drift > threshold:
                needs_rebalance = True
                trades[ticker] = target - current
        
        return {'needs_rebalance': needs_rebalance, 'trades': trades}
    
    def tax_loss_harvest (self, positions: dict) -> list:
        """
        Identify positions for tax-loss harvesting
        
        Sell losing positions, replace with similar ETF
        """
        harvest_opportunities = []
        
        for ticker, data in positions.items():
            if data['current_value'] < data['cost_basis']:
                loss = data['cost_basis'] - data['current_value']
                
                # Find replacement ETF (avoid wash sale)
                replacement_map = {
                    'VTI': 'ITOT',  # Vanguard → iShares Total Market
                    'VXUS': 'IXUS', # Vanguard → iShares International
                    'AGG': 'BND'    # iShares → Vanguard Bonds
                }
                
                harvest_opportunities.append({
                    'sell': ticker,
                    'buy': replacement_map.get (ticker, ticker),
                    'loss': loss
                })
        
        return harvest_opportunities


# Example: Betterment-style robo-advisor
robo = RoboAdvisor()

# Client onboarding
risk_profile = robo.assess_risk_tolerance (age=35, risk_preference='moderate', 
                                           time_horizon=30)
print(f"Risk Profile: {risk_profile}")

# Build portfolio
portfolio = robo.build_portfolio (risk_profile)
print(f"\\nTarget Portfolio:")
for ticker, weight in portfolio.items():
    print(f"  {ticker}: {weight:.1%}")

# Check rebalancing
current_weights = {'VTI': 0.40, 'VXUS': 0.15, 'VNQ': 0.05, 'AGG': 0.40}
rebalance = robo.rebalance_check (current_weights, portfolio)

if rebalance['needs_rebalance']:
    print("\\nRebalancing needed:")
    for ticker, trade in rebalance['trades'].items():
        print(f"  {ticker}: {trade:+.1%}")
\`\`\`

**Fees**: 0.25-0.50% annually (on top of ETF fees)
**Minimum**: $0-500 (lower than traditional advisors)
**Tax Alpha**: Tax-loss harvesting can add 0.5-1% annual return

---

## Key Takeaways

### Vehicle Comparison

| Vehicle | Liquidity | Fees | Tax Efficiency | Min Investment | Best For |
|---------|-----------|------|----------------|----------------|----------|
| **Individual Stocks** | High | Low (brokerage) | High | $50-500/share | Active investors |
| **Mutual Funds** | Low (EOD) | High (0.5-2%) | Low | $1K-10K | Traditional 401k |
| **ETFs** | High | Low (0.03-0.5%) | High | $50-500/share | DIY investors |
| **Target-Date** | Low (EOD) | Medium (0.1-0.5%) | Low | $1K | Hands-off |
| **Robo-Advisor** | High | Medium (+0.25%) | High | $0-500 | Automated |

### For Building Financial Products

When building a **robo-advisor**:
- Use ETFs (low fees, tax-efficient, liquid)
- Auto-rebalance quarterly (or threshold-based)
- Tax-loss harvest continuously
- Offer multiple risk profiles

When building a **trading platform**:
- Support all vehicle types (stocks, ETFs, mutual funds)
- Handle end-of-day NAV for mutual funds
- Real-time pricing for stocks/ETFs
- Fractional shares for accessibility

**Next section**: How Trading Actually Works (order flow, execution, settlement)
`,
};
