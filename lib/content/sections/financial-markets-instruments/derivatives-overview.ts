export const derivativesOverview = {
  title: 'Derivatives Overview',
  slug: 'derivatives-overview',
  description:
    'Master forwards, futures, options, and swaps - the building blocks of modern finance',
  content: `
# Derivatives Overview

## Introduction: The Power and Peril of Derivatives

Derivatives are financial contracts whose value is **derived** from an underlying asset (stock, bond, commodity, currency, or index). They're simultaneously:
- 🎯 **Essential risk management tools** (hedging)
- 💰 **Speculative vehicles** (directional bets)
- 🔧 **Arbitrage instruments** (exploiting mispricings)
- ⚠️ **Potential weapons of mass destruction** (Warren Buffett\'s words after 2008)

**Market Size**: The notional value of global derivatives exceeds **$600 TRILLION** (yes, trillion), more than 10x global GDP. Yet most people don't understand them.

**What you'll learn:**
- Four main derivative types: Forwards, Futures, Options, Swaps
- Why derivatives exist and who uses them
- Pricing fundamentals
- Risk mechanics (leverage amplifies everything)
- Building derivative pricing systems
- The role of derivatives in financial crises

---

## What Are Derivatives?

A derivative is a **contract** between two parties where:
1. **Value depends on an underlying asset** (stock, commodity, rate, etc.)
2. **Settlement happens in the future** (not immediate like spot trading)
3. **Leverage is inherent** (control large positions with small capital)

\`\`\`python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Literal
from enum import Enum
import numpy as np

class DerivativeType(Enum):
    """Main categories of derivatives"""
    FORWARD = "Forward Contract"
    FUTURE = "Futures Contract"
    OPTION = "Option"
    SWAP = "Swap"

@dataclass
class DerivativeContract:
    """
    Base class for all derivatives
    """
    contract_type: DerivativeType
    underlying_asset: str
    notional_amount: float
    contract_date: datetime
    maturity_date: datetime
    counterparty_a: str
    counterparty_b: str
    
    def time_to_maturity_days (self) -> int:
        """Days remaining until settlement"""
        return (self.maturity_date - datetime.now()).days
    
    def time_to_maturity_years (self) -> float:
        """Years remaining until settlement"""
        return self.time_to_maturity_days() / 365.25
    
    def is_expired (self) -> bool:
        """Check if contract has expired"""
        return datetime.now() >= self.maturity_date
    
    def calculate_leverage_ratio (self, upfront_margin: float) -> float:
        """
        Calculate leverage
        Leverage = Notional Amount / Margin Posted
        
        This is why derivatives are powerful and dangerous!
        """
        if upfront_margin == 0:
            return float('inf')
        return self.notional_amount / upfront_margin

# Example: Understanding leverage
oil_future = DerivativeContract(
    contract_type=DerivativeType.FUTURE,
    underlying_asset="Crude Oil (WTI)",
    notional_amount=100_000,  # 1,000 barrels × $100/barrel
    contract_date=datetime.now(),
    maturity_date=datetime.now() + timedelta (days=90),
    counterparty_a="Trader",
    counterparty_b="Exchange",
)

# Futures typically require 5-15% margin
margin_required = 10_000  # $10K margin
leverage = oil_future.calculate_leverage_ratio (margin_required)

print(f"Oil Futures Contract:")
print(f"Notional Value: \\$\{oil_future.notional_amount:,}")
print(f"Margin Required: \\$\{margin_required:,}")
print(f"Leverage: {leverage:.1f}x")
print(f"\\n⚠️  If oil moves 1%, your P&L moves {leverage}%!")
print(f"   Oil +10% = Your account +{leverage*0.10:.0f}% (+\\$\{oil_future.notional_amount * 0.10:,.0f})")
print(f"   Oil -10% = Your account -{leverage*0.10:.0f}% (-\\$\{oil_future.notional_amount * 0.10:,.0f})")
print(f"\\n⚡ With only \${margin_required:,}, you control \\$\{oil_future.notional_amount:,} of oil!")
\`\`\`

**Key Insight**: Derivatives provide **leverage** - small price moves in the underlying create large P&L swings in your account. This is both their power (amplify gains) and danger (amplify losses).

---

## Why Derivatives Exist: Three Primary Uses

### 1. Hedging (Risk Management)

**Problem**: You're exposed to price risk you don't want.

\`\`\`python
class HedgingExample:
    """
    Real-world hedging scenarios
    """
    
    @staticmethod
    def airline_fuel_hedge():
        """
        Airlines hedge jet fuel prices
        
        Problem: Fuel is 30% of operating costs, volatile prices hurt profits
        Solution: Lock in fuel price using futures
        """
        current_fuel_price = 3.00  # $/gallon
        annual_fuel_consumption = 100_000_000  # gallons
        
        # Scenario 1: Don't hedge
        unhedged_cost = current_fuel_price * annual_fuel_consumption
        
        # Scenario 2: Hedge with futures at $3.10/gallon
        hedged_price = 3.10
        hedged_cost = hedged_price * annual_fuel_consumption
        
        # What happens if fuel prices spike?
        future_fuel_price = 4.50  # +50% spike!
        
        unhedged_actual_cost = future_fuel_price * annual_fuel_consumption
        hedged_actual_cost = hedged_cost  # Still pay hedged price
        
        savings = unhedged_actual_cost - hedged_actual_cost
        
        return {
            'company': 'Major Airline',
            'hedging_strategy': 'Jet fuel futures',
            'hedged_price': hedged_price,
            'market_price': future_fuel_price,
            'unhedged_cost': unhedged_actual_cost,
            'hedged_cost': hedged_actual_cost,
            'savings': savings,
            'interpretation': f'Hedging saved \${savings / 1e6:.0f}M when fuel spiked'
        }

@staticmethod
    def exporter_currency_hedge():
"""
U.S.exporter hedges EUR / USD exchange rate

Problem: Selling to Europe, will receive €10M in 6 months
Risk: EUR might weaken vs USD → receive less $
Solution: Lock in exchange rate with forward contract
"""
euro_receivable = 10_000_000  # €10M
current_rate = 1.10  # €1 = $1.10
forward_rate = 1.08  # Lock in at $1.08
        
        # Expected revenue at current rate
expected_usd = euro_receivable * current_rate
        
        # Hedged revenue (locked in)
hedged_usd = euro_receivable * forward_rate
        
        # Scenario: EUR weakens to 1.00(10 % drop)
future_rate = 1.00
unhedged_usd = euro_receivable * future_rate

protection = hedged_usd - unhedged_usd

return {
    'company': 'U.S. Exporter',
    'receivable': f'€{euro_receivable/1e6:.0f}M',
    'forward_rate_locked': forward_rate,
    'market_rate_at_settlement': future_rate,
    'unhedged_revenue': unhedged_usd,
    'hedged_revenue': hedged_usd,
    'protection_value': protection,
    'interpretation': f'Forward contract protected \${protection/1e6:.1f}M of revenue'
        }

@staticmethod
    def portfolio_manager_equity_hedge():
"""
        Portfolio manager hedges market risk

Problem: Manage $100M stock portfolio, worried about crash
Solution: Buy S & P 500 put options
"""
portfolio_value = 100_000_000
spy_price = 450
put_strike = 430  # 4.4 % below current
put_premium = 5  # $5 per share
contracts_needed = portfolio_value / (spy_price * 100)  # Each option = 100 shares

total_premium = contracts_needed * put_premium * 100
        
        # Scenario: Market crashes 20 %
    crash_spy_price = 450 * 0.80  # $360
        
        # Unhedged loss
unhedged_loss = portfolio_value * 0.20
        
        # Put payoff = max(Strike - Spot, 0)
put_payoff_per_share = max (put_strike - crash_spy_price, 0)
total_put_payoff = contracts_needed * put_payoff_per_share * 100

net_loss_hedged = unhedged_loss - total_put_payoff + total_premium

return {
    'portfolio_value': portfolio_value,
    'hedge': 'S&P 500 Put Options',
    'cost_of_hedge': total_premium,
    'market_crash': '20%',
    'unhedged_loss': unhedged_loss,
    'put_protection': total_put_payoff,
    'net_loss_with_hedge': net_loss_hedged,
    'protection_ratio': (total_put_payoff / unhedged_loss),
    'interpretation': f'Put options protected {(total_put_payoff/unhedged_loss)*100:.0f}% of crash loss'
        }

# Run examples
print("=== Hedging Use Cases ===\\n")

airline = HedgingExample.airline_fuel_hedge()
print(f"1. {airline['company']} - {airline['hedging_strategy']}")
print(f"   {airline['interpretation']}")

exporter = HedgingExample.exporter_currency_hedge()
print(f"\\n2. {exporter['company']} - FX Forward")
print(f"   {exporter['interpretation']}")

pm = HedgingExample.portfolio_manager_equity_hedge()
print(f"\\n3. Portfolio Manager - {pm['hedge']}")
print(f"   {pm['interpretation']}")
print(f"\\n💡 All three used derivatives to REDUCE risk, not take on more risk!")
\`\`\`

### 2. Speculation (Directional Bets)

**Goal**: Make money from price movements (with leverage).

\`\`\`python
class SpeculationExample:
    """
    Speculative use of derivatives
    """
    
    @staticmethod
    def bull_futures_trade():
        """
        Trader believes oil will rise
        Uses futures for leveraged exposure
        """
        oil_price = 80  # $/barrel
        contract_size = 1000  # barrels per contract
        contracts = 5
        
        notional_value = oil_price * contract_size * contracts
        margin_required = notional_value * 0.10  # 10% initial margin
        leverage = notional_value / margin_required
        
        # Scenario: Oil rises to $90 (+12.5%)
        new_price = 90
        profit = (new_price - oil_price) * contract_size * contracts
        return_on_margin = profit / margin_required
        
        return {
            'position': 'Long Oil Futures',
            'contracts': contracts,
            'notional_value': notional_value,
            'margin_posted': margin_required,
            'leverage': leverage,
            'price_move': '+12.5%',
            'dollar_profit': profit,
            'return_on_margin': return_on_margin * 100,
            'comparison': f'{leverage:.0f}x levered return vs {0.125*100}% if bought oil directly'
        }
    
    @staticmethod
    def bear_put_option():
        """
        Trader believes stock will fall
        Buys put options for leveraged downside
        """
        stock_price = 100
        put_strike = 95
        put_premium = 2
        contracts = 10  # 1,000 shares
        
        cost = put_premium * 100 * contracts
        
        # Scenario: Stock crashes to $80 (-20%)
        new_price = 80
        put_value = max (put_strike - new_price, 0)
        payoff = put_value * 100 * contracts
        net_profit = payoff - cost
        return_pct = net_profit / cost
        
        # Compare to shorting stock
        short_profit = (stock_price - new_price) * 100 * contracts
        short_capital = stock_price * 100 * contracts  # Would need to borrow/collateral
        
        return {
            'position': 'Long Put Options',
            'cost': cost,
            'stock_drop': '20%',
            'option_profit': net_profit,
            'return_on_investment': return_pct * 100,
            'vs_short_stock': {
                'option_capital_needed': cost,
                'short_capital_needed': short_capital,
                'leverage_advantage': short_capital / cost
            }
        }

# Examples
print("\\n=== Speculation Use Cases ===\\n")

bull_trade = SpeculationExample.bull_futures_trade()
print(f"Bullish Trader - {bull_trade['position']}")
print(f"  Leverage: {bull_trade['leverage']:.0f}x")
print(f"  Oil +{bull_trade['price_move']} → Account +{bull_trade['return_on_margin']:.0f}%")
print(f"  Profit: \\$\{bull_trade['dollar_profit']:,}")

bear_trade = SpeculationExample.bear_put_option()
print(f"\\nBearish Trader - {bear_trade['position']}")
print(f"  Cost: \\$\{bear_trade['cost']:,}")
print(f"  Stock -{bear_trade['stock_drop']} → Return +{bear_trade['return_on_investment']:.0f}%")
print(f"  Profit: \\$\{bear_trade['option_profit']:,}")
print(f"\\n⚠️  Speculation = high risk, high reward. Can lose entire investment!")
\`\`\`

### 3. Arbitrage (Risk-Free Profits)

**Goal**: Exploit price differences across markets.

\`\`\`python
class ArbitrageExample:
    """
    Arbitrage opportunities with derivatives
    """
    
    @staticmethod
    def cash_and_carry_arbitrage():
        """
        Futures trading above fair value
        Arbitrage: Buy spot, sell futures, lock in profit
        """
        spot_price = 100
        futures_price = 105
        risk_free_rate = 0.02  # 2% annual
        time_to_maturity = 0.25  # 3 months
        
        # Fair value of futures
        fair_futures = spot_price * (1 + risk_free_rate * time_to_maturity)
        
        # Arbitrage exists if futures > fair value
        mispricing = futures_price - fair_futures
        
        if mispricing > 0:
            # Execute arbitrage
            steps = [
                f"1. Borrow \${spot_price} at {risk_free_rate*100}% for {time_to_maturity*12:.0f} months",
                f"2. Buy spot at \${spot_price}",
                f"3. Sell futures at \${futures_price}",
                f"4. At maturity: Deliver spot, collect \${futures_price}",
                f"5. Repay loan: \${spot_price * (1 + risk_free_rate * time_to_maturity):.2f}"
            ]

profit = futures_price - spot_price * (1 + risk_free_rate * time_to_maturity)

return {
    'arbitrage_type': 'Cash and Carry',
    'spot_price': spot_price,
    'futures_price': futures_price,
    'fair_value': fair_futures,
    'mispricing': mispricing,
    'steps': steps,
    'risk_free_profit': profit,
    'return': (profit / spot_price) * 100,
    'interpretation': f'Lock in \${profit:.2f} profit per unit with ZERO risk'
            }

return { 'arbitrage_opportunity': False }

@staticmethod
    def put_call_parity_arbitrage():
"""
Put - Call Parity violation
C - P = S - K×e ^ (-rT)
        
        If violated, arbitrage exists
"""
stock_price = 100
strike = 100
call_price = 8
put_price = 6
risk_free_rate = 0.05
time_to_maturity = 1.0
        
        # Left side: C - P
left_side = call_price - put_price
        
        # Right side: S - PV(K)
pv_strike = strike * np.exp(-risk_free_rate * time_to_maturity)
right_side = stock_price - pv_strike

discrepancy = abs (left_side - right_side)

if discrepancy > 0.10:  # More than $0.10 mispricing
if left_side > right_side:
                # Call relatively expensive
arbitrage = "Sell call, buy put, buy stock, borrow PV(K)"
            else:
                # Put relatively expensive
arbitrage = "Buy call, sell put, short stock, lend PV(K)"

return {
    'arbitrage_type': 'Put-Call Parity',
    'call_price': call_price,
    'put_price': put_price,
    'left_side': left_side,
    'right_side': right_side,
    'discrepancy': discrepancy,
    'arbitrage_strategy': arbitrage,
    'profit': discrepancy
}

return { 'arbitrage_opportunity': False }

# Examples
print("\\n=== Arbitrage Use Cases ===\\n")

arb1 = ArbitrageExample.cash_and_carry_arbitrage()
if arb1.get('arbitrage_opportunity') != False:
    print(f"{arb1['arbitrage_type']} Arbitrage:")
print(f"  Mispricing: \\$\{arb1['mispricing']:.2f}")
print(f"  Risk-Free Profit: \\$\{arb1['risk_free_profit']:.2f}")
print(f"  {arb1['interpretation']}")

print("\\n💡 In efficient markets, arbitrage opportunities disappear in seconds")
print("   HFT firms compete to find and exploit them")
\`\`\`

---

## Forward Contracts

The simplest derivative: Agreement to buy/sell an asset at a specified price on a future date.

\`\`\`python
@dataclass
class ForwardContract(DerivativeContract):
    """
    Forward contract: OTC, customized, bilateral agreement
    """
    forward_price: float
    long_party: str  # Party agreeing to BUY
    short_party: str  # Party agreeing to SELL
    
    def __post_init__(self):
        self.contract_type = DerivativeType.FORWARD
    
    def calculate_payoff (self, spot_price_at_maturity: float) -> dict:
        """
        Calculate payoff at maturity
        
        Long payoff = Spot - Forward
        Short payoff = Forward - Spot
        """
        long_payoff = spot_price_at_maturity - self.forward_price
        short_payoff = -long_payoff  # Zero-sum game
        
        return {
            'spot_at_maturity': spot_price_at_maturity,
            'forward_price': self.forward_price,
            'long_payoff': long_payoff,
            'short_payoff': short_payoff,
            'long_party': self.long_party,
            'short_party': self.short_party,
            'winner': self.long_party if long_payoff > 0 else self.short_party
        }
    
    def mark_to_market (self, current_spot_price: float, risk_free_rate: float) -> dict:
        """
        Calculate current market value before maturity
        
        Forward value = (Current Spot - Forward Price) × e^(-r×T)
        """
        time_remaining = self.time_to_maturity_years()
        
        if time_remaining <= 0:
            return {'value': 0, 'reason': 'Contract expired'}
        
        # Present value of expected payoff
        discount_factor = np.exp(-risk_free_rate * time_remaining)
        forward_value = (current_spot_price - self.forward_price) * discount_factor
        
        return {
            'current_spot': current_spot_price,
            'forward_price': self.forward_price,
            'time_remaining_years': time_remaining,
            'value_to_long': forward_value,
            'value_to_short': -forward_value,
            'interpretation': f'Long party {"gains" if forward_value > 0 else "loses"} \${abs (forward_value):.2f} if closed now'
        }
    
    def calculate_fair_forward_price (self,
    spot_price: float,
    risk_free_rate: float,
    dividend_yield: float = 0) -> float:
"""
        Fair forward price formula
F = S × e ^ ((r - q) × T)

where:
F = forward price
S = spot price
r = risk - free rate
q = dividend yield (or convenience yield for commodities)
    T = time to maturity
"""
time_years = self.time_to_maturity_years()
fair_price = spot_price * np.exp((risk_free_rate - dividend_yield) * time_years)
return fair_price

# Example: Currency forward
usd_eur_forward = ForwardContract(
    underlying_asset = "EUR/USD",
    notional_amount = 1_000_000,  # €1M
    contract_date = datetime.now(),
    maturity_date = datetime.now() + timedelta (days = 180),  # 6 months
    counterparty_a = "U.S. Importer",
    counterparty_b = "Bank",
    forward_price = 1.10,  # Lock in at $1.10 per euro
    long_party = "U.S. Importer",  # Agreeing to BUY euros
    short_party = "Bank"  # Agreeing to SELL euros
)

print("=== Forward Contract Example ===\\n")
print(f"Contract: {usd_eur_forward.underlying_asset}")
print(f"Forward Rate Locked: \\$\{usd_eur_forward.forward_price}")
print(f"Maturity: {usd_eur_forward.time_to_maturity_days()} days")

# Scenario 1: EUR strengthens to 1.15
payoff1 = usd_eur_forward.calculate_payoff(1.15)
print(f"\\nScenario 1: EUR at \\$\{payoff1['spot_at_maturity']}")
print(f"  \{payoff1['long_party']}: \{'Gains' if payoff1['long_payoff'] > 0 else 'Loses'} \\$\{abs(payoff1['long_payoff'] * usd_eur_forward.notional_amount):,.0f}")
print(f"  (Locked in $1.10, market is $1.15, saved $0.05 per euro)")

# Scenario 2: EUR weakens to 1.05
payoff2 = usd_eur_forward.calculate_payoff(1.05)
print(f"\\nScenario 2: EUR at \\$\{payoff2['spot_at_maturity']}")
print(f"  \{payoff2['long_party']}: \{'Gains' if payoff2['long_payoff'] > 0 else 'Loses'} \\$\{abs(payoff2['long_payoff'] * usd_eur_forward.notional_amount):,.0f}")
print(f"  (Locked in $1.10, market is $1.05, overpaid $0.05 per euro)")

# Mark to market now
mtm = usd_eur_forward.mark_to_market (current_spot_price = 1.12, risk_free_rate = 0.03)
print(f"\\nCurrent Market Value (spot at $1.12):")
print(f"  {mtm['interpretation']}")
\`\`\`

**Key Characteristics of Forwards:**
- ✅ **Customized**: Any size, any maturity, any underlying
- ✅ **No upfront cost**: Both parties enter for free
- ❌ **Counterparty risk**: Other party might default
- ❌ **Illiquid**: Hard to exit before maturity
- ❌ **No daily settlement**: Payoff only at maturity

---

## Futures Contracts

Standardized forwards that trade on exchanges.

\`\`\`python
@dataclass
class FuturesContract(DerivativeContract):
    """
    Futures: Exchange-traded, standardized, marked-to-market daily
    """
    contract_size: int
    tick_size: float  # Minimum price movement
    initial_margin: float
    maintenance_margin: float
    daily_settlement_price: Optional[float] = None
    
    def __post_init__(self):
        self.contract_type = DerivativeType.FUTURE
    
    def calculate_daily_pnl (self, 
                           yesterday_price: float, 
                           today_price: float,
                           position: Literal['long', 'short'],
                           num_contracts: int = 1) -> dict:
        """
        Calculate daily P&L with mark-to-market
        
        Futures are settled DAILY (unlike forwards at maturity only)
        """
        price_change = today_price - yesterday_price
        
        if position == 'long':
            pnl_per_contract = price_change * self.contract_size
        else:  # short
            pnl_per_contract = -price_change * self.contract_size
        
        total_pnl = pnl_per_contract * num_contracts
        
        return {
            'position': position,
            'num_contracts': num_contracts,
            'price_change': price_change,
            'pnl_per_contract': pnl_per_contract,
            'total_daily_pnl': total_pnl,
            'settlement': 'Cash transferred today (mark-to-market)'
        }
    
    def check_margin_call (self,
                         account_balance: float,
                         position_value: float,
                         position: Literal['long', 'short']) -> dict:
        """
        Check if margin call is triggered
        
        If account balance < maintenance margin, must add funds
        """
        margin_required = position_value * (self.maintenance_margin / self.initial_margin)
        
        is_margin_call = account_balance < margin_required
        amount_needed = max(0, self.initial_margin - account_balance)
        
        return {
            'account_balance': account_balance,
            'maintenance_margin_required': margin_required,
            'is_margin_call': is_margin_call,
            'amount_to_deposit': amount_needed if is_margin_call else 0,
            'warning': 'MARGIN CALL! Deposit funds or position will be liquidated' if is_margin_call else 'Account OK'
        }
    
    def calculate_contract_value (self, price: float) -> float:
        """Calculate notional value of one contract"""
        return price * self.contract_size

# Example: S&P 500 E-mini Futures
sp500_future = FuturesContract(
    underlying_asset="S&P 500 E-mini",
    notional_amount=225_000,  # $4,500 × 50 multiplier
    contract_date=datetime.now(),
    maturity_date=datetime.now() + timedelta (days=90),
    counterparty_a="Trader",
    counterparty_b="CME Exchange",
    contract_size=50,  # $50 per index point
    tick_size=0.25,  # Minimum move = $12.50
    initial_margin=12_000,  # ~5% of notional
    maintenance_margin=11_000
)

print("\\n=== Futures Contract Example ===\\n")
print(f"Contract: {sp500_future.underlying_asset}")
print(f"Multiplier: \\$\{sp500_future.contract_size} per point")
print(f"Initial Margin: \\$\{sp500_future.initial_margin:,}")

# Day 1: Enter long at 4500
entry_price = 4500
position_value = entry_price * sp500_future.contract_size
print(f"\\nDay 1: Enter long at {entry_price}")
print(f"Position Value: \\$\{position_value:,}")
print(f"Margin Posted: \\$\{sp500_future.initial_margin:,}")
print(f"Leverage: {position_value / sp500_future.initial_margin:.1f}x")

# Day 2: Price moves to 4520(+20 points)
day2_price = 4520
pnl_day2 = sp500_future.calculate_daily_pnl(
    yesterday_price = entry_price,
    today_price = day2_price,
    position = 'long',
    num_contracts = 1
)

print(f"\\nDay 2: Price moves to {day2_price} (+{day2_price - entry_price} points)")
print(f"Daily P&L: \\$\{pnl_day2['total_daily_pnl']:,}")
print(f"Account Balance: \\$\{sp500_future.initial_margin + pnl_day2['total_daily_pnl']:,}")

# Day 3: Market crashes to 4400(-120 points)
day3_price = 4400
pnl_day3 = sp500_future.calculate_daily_pnl(
    yesterday_price = day2_price,
    today_price = day3_price,
    position = 'long',
    num_contracts = 1
)

new_balance = sp500_future.initial_margin + pnl_day2['total_daily_pnl'] + pnl_day3['total_daily_pnl']

print(f"\\nDay 3: Market crashes to {day3_price} (-{day2_price - day3_price} points)")
print(f"Daily P&L: \\$\{pnl_day3['total_daily_pnl']:,}")
print(f"Account Balance: \\$\{new_balance:,}")

# Check for margin call
margin_check = sp500_future.check_margin_call(
    account_balance = new_balance,
    position_value = day3_price * sp500_future.contract_size,
    position = 'long'
)

print(f"\\n{margin_check['warning']}")
if margin_check['is_margin_call']:
    print(f"Must deposit: \\$\{margin_check['amount_to_deposit']:,}")
\`\`\`

**Forwards vs Futures Comparison:**

| Feature | Forward | Future |
|---------|---------|--------|
| Trading | OTC (bilateral) | Exchange |
| Standardization | Customized | Standardized |
| Settlement | At maturity | Daily (mark-to-market) |
| Counterparty Risk | HIGH | LOW (clearinghouse) |
| Liquidity | LOW | HIGH |
| Margin | None or minimal | Required daily |
| Regulation | Minimal | Heavily regulated |

---

## Options Primer

Rights (not obligations) to buy/sell at a specified price.

\`\`\`python
from scipy.stats import norm

class OptionContract:
    """
    Option contracts: Call (right to buy) or Put (right to sell)
    """
    
    def __init__(self,
                 option_type: Literal['call', 'put'],
                 strike_price: float,
                 expiration: datetime,
                 underlying_price: float,
                 premium: float):
        self.option_type = option_type
        self.strike = strike_price
        self.expiration = expiration
        self.underlying_price = underlying_price
        self.premium = premium
    
    def calculate_payoff (self, spot_price_at_expiration: float) -> dict:
        """
        Intrinsic value at expiration
        
        Call: max(S - K, 0)
        Put: max(K - S, 0)
        """
        if self.option_type == 'call':
            intrinsic_value = max (spot_price_at_expiration - self.strike, 0)
        else:  # put
            intrinsic_value = max (self.strike - spot_price_at_expiration, 0)
        
        # Profit = Payoff - Premium Paid
        profit = intrinsic_value - self.premium
        
        return {
            'spot_at_expiration': spot_price_at_expiration,
            'strike': self.strike,
            'intrinsic_value': intrinsic_value,
            'premium_paid': self.premium,
            'net_profit': profit,
            'return': (profit / self.premium) * 100 if self.premium > 0 else 0
        }
    
    def moneyness (self) -> str:
        """
        Determine if option is ITM, ATM, or OTM
        """
        if self.option_type == 'call':
            if self.underlying_price > self.strike:
                return "ITM (In The Money)"
            elif self.underlying_price == self.strike:
                return "ATM (At The Money)"
            else:
                return "OTM (Out of The Money)"
        else:  # put
            if self.underlying_price < self.strike:
                return "ITM"
            elif self.underlying_price == self.strike:
                return "ATM"
            else:
                return "OTM"

# Example: Call option
call = OptionContract(
    option_type='call',
    strike_price=100,
    expiration=datetime.now() + timedelta (days=30),
    underlying_price=100,
    premium=5
)

print("\\n=== Option Payoff Examples ===\\n")
print(f"Call Option: Strike \${call.strike}, Premium \\$\{call.premium}")
print(f"Moneyness: {call.moneyness()}")

# Calculate payoffs at different prices
for price in [80, 90, 100, 110, 120]:
    result = call.calculate_payoff (price)
    print(f"\\nStock at \\$\{price}:")
    print(f"  Intrinsic Value: \\$\{result['intrinsic_value']:.2f}")
print(f"  Net Profit: \\$\{result['net_profit']:.2f}")
print(f"  Return: {result['return']:.0f}%")

print("\\n💡 Key Insight:")
print("   • Options have LIMITED downside (premium) but UNLIMITED upside")
print("   • This is why options are popular for leverage and speculation")
\`\`\`

We'll cover options in extreme depth in Module 8.

---

## Swaps

Exchange of cash flows between two parties.

\`\`\`python
class InterestRateSwap:
    """
    Interest Rate Swap: Exchange fixed for floating rates
    Most common type of derivative by notional value
    """
    
    def __init__(self,
                 notional: float,
                 fixed_rate: float,
                 floating_rate_index: str,
                 tenor_years: int,
                 payment_frequency: int = 2):  # Semi-annual
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.floating_rate_index = floating_rate_index
        self.tenor_years = tenor_years
        self.payment_frequency = payment_frequency
        self.num_payments = tenor_years * payment_frequency
    
    def calculate_payments (self, floating_rates: list[float]) -> pd.DataFrame:
        """
        Calculate swap cash flows
        
        Fixed payer: Pays fixed, receives floating
        Floating payer: Pays floating, receives fixed
        """
        if len (floating_rates) != self.num_payments:
            raise ValueError (f"Need {self.num_payments} floating rates")
        
        payments = []
        
        for period in range (self.num_payments):
            # Fixed leg
            fixed_payment = (self.fixed_rate / self.payment_frequency) * self.notional
            
            # Floating leg
            floating_payment = (floating_rates[period] / self.payment_frequency) * self.notional
            
            # Net payment (from fixed payer's perspective)
            net_payment = fixed_payment - floating_payment
            
            payments.append({
                'period': period + 1,
                'fixed_rate': self.fixed_rate,
                'floating_rate': floating_rates[period],
                'fixed_payment': fixed_payment,
                'floating_payment': floating_payment,
                'net_payment_fixed_payer': net_payment,
                'cumulative_pnl': sum (p['net_payment_fixed_payer'] 
                                     for p in payments)
            })
        
        return pd.DataFrame (payments)

# Example: $10M interest rate swap
swap = InterestRateSwap(
    notional=10_000_000,
    fixed_rate=0.04,  # 4% fixed
    floating_rate_index="SOFR",
    tenor_years=2,
    payment_frequency=2
)

# Simulate floating rates
np.random.seed(42)
floating_rates = [0.03 + np.random.normal(0, 0.005) 
                 for _ in range (swap.num_payments)]

payments_df = swap.calculate_payments (floating_rates)

print("\\n=== Interest Rate Swap Example ===\\n")
print(f"Notional: \\$\{swap.notional:,}")
print(f"Fixed Rate: {swap.fixed_rate*100}%")
print(f"Tenor: {swap.tenor_years} years")
print(f"\\nPayment Schedule:")
print(payments_df.to_string (index = False))

final_pnl = payments_df['cumulative_pnl'].iloc[-1]
print(f"\\nFinal P&L (Fixed Payer): \\$\{final_pnl:,.0f}")
if final_pnl < 0:
    print("Fixed payer paid more than received (rates stayed below 4%)")
else:
print("Fixed payer received more than paid (rates rose above 4%)")
\`\`\`

**Why Swaps Exist:**
- **Hedge interest rate risk**: Convert floating-rate debt to fixed
- **Speculation**: Bet on direction of interest rates
- **Balance sheet management**: Banks use to manage asset-liability duration

---

## The Dark Side: 2008 Financial Crisis

Derivatives played a central role in the 2008 crisis:

\`\`\`python
def explain_2008_crisis():
    """
    How derivatives amplified the crisis
    """
    
    crisis_timeline = {
        'root_cause': 'Subprime mortgages (risky loans to borrowers with bad credit)',
        
        'securitization': {
            'process': 'Banks bundled mortgages into MBS (Mortgage-Backed Securities)',
            'volume': '$1+ trillion in MBS created'
        },
        
        'derivatives_layer_1': {
            'product': 'CDOs (Collateralized Debt Obligations)',
            'description': 'Sliced MBS into tranches (AAA, AA, BBB, etc.)',
            'problem': 'Rating agencies gave AAA ratings to junk',
            'volume': '$640 billion in CDOs'
        },
        
        'derivatives_layer_2': {
            'product': 'CDO-squared (CDOs of CDOs)',
            'description': 'CDOs made from other CDOs tranches',
            'problem': 'No one understood the risk',
            'leverage': '30-40x leverage common'
        },
        
        'derivatives_layer_3': {
            'product': 'Credit Default Swaps (CDS)',
            'description': 'Insurance against defaults',
            'problem': 'AIG sold $500B+ in CDS without capital to pay',
            'systemic_risk': 'Counterparty chains - if AIG fails, everyone fails'
        },
        
        'collapse': {
            'trigger': 'Housing prices fell 20%',
            'cascade': [
                '1. Subprime borrowers defaulted',
                '2. MBS values collapsed',
                '3. CDOs became worthless',
                '4. Banks holding CDOs faced huge losses',
                '5. CDS payouts triggered',
                '6. AIG couldn\'t pay → required $182B bailout',
                '7. Lehman Brothers bankruptcy',
                '8. Credit markets froze',
                '9. Global financial system nearly collapsed'
            ],
            'amplification': 'Derivatives turned $1T mortgage problem into $10T+ crisis'
        }
    }
    
    print("\\n=== 2008 Financial Crisis: Role of Derivatives ===\\n")
    print(f"Root Cause: {crisis_timeline['root_cause']}")
    print(f"\\nDerivatives Amplified the Problem:")
    print(f"  - {crisis_timeline['derivatives_layer_1']['description']}")
    print(f"  - {crisis_timeline['derivatives_layer_2']['description']}")
    print(f"  - {crisis_timeline['derivatives_layer_3']['description']}")
    print(f"\\nCollapse:")
    for step in crisis_timeline['collapse']['cascade']:
        print(f"  {step}")
    print(f"\\n⚠️  {crisis_timeline['collapse']['amplification']}")
    print("\\n📚 Lessons:")
    print("  1. Complexity hides risk")
    print("  2. Leverage amplifies losses")
    print("  3. Counterparty chains create systemic risk")
    print("  4. Rating agencies can be wrong")
    print("  5. 'Too big to fail' is a real problem")

explain_2008_crisis()
\`\`\`

**Warren Buffett\'s Warning (2003):**
> "Derivatives are financial weapons of mass destruction, carrying dangers that, while now latent, are potentially lethal."

He was right.

---

## Summary

**Key Takeaways:**1. **Derivatives = Contracts** whose value derives from underlying assets
2. **Three uses**: Hedging (reduce risk), Speculation (take risk), Arbitrage (exploit mispricings)
3. **Four types**: Forwards, Futures, Options, Swaps
4. **Leverage is inherent** - small moves = big P&L swings
5. **Forwards vs Futures**: OTC vs exchange, customized vs standardized
6. **Options = Asymmetric**: Limited downside, unlimited upside
7. **Swaps = Exchange cash flows**: Manage interest rate/currency risk
8. **Danger**: Complexity, leverage, counterparty risk can cause systemic crises

**For Engineers:**
- Derivatives are mathematically sophisticated (pricing models)
- High-performance systems needed (low latency)
- Risk management is critical
- Regulatory compliance is complex
- Most lucrative area of fintech/finance

**Next Steps:**
- Module 7: Deep dive into futures and swaps
- Module 8: Master options pricing and Greeks
- Module 12: High-frequency derivatives trading
- Module 14: Build derivatives pricing engines

You now understand derivatives - the most powerful and dangerous instruments in finance!
`,
  exercises: [
    {
      prompt:
        'Build a futures margin calculator that simulates daily mark-to-market, detects margin calls, and calculates optimal position sizing given risk tolerance. Include scenarios for extreme market moves (circuit breakers, limit moves).',
      solution:
        '// Implementation: 1) Track daily P&L with mark-to-market, 2) Monitor account balance vs maintenance margin, 3) Simulate gap risk (overnight moves), 4) Calculate VAR and position limits, 5) Alert on margin calls, 6) Handle forced liquidation scenarios, 7) Optimize position size using Kelly criterion or risk-adjusted returns',
    },
    {
      prompt:
        'Create an arbitrage detector that monitors cash-and-carry opportunities between spot and futures markets. Calculate fair value, detect mispricings, and simulate arbitrage execution accounting for transaction costs and funding costs.',
      solution:
        '// Implementation: 1) Real-time spot and futures price feeds, 2) Calculate fair futures value using cost-of-carry model, 3) Detect mispricings > threshold, 4) Account for bid-ask spreads, borrowing costs, storage costs (commodities), 5) Simulate P&L including all costs, 6) Alert when net profit > minimum threshold, 7) Track arbitrage convergence over time',
    },
  ],
};
