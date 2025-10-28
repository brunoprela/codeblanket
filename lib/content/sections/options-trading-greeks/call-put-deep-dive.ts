export const callPutDeepDive = {
  title: 'Call and Put Options Deep Dive',
  id: 'call-put-deep-dive',
  content: `
# Call and Put Options Deep Dive

## Introduction

In the previous section, we covered options fundamentals. Now we'll dive deeper into the four basic option positions - long call, short call, long put, short put - and explore advanced concepts like **put-call parity**, **synthetic positions**, and **risk management** for each strategy.

**Why This Matters:**
- Understanding all four positions is essential for building complex strategies
- Short options have different risk profiles (potentially unlimited loss)
- Put-call parity enables arbitrage detection and synthetic position construction
- Professional traders use synthetics for tax, margin, and execution advantages

By mastering these concepts, you'll be able to:
- Construct any option strategy from basic building blocks
- Identify arbitrage opportunities using put-call parity
- Create synthetic positions for better execution or margin efficiency
- Manage risk appropriately for long and short option positions

---

## Long Call Position

### Mechanics

**Long call** = Buying a call option (paying premium for the right to buy stock at strike).

**When to Use:**
- Bullish on stock (expect price to rise)
- Want leverage (control more shares with less capital)
- Defined risk (max loss = premium paid)
- Event-driven (earnings, product launch, regulatory approval)

**Maximum Profit:** Unlimited (stock can go to infinity)
**Maximum Loss:** Premium paid
**Breakeven:** Strike + Premium

### Example

\`\`\`python
"""
Long Call Analysis
"""

# Trade setup
stock_price = 100
strike = 105
premium = 3.00
expiration_days = 45

# Position characteristics
max_loss = premium * 100  # $300
max_profit = float('inf')  # Unlimited
breakeven = strike + premium  # $108

print(f"Long Call: \${strike} Strike, \\$\{premium} Premium")
print(f"Max Loss: \\$\{max_loss}")
print(f"Max Profit: Unlimited")
print(f"Breakeven: \\$\{breakeven}")
print(f"Required Move: {(breakeven/stock_price - 1)*100:.1f}%")
\`\`\`

**Scenarios at Expiration:**

| Stock Price | Intrinsic | P&L | Return |
|-------------|-----------|-----|--------|
| $95  | $0  | -$300 | -100% |
| $100 | $0  | -$300 | -100% |
| $105 | $0  | -$300 | -100% (strike) |
| $108 | $3  | $0    | 0% (breakeven) |
| $110 | $5  | +$200 | +67% |
| $115 | $10 | +$700 | +233% |
| $120 | $15 | +$1200 | +400% |

**Key Insight:** Need stock to move to $108 (+8%) just to breakeven. But if stock rallies to $120 (+20%), you make $1200 (+400%) - that's the power of leverage.

### Risk Management

**Position Sizing:**
\`\`\`python
def calculate_position_size(account_size: float,
                           risk_per_trade: float,
                           premium: float) -> int:
    """
    Calculate number of contracts based on risk tolerance
    
    Args:
        account_size: Total account value
        risk_per_trade: % of account willing to risk (e.g., 0.02 for 2%)
        premium: Option premium per contract
        
    Returns:
        Number of contracts
    """
    max_risk_dollars = account_size * risk_per_trade
    contracts = int(max_risk_dollars / (premium * 100))
    return max(contracts, 0)

# Example
account = 50000
risk_pct = 0.02  # Risk 2% per trade
premium = 3.00

contracts = calculate_position_size(account, risk_pct, premium)
print(f"Account: \\$\{account:,}")
print(f"Risk per trade: {risk_pct*100}% (\\$\{account*risk_pct:,})")
print(f"Max contracts: {contracts}")
print(f"Total risk: \\$\{contracts * premium * 100:,}")
\`\`\`

**Stop Loss Strategy:**
- Time-based: Exit if 50% time to expiration passed and stock hasn't moved
- Price-based: Exit if option loses 50% of value
- Delta-based: Exit if delta falls below 0.30 (low probability)

---

## Short Call Position

### Mechanics

**Short call** = Selling (writing) a call option (collecting premium, obligated to sell stock at strike if assigned).

**When to Use:**
- Neutral to bearish on stock
- Generate income from premium collection
- Expect stock to stay below strike
- Covered call strategy (own stock + sell call)

**Maximum Profit:** Premium collected
**Maximum Loss:** Unlimited (if stock rallies significantly)
**Breakeven:** Strike + Premium

### Example

\`\`\`python
"""
Short Call Analysis
"""

# Trade setup (same as long call, but opposite side)
stock_price = 100
strike = 105
premium = 3.00
expiration_days = 45

# Position characteristics
max_profit = premium * 100  # $300
max_loss = float('inf')  # Unlimited
breakeven = strike + premium  # $108

print(f"Short Call: \${strike} Strike, \\$\{premium} Premium")
print(f"Max Profit: \\$\{max_profit}")
print(f"Max Loss: Unlimited ⚠️")
print(f"Breakeven: \\$\{breakeven}")
\`\`\`

**Scenarios at Expiration:**

| Stock Price | Intrinsic | P&L | Note |
|-------------|-----------|-----|------|
| $95  | $0  | +$300 | Max profit |
| $100 | $0  | +$300 | Max profit |
| $105 | $0  | +$300 | Max profit (at strike) |
| $108 | $3  | $0    | Breakeven |
| $110 | $5  | -$200 | Starting losses |
| $115 | $10 | -$700 | Significant loss |
| $120 | $15 | -$1200 | Large loss |
| $150 | $45 | -$4200 | Devastating loss |

**Warning:** Unlimited risk! If stock goes to $200, you lose $9,700 per contract. **NEVER sell naked calls** unless you fully understand the risk.

### Risk Management

**Covered Call (Defined Risk):**
\`\`\`python
def covered_call_analysis(stock_price: float,
                         strike: float,
                         premium: float,
                         shares: int = 100) -> dict:
    """
    Analyze covered call position (long stock + short call)
    
    This DEFINES the risk - max loss is if stock goes to $0
    """
    # Cost basis
    stock_cost = stock_price * shares
    premium_collected = premium * shares
    
    # Adjusted cost basis after premium
    net_cost = stock_cost - premium_collected
    
    # Max profit (if stock >= strike at expiration)
    profit_from_stock = (strike - stock_price) * shares
    max_profit = profit_from_stock + premium_collected
    
    # Max loss (if stock goes to $0)
    max_loss = net_cost
    
    return {
        'stock_cost': stock_cost,
        'premium_collected': premium_collected,
        'net_cost': net_cost,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'breakeven': net_cost / shares,
        'return_if_called': (max_profit / stock_cost) * 100
    }

# Example
result = covered_call_analysis(
    stock_price=100,
    strike=105,
    premium=3.00
)

print("Covered Call Analysis:")
print(f"Stock cost: \\$\{result['stock_cost']:,}")
print(f"Premium collected: \\$\{result['premium_collected']}")
print(f"Net cost: \\$\{result['net_cost']}")
print(f"Max profit: \\$\{result['max_profit']} ({result['return_if_called']:.1f}% return)")
print(f"Max loss: \\$\{result['max_loss']} (if stock → $0)")
print(f"Breakeven: \\$\{result['breakeven']}")
\`\`\`

**Naked Call Margin Requirements:**
- Initial margin: ~20% of stock value + option premium
- Maintenance: Marked-to-market daily
- Margin call if account falls below maintenance requirement

---

## Long Put Position

### Mechanics

**Long put** = Buying a put option (paying premium for the right to sell stock at strike).

**When to Use:**
- Bearish on stock (expect price to fall)
- Portfolio protection (insurance against decline)
- Event-driven hedging
- Defined risk (max loss = premium)

**Maximum Profit:** Strike - Premium (stock can't go below $0)
**Maximum Loss:** Premium paid
**Breakeven:** Strike - Premium

### Example

\`\`\`python
"""
Long Put Analysis
"""

# Trade setup
stock_price = 100
strike = 95
premium = 2.50
expiration_days = 45

# Position characteristics
max_loss = premium * 100  # $250
max_profit = (strike - premium) * 100  # $9,250 (if stock → $0)
breakeven = strike - premium  # $92.50

print(f"Long Put: \${strike} Strike, \\$\{premium} Premium")
print(f"Max Loss: \\$\{max_loss}")
print(f"Max Profit: \\$\{max_profit} (if stock → $0)")
print(f"Breakeven: \\$\{breakeven}")
print(f"Required Move: {(stock_price/breakeven - 1)*100:.1f}% down")
\`\`\`

**Scenarios at Expiration:**

| Stock Price | Intrinsic | P&L | Return |
|-------------|-----------|-----|--------|
| $110 | $0  | -$250 | -100% |
| $100 | $0  | -$250 | -100% |
| $95  | $0  | -$250 | -100% (strike) |
| $92.50 | $2.50 | $0 | 0% (breakeven) |
| $90  | $5  | +$250 | +100% |
| $85  | $10 | +$750 | +300% |
| $80  | $15 | +$1250 | +500% |

### Protective Put (Portfolio Insurance)

\`\`\`python
def protective_put_analysis(stock_price: float,
                           stock_shares: int,
                           put_strike: float,
                           put_premium: float) -> dict:
    """
    Analyze protective put strategy (long stock + long put)
    
    This is portfolio insurance - limits downside
    """
    # Costs
    stock_value = stock_price * stock_shares
    insurance_cost = put_premium * stock_shares
    total_cost = stock_value + insurance_cost
    
    # Floor (minimum value)
    floor_value = put_strike * stock_shares
    
    # Max loss (if stock falls below put strike)
    max_loss = total_cost - floor_value
    
    # Breakeven (need stock to rise to cover insurance)
    breakeven = stock_price + put_premium
    
    return {
        'stock_value': stock_value,
        'insurance_cost': insurance_cost,
        'total_cost': total_cost,
        'floor_value': floor_value,
        'max_loss': max_loss,
        'max_loss_pct': (max_loss / stock_value) * 100,
        'breakeven': breakeven,
        'protection_level': (stock_price - put_strike) / stock_price * 100
    }

# Example: Own 100 shares at $100, buy 95 put for protection
result = protective_put_analysis(
    stock_price=100,
    stock_shares=100,
    put_strike=95,
    put_premium=2.50
)

print("Protective Put Analysis:")
print(f"Stock value: \\$\{result['stock_value']:,}")
print(f"Insurance cost: \\$\{result['insurance_cost']} ({(result['insurance_cost']/result['stock_value'])*100:.1f}%)")
print(f"Floor value: \\$\{result['floor_value']:,} (minimum portfolio value)")
print(f"Max loss: \\$\{result['max_loss']} ({result['max_loss_pct']:.1f}%)")
print(f"Breakeven: \\$\{result['breakeven']}")
print(f"Protection: {result['protection_level']:.0f}% below current price")
\`\`\`

**Real-World Use Case:** You own 1000 shares of NVDA at $450 ($450K position). Worried about earnings volatility. Buy 10 × 420 puts (7% below current) for $8 each ($8K total insurance). If NVDA crashes to $350, your puts are worth $70 each ($70K), limiting loss to $20K + $8K insurance = $28K total loss (6.2% of position) instead of $100K (22% loss) without protection.

---

## Short Put Position

### Mechanics

**Short put** = Selling (writing) a put option (collecting premium, obligated to buy stock at strike if assigned).

**When to Use:**
- Bullish to neutral on stock
- Generate income from premium
- Willing to own stock at strike price (cash-secured put)
- Selling volatility premium

**Maximum Profit:** Premium collected
**Maximum Loss:** Strike - Premium (if stock → $0)
**Breakeven:** Strike - Premium

### Example

\`\`\`python
"""
Short Put Analysis
"""

# Trade setup
stock_price = 100
strike = 95
premium = 2.50
expiration_days = 45

# Position characteristics
max_profit = premium * 100  # $250
max_loss = (strike - premium) * 100  # $9,250 (if stock → $0)
breakeven = strike - premium  # $92.50

print(f"Short Put: \${strike} Strike, \\$\{premium} Premium")
print(f"Max Profit: \\$\{max_profit}")
print(f"Max Loss: \\$\{max_loss} (if stock → $0)")
print(f"Breakeven: \\$\{breakeven}")
print(f"Required cash collateral: \\$\{strike * 100:,}")
\`\`\`

**Scenarios at Expiration:**

| Stock Price | Intrinsic | P&L | Note |
|-------------|-----------|-----|------|
| $110 | $0 | +$250 | Max profit |
| $100 | $0 | +$250 | Max profit |
| $95  | $0 | +$250 | Max profit (at strike) |
| $92.50 | $2.50 | $0 | Breakeven |
| $90  | $5 | -$250 | Starting losses |
| $85  | $10 | -$750 | Significant loss |
| $80  | $15 | -$1250 | Large loss |
| $50  | $45 | -$4250 | Devastating loss |

### Cash-Secured Put Strategy

\`\`\`python
def cash_secured_put_analysis(stock_price: float,
                              strike: float,
                              premium: float,
                              target_entry_price: float = None) -> dict:
    """
    Analyze cash-secured put (selling put with cash set aside to buy stock)
    
    This is how traders "get paid to set a limit order"
    """
    # Required cash
    cash_required = strike * 100
    premium_collected = premium * 100
    
    # Effective purchase price if assigned
    net_purchase_price = strike - premium
    
    # Compare to target entry price
    if target_entry_price is None:
        target_entry_price = stock_price * 0.95  # Default: 5% below current
    
    # Returns
    return_if_expires = (premium_collected / cash_required) * 100
    
    # Max loss (if stock → $0)
    max_loss = net_purchase_price * 100
    
    return {
        'cash_required': cash_required,
        'premium_collected': premium_collected,
        'net_purchase_price': net_purchase_price,
        'discount_to_current': ((stock_price - net_purchase_price) / stock_price) * 100,
        'return_if_expires': return_if_expires,
        'max_loss': max_loss,
        'better_than_limit_order': net_purchase_price <= target_entry_price
    }

# Example: Sell 95 put on $100 stock, collect $2.50
result = cash_secured_put_analysis(
    stock_price=100,
    strike=95,
    premium=2.50,
    target_entry_price=95
)

print("Cash-Secured Put Analysis:")
print(f"Cash required: \\$\{result['cash_required']:,}")
print(f"Premium collected: \\$\{result['premium_collected']}")
print(f"Net purchase price if assigned: \\$\{result['net_purchase_price']}")
print(f"Discount to current: {result['discount_to_current']:.1f}%")
print(f"Return if expires worthless: {result['return_if_expires']:.2f}%")
print(f"Better than limit order at $95? {result['better_than_limit_order']}")
\`\`\`

**Real-World Use Case:** You want to buy AAPL at $140 (currently $150). Instead of placing limit order, sell 140 put for $3. Two outcomes: (1) AAPL stays above $140 → put expires, keep $300 premium, try again next month. (2) AAPL falls below $140 → assigned, buy at $140 but net cost $137 (got paid $3 to enter). Win-win!

---

## Put-Call Parity

### The Relationship

For European options with same strike and expiration:

\`\`\`
C - P = S - K × e^(-rT)

where:
  C = Call price
  P = Put price
  S = Stock price
  K = Strike price
  r = Risk-free rate
  T = Time to expiration (years)
\`\`\`

For American options (approximately):
\`\`\`
C - P ≈ S - K (ignoring time value of money for short durations)
\`\`\`

**Intuition:** A long call and short put (at same strike) creates a synthetic long stock position. The cost should equal buying stock minus present value of strike.

### Arbitrage Detection

\`\`\`python
"""
Put-Call Parity Arbitrage Detector
"""

import numpy as np
from typing import Optional, Dict

def check_put_call_parity(call_price: float,
                          put_price: float,
                          stock_price: float,
                          strike: float,
                          risk_free_rate: float,
                          time_to_expiration: float,
                          transaction_cost: float = 0.10) -> Dict:
    """
    Check if put-call parity holds; detect arbitrage opportunities
    
    Args:
        call_price: Market price of call
        put_price: Market price of put
        stock_price: Current stock price
        strike: Option strike price
        risk_free_rate: Annual risk-free rate (e.g., 0.05 for 5%)
        time_to_expiration: Years to expiration
        transaction_cost: Round-trip transaction costs per share
        
    Returns:
        Dictionary with arbitrage analysis
    """
    # Calculate theoretical relationship
    pv_strike = strike * np.exp(-risk_free_rate * time_to_expiration)
    theoretical_diff = stock_price - pv_strike
    
    # Market relationship
    market_diff = call_price - put_price
    
    # Arbitrage opportunity
    deviation = market_diff - theoretical_diff
    
    # Check if arbitrage exists after transaction costs
    arbitrage_profit = abs(deviation) - transaction_cost
    
    result = {
        'theoretical_diff': theoretical_diff,
        'market_diff': market_diff,
        'deviation': deviation,
        'arbitrage_profit': arbitrage_profit,
        'arbitrage_exists': arbitrage_profit > 0,
    }
    
    # Determine arbitrage strategy
    if arbitrage_profit > 0:
        if deviation > 0:  # Call overpriced relative to put
            result['strategy'] = "Sell call, buy put, buy stock, borrow K×e^(-rT)"
            result['direction'] = "Call too expensive"
        else:  # Put overpriced relative to call
            result['strategy'] = "Buy call, sell put, short stock, lend K×e^(-rT)"
            result['direction'] = "Put too expensive"
    else:
        result['strategy'] = "No arbitrage"
        result['direction'] = "Market in equilibrium"
    
    return result

# Example 1: Market in equilibrium
print("=" * 70)
print("Example 1: Equilibrium (No Arbitrage)")
print("=" * 70)

result1 = check_put_call_parity(
    call_price=6.50,
    put_price=5.30,
    stock_price=100,
    strike=100,
    risk_free_rate=0.05,
    time_to_expiration=0.25,  # 3 months
    transaction_cost=0.10
)

print(f"Call price: \\$\{result1['market_diff'] + result1['theoretical_diff'] - result1['market_diff']:.2f}")
print(f"Theoretical C - P: \\$\{result1['theoretical_diff']:.2f}")
print(f"Market C - P: \\$\{result1['market_diff']:.2f}")
print(f"Deviation: \\$\{result1['deviation']:.2f}")
print(f"Arbitrage profit (after costs): \\$\{result1['arbitrage_profit']:.2f}")
print(f"Arbitrage exists: {result1['arbitrage_exists']}")
print(f"Strategy: {result1['strategy']}")

# Example 2: Arbitrage opportunity
print("\\n" + "=" * 70)
print("Example 2: Arbitrage Opportunity")
print("=" * 70)

result2 = check_put_call_parity(
    call_price=7.50,  # Call overpriced
    put_price=5.30,
    stock_price=100,
    strike=100,
    risk_free_rate=0.05,
    time_to_expiration=0.25,
    transaction_cost=0.10
)

print(f"Theoretical C - P: \\$\{result2['theoretical_diff']:.2f}")
print(f"Market C - P: \\$\{result2['market_diff']:.2f}")
print(f"Deviation: \\$\{result2['deviation']:.2f}")
print(f"Arbitrage profit (after costs): \\$\{result2['arbitrage_profit']:.2f}")
print(f"Arbitrage exists: {result2['arbitrage_exists']} ✓")
print(f"Strategy: {result2['strategy']}")
print(f"Direction: {result2['direction']}")
print(f"\\nProfit per share: \\$\{result2['arbitrage_profit']:.2f}")
print(f"Profit per contract: \\$\{result2['arbitrage_profit'] * 100:.2f}")
\`\`\`

---

## Synthetic Positions

Using put-call parity, we can create **synthetic positions** that replicate stocks or options using combinations of other instruments.

### Synthetic Long Stock

\`\`\`
Synthetic Long Stock = Long Call + Short Put (same strike)
\`\`\`

**Why Use Synthetic:**
- Margin efficiency (options margin < stock margin)
- Tax treatment differences
- No need to borrow shares (for previously short positions)

\`\`\`python
"""
Synthetic Positions
"""

def synthetic_long_stock(call_price: float,
                        put_price: float,
                        strike: float,
                        stock_price: float) -> Dict:
    """
    Analyze synthetic long stock position
    
    Synthetic = Long Call + Short Put
    """
    # Cost to enter
    net_debit = call_price - put_price
    
    # Effective stock purchase price
    effective_purchase = strike + net_debit
    
    # Compare to buying stock directly
    savings = stock_price - effective_purchase
    
    return {
        'call_price': call_price,
        'put_price': put_price,
        'net_cost': net_debit,
        'effective_purchase': effective_purchase,
        'stock_price': stock_price,
        'savings_vs_stock': savings,
        'equivalent': abs(savings) < 0.10  # Within $0.10
    }

result = synthetic_long_stock(
    call_price=6.50,
    put_price=5.30,
    strike=100,
    stock_price=100
)

print("Synthetic Long Stock:")
print(f"Buy {result['call_price']} call + Sell \\$\{result['put_price']} put")
print(f"Net cost: \\$\{result['net_cost']}")
print(f"Effective purchase price: \\$\{result['effective_purchase']}")
print(f"Actual stock price: \\$\{result['stock_price']}")
print(f"Savings vs buying stock: \\$\{result['savings_vs_stock']}")
print(f"Equivalent to stock? {result['equivalent']}")
\`\`\`

### Other Synthetic Positions

\`\`\`python
# Synthetic Short Stock
# = Short Call + Long Put

# Synthetic Long Call
# = Long Stock + Long Put
# (also known as protective put)

# Synthetic Short Call  
# = Short Stock + Short Put

# Synthetic Long Put
# = Short Stock + Long Call

# Synthetic Short Put
# = Long Stock + Short Call
# (also known as covered call)
\`\`\`

---

## Risk Management for Each Position

\`\`\`python
"""
Risk Management Framework
"""

from enum import Enum
from dataclasses import dataclass

class RiskLevel(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    EXTREME = "Extreme"

@dataclass
class PositionRisk:
    """Risk profile for option position"""
    position_type: str
    max_loss: float
    max_loss_defined: bool
    margin_required: float
    risk_level: RiskLevel
    suitable_for_beginners: bool
    warnings: list

def analyze_position_risk(position_type: str,
                         premium: float,
                         strike: float,
                         stock_price: float) -> PositionRisk:
    """
    Analyze risk for each basic position type
    """
    warnings = []
    
    if position_type == "long_call":
        return PositionRisk(
            position_type="Long Call",
            max_loss=premium * 100,
            max_loss_defined=True,
            margin_required=0,  # No margin, just premium
            risk_level=RiskLevel.LOW,
            suitable_for_beginners=True,
            warnings=["Premium can go to $0", "Time decay works against you"]
        )
    
    elif position_type == "short_call":
        warnings = [
            "⚠️ UNLIMITED LOSS POTENTIAL",
            "Requires margin",
            "Can be assigned early",
            "NOT suitable for beginners"
        ]
        return PositionRisk(
            position_type="Short Call (Naked)",
            max_loss=float('inf'),
            max_loss_defined=False,
            margin_required=stock_price * 20,  # ~20% of stock value
            risk_level=RiskLevel.EXTREME,
            suitable_for_beginners=False,
            warnings=warnings
        )
    
    elif position_type == "long_put":
        return PositionRisk(
            position_type="Long Put",
            max_loss=premium * 100,
            max_loss_defined=True,
            margin_required=0,
            risk_level=RiskLevel.LOW,
            suitable_for_beginners=True,
            warnings=["Premium can go to $0", "Time decay works against you"]
        )
    
    elif position_type == "short_put":
        warnings = [
            f"Max loss: \${(strike - premium) * 100:,.0f} if stock → $0",
            "Requires cash collateral or margin",
            "Can be assigned early",
            "Suitable for investors willing to own stock at strike"
        ]
        return PositionRisk(
            position_type="Short Put (Cash-Secured)",
            max_loss=(strike - premium) * 100,
            max_loss_defined=True,
            margin_required=strike * 100,  # Cash collateral
            risk_level=RiskLevel.MEDIUM,
            suitable_for_beginners=True,  # If cash-secured
            warnings=warnings
        )

# Example usage
for pos_type in ["long_call", "short_call", "long_put", "short_put"]:
    risk = analyze_position_risk(pos_type, 3.0, 100, 100)
    print(f"\\n{'='*60}")
    print(f"{risk.position_type}")
    print(f"{'='*60}")
    print(f"Max Loss: \\$\{risk.max_loss if risk.max_loss != float('inf') else 'UNLIMITED'}")
    print(f"Defined Risk: {risk.max_loss_defined}")
    print(f"Margin Required: \\$\{risk.margin_required:,.0f}")
    print(f"Risk Level: {risk.risk_level.value}")
    print(f"Beginner-Friendly: {risk.suitable_for_beginners}")
    print(f"\\nWarnings:")
    for warning in risk.warnings:
        print(f"  - {warning}")
\`\`\`

---

## Production Considerations

### Assignment and Exercise

**Assignment Risk:**
- Short options can be assigned at ANY time (American style)
- Usually happens when ITM at expiration or before ex-dividend
- You must fulfill the obligation (deliver or accept shares)

**Auto-Exercise:**
- Most brokers auto-exercise ITM options >$0.01 at expiration
- Can notify broker to NOT exercise if desired

**Corporate Actions:**
- Stock splits: Strike and quantity adjusted
- Dividends: Early exercise possible on deep ITM calls
- Mergers: Settlement may change to cash or new ticker

### Margin Requirements

\`\`\`python
def calculate_margin_requirement(position_type: str,
                                strike: float,
                                stock_price: float,
                                contracts: int) -> float:
    """
    Calculate margin requirement (simplified)
    
    Actual requirements vary by broker and account type
    """
    if position_type in ["long_call", "long_put"]:
        # No margin, just pay premium
        return 0
    
    elif position_type == "short_call":
        # Naked call: ~20% of stock value + premium
        margin_per_share = stock_price * 0.20
        return margin_per_share * 100 * contracts
    
    elif position_type == "short_put":
        # Cash-secured put: Full strike value
        return strike * 100 * contracts
    
    else:
        raise ValueError(f"Unknown position type: {position_type}")

# Examples
print("Margin Requirements:")
print(f"Long call: \\$\{calculate_margin_requirement('long_call', 100, 100, 1):,.0f}")
print(f"Short call: \\$\{calculate_margin_requirement('short_call', 100, 100, 1):,.0f}")
print(f"Short put: \\$\{calculate_margin_requirement('short_put', 100, 100, 1):,.0f}")
\`\`\`

### Tax Considerations

**Long Options:**
- Holding period starts when opened
- If held >1 year, long-term capital gains (lower rate)
- If closed <1 year, short-term gains (ordinary income)

**Short Options:**
- Premium collected is income when option expires or is bought back
- If assigned, affects cost basis of stock

**Wash Sale Rule:**
- Buying back substantially identical position within 30 days
- Disallows loss deduction

---

## Common Mistakes

### 1. Not Understanding Assignment

**Mistake:** Selling options without understanding you can be assigned.

**Example:** Sell 100 call on stock trading at $110. Think "it won't be exercised until expiration." Wrong! Buyer can exercise any time. You wake up short 100 shares at $100, stock now $115. $1500 loss + transaction costs.

### 2. Over-Leveraging

**Mistake:** Buying too many options relative to account size.

**Example:** $10K account, buy 20 contracts ($500 each) = $10K. Stock doesn't move, lose entire account.

**Correct:** Risk only 2-5% per trade. With $10K, risk $200-500 max → 1-2 contracts.

### 3. Ignoring Theta Decay

**Mistake:** Holding long options too long without stock movement.

**Example:** Buy ATM call for $5, stock flat for 2 weeks, option now $3 (40% loss from decay alone).

**Lesson:** Have a time-based exit plan. If stock hasn't moved by 50% of time to expiration, exit.

---

## Summary

**Four Basic Positions:**

| Position | Max Profit | Max Loss | Risk Level | Suitable For |
|----------|-----------|----------|------------|--------------|
| Long Call | Unlimited | Premium | Low | Beginners |
| Short Call | Premium | Unlimited | Extreme | Experts only (or covered) |
| Long Put | Strike - Premium | Premium | Low | Beginners |
| Short Put | Premium | Strike - Premium | Medium | Cash-secured OK |

**Key Concepts:**
- **Put-Call Parity**: C - P = S - K×e^(-rT)
- **Synthetic Positions**: Replicate stocks/options using combinations
- **Assignment Risk**: Short options can be assigned anytime
- **Margin Requirements**: Vary dramatically by position type

**Risk Management:**
- Long options: Max loss = premium (defined risk)
- Short naked calls: Unlimited loss potential (AVOID)
- Short cash-secured puts: Defined risk, suitable for patient buyers
- Always size positions appropriately (2-5% risk per trade)

**Next Steps:**
- Learn option pricing (Black-Scholes model)
- Understand the Greeks (delta, gamma, theta, vega, rho)
- Study spreads and multi-leg strategies
- Build an options trading system

In the next section, we'll dive into the **Black-Scholes model** - how options are actually priced in the market.
`,
};
