export const optionsFundamentals = {
  title: 'Options Fundamentals',
  id: 'options-fundamentals',
  content: `
# Options Fundamentals

## Introduction

Options are financial contracts that give the holder the **right, but not the obligation**, to buy or sell an underlying asset at a specified price (strike) on or before a specified date (expiration). Unlike stocks where you own a piece of a company, options are derivative contracts whose value derives from the underlying asset.

**Why Options Matter for Engineers:**
- **Leverage**: Control $10,000 worth of stock with $500
- **Defined risk**: Maximum loss is premium paid (for buyers)
- **Flexibility**: Profit from up, down, or sideways markets
- **Hedging**: Protect portfolios like insurance
- **Income generation**: Sell options to collect premium

**Options in the Real World:**
- **Retail traders**: 40M+ options contracts trade daily
- **Market makers**: Citadel Securities, Jane Street, Virtu
- **Hedge funds**: Volatility arbitrage, dispersion trading
- **Corporations**: Hedging currency and commodity exposure

By the end of this section, you'll understand:
- Call and put option mechanics
- Payoff diagrams and profit/loss profiles
- Intrinsic vs time value
- American vs European exercise styles
- Moneyness (ITM, ATM, OTM)
- How to calculate and visualize option payoffs in Python

---

## Call Options

### Definition

A **call option** gives the buyer the **right to BUY** the underlying asset at the strike price.

**Contract Specifications:**
- **Underlying**: Stock, index, ETF (e.g., AAPL, SPY)
- **Strike Price (K)**: Price at which you can buy (e.g., $100)
- **Expiration Date**: When the option expires (e.g., Jan 20, 2024)
- **Premium**: Price paid for the option (e.g., $5.00 per share)
- **Contract Size**: Usually 100 shares per contract

**Example:**
- Buy 1 AAPL Jan 100 Call for $5.00
- You pay: $5.00 × 100 = $500
- You have the right to buy 100 AAPL shares at $100 until January expiration
- If AAPL is at $110 at expiration, you exercise and make profit

### Call Option Payoff

**At Expiration:**
\`\`\`
Call Payoff = max(S - K, 0)
where:
  S = Stock price at expiration
  K = Strike price
\`\`\`

**Why max()?** You only exercise if profitable (S > K). Otherwise, let it expire worthless.

**Profit/Loss:**
\`\`\`
Call P&L = max(S - K, 0) - Premium Paid
\`\`\`

**Example Scenarios:**

| Stock Price (S) | Strike (K) | Payoff | Premium | Profit/Loss |
|-----------------|-----------|---------|---------|-------------|
| $90 | $100 | $0 | $5 | -$5 |
| $100 | $100 | $0 | $5 | -$5 |
| $105 | $100 | $5 | $5 | $0 (breakeven) |
| $110 | $100 | $10 | $5 | +$5 |
| $120 | $100 | $20 | $5 | +$15 |

**Key Insight:** Maximum loss is limited to premium paid ($5). Maximum profit is unlimited as stock can go to infinity.

---

## Put Options

### Definition

A **put option** gives the buyer the **right to SELL** the underlying asset at the strike price.

**Example:**
- Buy 1 AAPL Jan 100 Put for $4.00
- You pay: $4.00 × 100 = $400
- You have the right to sell 100 AAPL shares at $100 until January expiration
- If AAPL drops to $90, you exercise and make profit

### Put Option Payoff

**At Expiration:**
\`\`\`
Put Payoff = max(K - S, 0)
\`\`\`

**Profit/Loss:**
\`\`\`
Put P&L = max(K - S, 0) - Premium Paid
\`\`\`

**Example Scenarios:**

| Stock Price (S) | Strike (K) | Payoff | Premium | Profit/Loss |
|-----------------|-----------|---------|---------|-------------|
| $110 | $100 | $0 | $4 | -$4 |
| $100 | $100 | $0 | $4 | -$4 |
| $96 | $100 | $4 | $4 | $0 (breakeven) |
| $90 | $100 | $10 | $4 | +$6 |
| $80 | $100 | $20 | $4 | +$16 |

**Key Insight:** Maximum loss is premium paid ($4). Maximum profit is K - 0 (stock can't go below $0).

---

## Payoff Diagrams

Visual representation of profit/loss at expiration.

### Long Call Payoff

\`\`\`
Profit
  ^
  |     /
  |    /
  |   /
--+--/--------> Stock Price
  | /    K
  |/
  |(breakeven = K + premium)
\`\`\`

### Long Put Payoff

\`\`\`
Profit
  ^
  |\\
  | \\
  |  \\
--+---\\------> Stock Price
  |    \\  K
  |     \\
  |      (breakeven = K - premium)
\`\`\`

---

## Python: Call and Put Payoff Calculator

\`\`\`python
"""
Options Payoff Calculator
Production-ready implementation with type hints and validation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple, Optional
from dataclasses import dataclass

@dataclass
class OptionContract:
    """Represents an option contract"""
    option_type: str  # 'call' or 'put'
    strike: float
    premium: float
    quantity: int = 1  # Number of contracts (positive = long, negative = short)
    
    def __post_init__(self):
        """Validate inputs"""
        if self.option_type.lower() not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.premium < 0:
            raise ValueError("premium cannot be negative")


def call_payoff(stock_price: Union[float, np.ndarray], 
                strike: float) -> Union[float, np.ndarray]:
    """
    Calculate call option payoff at expiration
    
    Payoff = max(S - K, 0)
    
    Args:
        stock_price: Current stock price (or array of prices)
        strike: Strike price of the option
        
    Returns:
        Payoff per share (not including premium)
    """
    return np.maximum(stock_price - strike, 0)


def put_payoff(stock_price: Union[float, np.ndarray], 
               strike: float) -> Union[float, np.ndarray]:
    """
    Calculate put option payoff at expiration
    
    Payoff = max(K - S, 0)
    
    Args:
        stock_price: Current stock price (or array of prices)
        strike: Strike price of the option
        
    Returns:
        Payoff per share (not including premium)
    """
    return np.maximum(strike - stock_price, 0)


def option_profit_loss(option: OptionContract, 
                       stock_price: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate profit/loss for an option position
    
    P&L = Quantity × (Payoff - Premium) × 100 (shares per contract)
    
    Args:
        option: OptionContract with specifications
        stock_price: Current stock price (or array of prices)
        
    Returns:
        Profit/loss in dollars
    """
    if option.option_type.lower() == 'call':
        payoff = call_payoff(stock_price, option.strike)
    else:
        payoff = put_payoff(stock_price, option.strike)
    
    # P&L per share, then multiply by quantity and contract size
    pnl_per_share = (payoff - option.premium) * option.quantity
    pnl_total = pnl_per_share * 100  # 100 shares per contract
    
    return pnl_total


def calculate_breakeven(option: OptionContract) -> float:
    """
    Calculate breakeven stock price at expiration
    
    For long call: breakeven = strike + premium
    For long put: breakeven = strike - premium
    For short positions: opposite
    
    Args:
        option: OptionContract
        
    Returns:
        Breakeven stock price
    """
    if option.quantity > 0:  # Long position
        if option.option_type.lower() == 'call':
            return option.strike + option.premium
        else:  # put
            return option.strike - option.premium
    else:  # Short position
        if option.option_type.lower() == 'call':
            return option.strike + option.premium
        else:  # put
            return option.strike - option.premium


# Example Usage
if __name__ == "__main__":
    print("=" * 60)
    print("OPTIONS PAYOFF CALCULATOR")
    print("=" * 60)
    
    # Example 1: Long Call
    print("\\n### Example 1: Long Call ###")
    long_call = OptionContract(
        option_type='call',
        strike=100,
        premium=5,
        quantity=1
    )
    
    # Test at different stock prices
    test_prices = np.array([90, 95, 100, 105, 110, 115, 120])
    
    print(f"\\nOption: {long_call.quantity} × {long_call.option_type.upper()} "
          f"@ Strike ${long_call.strike}, Premium ${long_call.premium}")
    print(f"Breakeven: ${calculate_breakeven(long_call):.2f}")
    print("\\nStock Price | Payoff | P&L")
    print("-" * 40)
    
    for price in test_prices:
        payoff = call_payoff(price, long_call.strike)
        pnl = option_profit_loss(long_call, price)
        print(f"${price:6.2f}  | ${payoff:6.2f} | ${pnl:7.2f}")
    
    # Example 2: Long Put
    print("\\n### Example 2: Long Put ###")
    long_put = OptionContract(
        option_type='put',
        strike=100,
        premium=4,
        quantity=1
    )
    
    print(f"\\nOption: {long_put.quantity} × {long_put.option_type.upper()} "
          f"@ Strike ${long_put.strike}, Premium ${long_put.premium}")
    print(f"Breakeven: ${calculate_breakeven(long_put):.2f}")
    print("\\nStock Price | Payoff | P&L")
    print("-" * 40)
    
    for price in test_prices:
        payoff = put_payoff(price, long_put.strike)
        pnl = option_profit_loss(long_put, price)
        print(f"${price:6.2f}  | ${payoff:6.2f} | ${pnl:7.2f}")
    
    # Example 3: Short Call (negative quantity)
    print("\\n### Example 3: Short Call ###")
    short_call = OptionContract(
        option_type='call',
        strike=100,
        premium=5,
        quantity=-1  # Negative = short
    )
    
    print(f"\\nOption: {short_call.quantity} × {short_call.option_type.upper()} "
          f"@ Strike ${short_call.strike}, Premium ${short_call.premium}")
    print(f"Breakeven: ${calculate_breakeven(short_call):.2f}")
    print("\\nStock Price | Payoff | P&L")
    print("-" * 40)
    
    for price in test_prices:
        payoff = call_payoff(price, short_call.strike)
        pnl = option_profit_loss(short_call, price)
        print(f"${price:6.2f}  | ${payoff:6.2f} | ${pnl:7.2f}")
\`\`\`

**Output:**
\`\`\`
============================================================
OPTIONS PAYOFF CALCULATOR
============================================================

### Example 1: Long Call ###

Option: 1 × CALL @ Strike $100, Premium $5
Breakeven: $105.00

Stock Price | Payoff | P&L
----------------------------------------
$ 90.00  | $  0.00 | $-500.00
$ 95.00  | $  0.00 | $-500.00
$100.00  | $  0.00 | $-500.00
$105.00  | $  5.00 | $   0.00  ← Breakeven
$110.00  | $ 10.00 | $ 500.00
$115.00  | $ 15.00 | $1000.00
$120.00  | $ 20.00 | $1500.00
\`\`\`

---

## Visualizing Option Payoffs

\`\`\`python
"""
Option Payoff Visualization
"""

def plot_option_payoff(option: OptionContract, 
                       price_range: Optional[Tuple[float, float]] = None,
                       num_points: int = 200) -> None:
    """
    Plot payoff diagram for an option
    
    Args:
        option: OptionContract to plot
        price_range: (min_price, max_price) tuple, defaults to ±30% around strike
        num_points: Number of points for smooth curve
    """
    # Default price range: ±30% around strike
    if price_range is None:
        min_price = option.strike * 0.7
        max_price = option.strike * 1.3
    else:
        min_price, max_price = price_range
    
    # Generate price range
    stock_prices = np.linspace(min_price, max_price, num_points)
    
    # Calculate P&L
    pnl = option_profit_loss(option, stock_prices)
    
    # Calculate breakeven
    breakeven = calculate_breakeven(option)
    
    # Plot
    plt.figure(figsize=(12, 7))
    
    # P&L curve
    plt.plot(stock_prices, pnl, 'b-', linewidth=2, 
             label=f'{option.quantity} × {option.option_type.upper()} @ ${option.strike}')
    
    # Zero line
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Strike line
    plt.axvline(x=option.strike, color='red', linestyle='--', 
                linewidth=1, alpha=0.7, label=f'Strike: ${option.strike}')
    
    # Breakeven line
    plt.axvline(x=breakeven, color='green', linestyle='--', 
                linewidth=1, alpha=0.7, label=f'Breakeven: ${breakeven:.2f}')
    
    # Styling
    plt.xlabel('Stock Price at Expiration', fontsize=12)
    plt.ylabel('Profit / Loss ($)', fontsize=12)
    plt.title(f'Option Payoff Diagram: {option.quantity} × {option.option_type.upper()} ' +
              f'Strike ${option.strike}, Premium ${option.premium}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Format y-axis as currency
    from matplotlib.ticker import FuncFormatter
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Shade profit/loss regions
    plt.fill_between(stock_prices, 0, pnl, where=(pnl > 0), 
                     color='green', alpha=0.2, label='Profit')
    plt.fill_between(stock_prices, 0, pnl, where=(pnl < 0), 
                     color='red', alpha=0.2, label='Loss')
    
    plt.tight_layout()
    plt.show()


# Example: Plot all four basic positions
if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    options = [
        OptionContract('call', 100, 5, 1),   # Long call
        OptionContract('put', 100, 4, 1),    # Long put
        OptionContract('call', 100, 5, -1),  # Short call
        OptionContract('put', 100, 4, -1),   # Short put
    ]
    
    titles = ['Long Call', 'Long Put', 'Short Call', 'Short Put']
    
    for idx, (option, title) in enumerate(zip(options, titles)):
        ax = axes[idx // 2, idx % 2]
        
        # Generate price range
        stock_prices = np.linspace(70, 130, 200)
        pnl = option_profit_loss(option, stock_prices)
        
        # Plot
        ax.plot(stock_prices, pnl, 'b-', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=option.strike, color='red', linestyle='--', alpha=0.7)
        ax.fill_between(stock_prices, 0, pnl, where=(pnl > 0), color='green', alpha=0.2)
        ax.fill_between(stock_prices, 0, pnl, where=(pnl < 0), color='red', alpha=0.2)
        
        ax.set_xlabel('Stock Price at Expiration')
        ax.set_ylabel('Profit / Loss ($)')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('four_basic_option_positions.png', dpi=300, bbox_inches='tight')
    plt.show()
\`\`\`

---

## Intrinsic Value vs Time Value

Every option's price consists of two components:

\`\`\`
Option Price = Intrinsic Value + Time Value
\`\`\`

### Intrinsic Value

The value if you exercised the option **right now**.

**For Calls:**
\`\`\`
Intrinsic Value = max(S - K, 0)
\`\`\`

**For Puts:**
\`\`\`
Intrinsic Value = max(K - S, 0)
\`\`\`

**Example:**
- AAPL trading at $110
- 100-strike call: Intrinsic = max(110 - 100, 0) = $10
- 100-strike put: Intrinsic = max(100 - 110, 0) = $0

### Time Value

The **extra** premium above intrinsic value, representing:
- Time remaining until expiration
- Volatility of the underlying
- Interest rates
- Dividends

\`\`\`
Time Value = Option Price - Intrinsic Value
\`\`\`

**Example:**
- 100-strike call trading at $15
- Intrinsic value = $10
- Time value = $15 - $10 = $5

**Key Properties:**
- Time value is always ≥ 0
- Time value decays to $0 at expiration (theta decay)
- ATM options have maximum time value
- Deep ITM/OTM options have minimal time value

---

## Moneyness

**Moneyness** describes the relationship between stock price and strike price.

### In-The-Money (ITM)

Option has intrinsic value.

- **Call ITM**: Stock price > Strike (S > K)
- **Put ITM**: Strike > Stock price (K > S)

**Example:**
- Stock at $110
- 100-strike call is ITM (can buy at $100, sell at $110)
- 120-strike put is ITM (can sell at $120, buy at $110)

### At-The-Money (ATM)

Strike equals (or very close to) stock price.

- **ATM**: S ≈ K

**Example:**
- Stock at $100
- 100-strike call and put are both ATM

**Key Property:** ATM options have the **highest time value** and **highest gamma** (sensitivity to price changes).

### Out-Of-The-Money (OTM)

Option has no intrinsic value.

- **Call OTM**: Strike > Stock price (K > S)
- **Put OTM**: Stock price > Strike (S > K)

**Example:**
- Stock at $100
- 110-strike call is OTM (no benefit to buy at $110 when market is $100)
- 90-strike put is OTM (no benefit to sell at $90 when market is $100)

---

## Python: Moneyness Calculator

\`\`\`python
"""
Option Moneyness Classification
"""

from enum import Enum

class Moneyness(Enum):
    """Moneyness classification"""
    DEEP_ITM = "Deep In-The-Money"
    ITM = "In-The-Money"
    ATM = "At-The-Money"
    OTM = "Out-Of-The-Money"
    DEEP_OTM = "Deep Out-Of-The-Money"


def classify_moneyness(stock_price: float, 
                       strike: float, 
                       option_type: str,
                       atm_threshold: float = 0.02) -> Moneyness:
    """
    Classify option moneyness
    
    Args:
        stock_price: Current stock price
        strike: Option strike price
        option_type: 'call' or 'put'
        atm_threshold: % threshold for ATM classification (default 2%)
        
    Returns:
        Moneyness classification
    """
    # Calculate moneyness ratio
    moneyness_ratio = stock_price / strike
    
    # ATM check (within threshold)
    if abs(moneyness_ratio - 1.0) < atm_threshold:
        return Moneyness.ATM
    
    if option_type.lower() == 'call':
        if moneyness_ratio >= 1.15:  # 15% ITM
            return Moneyness.DEEP_ITM
        elif moneyness_ratio > 1.0:
            return Moneyness.ITM
        elif moneyness_ratio <= 0.85:  # 15% OTM
            return Moneyness.DEEP_OTM
        else:
            return Moneyness.OTM
    else:  # put
        if moneyness_ratio <= 0.85:  # 15% ITM
            return Moneyness.DEEP_ITM
        elif moneyness_ratio < 1.0:
            return Moneyness.ITM
        elif moneyness_ratio >= 1.15:  # 15% OTM
            return Moneyness.DEEP_OTM
        else:
            return Moneyness.OTM


def calculate_intrinsic_time_value(stock_price: float,
                                   strike: float,
                                   option_price: float,
                                   option_type: str) -> Tuple[float, float]:
    """
    Decompose option price into intrinsic and time value
    
    Args:
        stock_price: Current stock price
        strike: Option strike price
        option_price: Current option price
        option_type: 'call' or 'put'
        
    Returns:
        (intrinsic_value, time_value)
    """
    if option_type.lower() == 'call':
        intrinsic = max(stock_price - strike, 0)
    else:  # put
        intrinsic = max(strike - stock_price, 0)
    
    time_value = option_price - intrinsic
    
    return intrinsic, time_value


# Example Usage
if __name__ == "__main__":
    stock_price = 100
    
    print("=" * 70)
    print(f"OPTION MONEYNESS ANALYSIS (Stock Price: ${stock_price})")
    print("=" * 70)
    
    # Test different strikes
    strikes = [80, 90, 95, 98, 100, 102, 105, 110, 120]
    option_prices_call = [20.5, 11.2, 7.1, 4.8, 3.5, 2.4, 1.3, 0.5, 0.1]
    
    print("\\n### CALL OPTIONS ###")
    print(f"{'Strike':<8} {'Price':<8} {'Intrinsic':<12} {'Time Val':<12} {'Moneyness':<25}")
    print("-" * 70)
    
    for strike, price in zip(strikes, option_prices_call):
        intrinsic, time_val = calculate_intrinsic_time_value(
            stock_price, strike, price, 'call'
        )
        moneyness = classify_moneyness(stock_price, strike, 'call')
        
        print(f"${strike:<7.0f} ${price:<7.2f} ${intrinsic:<11.2f} ${time_val:<11.2f} {moneyness.value}")
\`\`\`

**Output:**
\`\`\`
======================================================================
OPTION MONEYNESS ANALYSIS (Stock Price: $100)
======================================================================

### CALL OPTIONS ###
Strike   Price    Intrinsic    Time Val     Moneyness
----------------------------------------------------------------------
$80      $20.50   $20.00       $0.50        Deep In-The-Money
$90      $11.20   $10.00       $1.20        In-The-Money
$95      $7.10    $5.00        $2.10        In-The-Money
$98      $4.80    $2.00        $2.80        At-The-Money
$100     $3.50    $0.00        $3.50        At-The-Money
$102     $2.40    $0.00        $2.40        At-The-Money
$105     $1.30    $0.00        $1.30        Out-Of-The-Money
$110     $0.50    $0.00        $0.50        Out-Of-The-Money
$120     $0.10    $0.00        $0.10        Deep Out-Of-The-Money
\`\`\`

**Key Observations:**
- ATM options ($98-$102 strikes) have maximum time value ($2.40-$3.50)
- Deep ITM option ($80 strike) mostly intrinsic ($20), minimal time ($0.50)
- Deep OTM option ($120 strike) almost worthless ($0.10), all time value

---

## American vs European Options

### European Options

- Can only be exercised **at expiration**
- Most index options (SPX, VIX)
- Simpler to price (Black-Scholes works directly)

**Example:** SPX 4000 Call expires Jan 20
- You can only exercise on Jan 20
- Before then, you can sell the option but not exercise

### American Options

- Can be exercised **anytime** before expiration
- Most stock options (AAPL, TSLA, etc.)
- More valuable than European (extra flexibility)
- Harder to price (binomial trees, finite difference)

**Example:** AAPL 150 Call expires Jan 20
- You can exercise today, tomorrow, or any day until Jan 20
- Useful if dividend is paid before expiration

**When to Exercise Early (American):**
1. **Deep ITM call with dividend**: Capture dividend
2. **Deep ITM put**: Time value very small, capture intrinsic now
3. **Risk management**: Lock in profits

**In Practice:** Rarely optimal to exercise early due to time value. Usually better to sell the option.

---

## Common Pitfalls

### 1. Confusing Payoff and Profit

**Wrong:**
\`\`\`python
# Forgetting premium
profit = max(stock_price - strike, 0)  # ❌ This is payoff, not profit
\`\`\`

**Correct:**
\`\`\`python
# Include premium in P&L calculation
profit = max(stock_price - strike, 0) - premium_paid  # ✅
\`\`\`

### 2. Not Accounting for Contract Size

**Wrong:**
\`\`\`python
pnl = option_payoff - premium  # ❌ Per share, not per contract
\`\`\`

**Correct:**
\`\`\`python
pnl = (option_payoff - premium) * 100  # ✅ 100 shares per contract
\`\`\`

### 3. Exercising vs Selling

**Mistake:** Exercising an option when selling is more profitable.

**Example:**
- Own 100-strike call, stock at $110
- Option trading at $12 (intrinsic $10 + time $2)
- **Exercise**: Get $10 payoff, lose $2 time value
- **Sell option**: Get $12, capture all value ✅

### 4. Ignoring Assignment Risk

**Short options can be assigned** at any time if American style.
- Short 100 call, stock jumps to $120
- Counterparty exercises, you must deliver shares at $100
- If you don't own shares, you're forced to buy at $120 (loss)

---

## Production Considerations

### 1. Data Quality

**Critical for pricing:**
- Clean bid/ask data
- Handle stale quotes
- Adjust for corporate actions (splits, dividends)

### 2. Contract Specifications

\`\`\`python
@dataclass
class OptionSpecification:
    """Production option contract specification"""
    symbol: str          # "AAPL"
    underlying: str      # "AAPL" (usually same, but can differ for indices)
    strike: float        # 100.00
    expiration: date     # 2024-01-20
    option_type: str     # "call" or "put"
    style: str           # "american" or "european"
    multiplier: int      # 100 (shares per contract)
    currency: str        # "USD"
    exchange: str        # "CBOE", "ISE", etc.
\`\`\`

### 3. Risk Management

**For option buyers:**
- Maximum loss = premium × contracts × multiplier
- Set stop losses based on option price (e.g., -50%)

**For option sellers:**
- Potentially unlimited loss (naked calls)
- Require margin (collateral)
- Use defined-risk strategies (spreads)

### 4. Transaction Costs

\`\`\`python
def calculate_all_in_cost(premium: float, 
                         contracts: int,
                         commission_per_contract: float = 0.65) -> float:
    """
    Calculate total cost including commissions
    
    Typical retail: $0.65 per contract
    Typical institutional: $0.10-0.25 per contract
    """
    option_cost = premium * contracts * 100
    commission = commission_per_contract * contracts
    total = option_cost + commission
    return total
\`\`\`

---

## Regulatory Considerations

### Options Trading Approval Levels

**Retail brokers** (TD Ameritrade, Interactive Brokers) have approval levels:

**Level 1:** Covered calls, cash-secured puts
**Level 2:** Long calls and puts (limited risk)
**Level 3:** Spreads (defined risk)
**Level 4:** Naked puts (high risk, margin required)
**Level 5:** Naked calls (unlimited risk, significant margin)

### Pattern Day Trader Rule (PDT)

- If account < $25K and make 4+ day trades in 5 days → flagged as PDT
- PDT accounts restricted to 3 day trades per 5 days
- **Options count:** Buying and selling same option same day = day trade

### Position Limits

**Exchange-imposed limits** on number of contracts:
- Prevent market manipulation
- Varies by underlying and time to expiration
- Example: AAPL might have 250,000 contract limit

### Assignment and Exercise

- **Exercise**: Initiated by long holder
- **Assignment**: Short seller is obligated to fulfill
- **Auto-exercise**: Most brokers auto-exercise ITM options at expiration (>$0.01)

---

## Real-World Example: SPY Options

\`\`\`python
"""
Real-world SPY option analysis
"""

# SPY trading at $450
spy_price = 450

# Option chain snapshot (hypothetical)
spy_calls = {
    440: {'bid': 12.50, 'ask': 12.60, 'volume': 10500},
    445: {'bid': 8.20, 'ask': 8.30, 'volume': 15200},
    450: {'bid': 5.10, 'ask': 5.20, 'volume': 25600},  # ATM
    455: {'bid': 3.00, 'ask': 3.10, 'volume': 18400},
    460: {'bid': 1.50, 'ask': 1.60, 'volume': 12100},
}

print("SPY OPTIONS (30 days to expiration)")
print(f"Underlying: ${spy_price}")
print("\\nStrike | Mid Price | Intrinsic | Time Value | Moneyness")
print("-" * 70)

for strike, data in spy_calls.items():
    mid_price = (data['bid'] + data['ask']) / 2
    intrinsic, time_val = calculate_intrinsic_time_value(
        spy_price, strike, mid_price, 'call'
    )
    moneyness = classify_moneyness(spy_price, strike, 'call')
    
    print(f"${strike:<4} | ${mid_price:>7.2f} | ${intrinsic:>9.2f} | ${time_val:>10.2f} | {moneyness.value}")
\`\`\`

**Key Insights:**
- ATM strike (450) has highest time value ($5.15)
- Deep ITM (440) is mostly intrinsic value
- Deep OTM (460) is almost all time value

---

## Summary

**Key Concepts:**
- **Call**: Right to buy at strike
- **Put**: Right to sell at strike
- **Payoff**: Value at expiration (not including premium)
- **Profit**: Payoff minus premium paid
- **Intrinsic value**: What you get if exercised now
- **Time value**: Extra premium for potential future moves
- **Moneyness**: ITM (has intrinsic), ATM (at strike), OTM (no intrinsic)
- **American**: Exercise anytime, **European**: Exercise only at expiration

**Maximum Loss (Buyers):**
- Long call: Premium paid
- Long put: Premium paid

**Maximum Profit (Buyers):**
- Long call: Unlimited
- Long put: Strike - premium (stock can't go below 0)

**Next Steps:**
- Understand how to price options (Black-Scholes)
- Learn the Greeks (delta, gamma, theta, vega)
- Explore options strategies (spreads, straddles)
- Build a trading system

**Remember:** Options are powerful tools but come with unique risks. Always understand your maximum loss before entering a trade. In the next section, we'll dive deeper into calls and puts, including short positions and advanced concepts like put-call parity.
`,
};

