export const derivativesOverview = {
    title: 'Derivatives Overview',
    id: 'derivatives-overview',
    content: `
# Derivatives Overview

## Introduction

A **derivative** is a financial contract whose value depends on (derives from) an underlying asset, index, or rate.

**Why critical for engineers**:
- $600+ trillion notional derivatives market (10x global GDP!)
- Essential for hedging interest rate, currency, credit risk
- Complex pricing requires sophisticated models
- High leverage = significant risk + opportunity

**What you'll build**: Derivatives pricing library, risk calculator, hedging optimizer.

---

## Core Derivative Types

### 1. Forwards

**Forward contract** = Agreement to buy/sell asset at future date for predetermined price.

**Characteristics**:
- **OTC (Over-the-counter)**: Customized, bilateral agreement
- **No daily settlement**: Settles at maturity only
- **Counter party risk**: Risk that other party defaults

**Example**:
\`\`\`
Company needs €10M in 6 months
Forward contract: Buy €10M at $1.10/€ in 6 months
Locks in exchange rate (hedge FX risk)
\`\`\`

### 2. Futures

**Futures contract** = Standardized forward traded on exchange.

**Characteristics**:
- **Exchange-traded**: Standardized (size, maturity, delivery)
- **Daily settlement**: Marked-to-market daily
- **Clearinghouse**: Eliminates counterparty risk
- **Margin requirements**: Post collateral

**Example**:
\`\`\`
Treasury bond futures (CME)
- Contract size: $100,000 face value
- Delivery: Specified bonds ("cheapest-to-deliver")
- Margin: ~$4,000 per contract
- Leverage: Control $100K with $4K (25:1)
\`\`\`

### 3. Swaps

**Swap** = Agreement to exchange cash flows based on different terms.

**Types**:
- **Interest rate swap**: Fixed ↔ Floating rates
- **Currency swap**: Cash flows in different currencies
- **Credit default swap (CDS)**: Default protection

**Example (Interest Rate Swap)**:
\`\`\`
Company has floating-rate debt (SOFR + 2%), wants fixed
Swap: Pay 4.5% fixed, receive SOFR
Net: Pay 4.5% + 2% - SOFR = 6.5% fixed (hedged)
\`\`\`

### 4. Options

**Option** = Right (not obligation) to buy/sell at specified price.

**Types**:
- **Call option**: Right to BUY
- **Put option**: Right to SELL

**Characteristics**:
- **Premium**: Upfront cost to buy option
- **Strike price**: Price at which can exercise
- **Expiration**: Last date to exercise

**Example**:
\`\`\`
Buy call option on 10-year Treasury futures
- Strike: 112 (112% of par)
- Premium: $2,000
- Expiration: 3 months

If futures > 112: Exercise, profit = (Futures - 112) - $2
If futures < 112: Let expire, lose $2,000 premium
\`\`\`

---

## Key Derivatives Concepts

### Notional vs Market Value

**Notional amount** = Reference amount for calculating payments (not exchanged).

**Market value** = Current worth of contract (usually much smaller).

**Example**:
\`\`\`
Interest rate swap: $100M notional
- Notional not exchanged
- Only net interest payments exchanged
- Market value: ~$2M (PV of future payment differences)

Leverage: $2M market exposure controls $100M notional (50:1)
\`\`\`

### Mark-to-Market

**Mark-to-market (MTM)** = Daily revaluation at current market prices.

**Futures**: Settled daily (gains/losses posted to margin account).

**Swaps/OTC**: Revalued daily but settled at payment dates.

**Example**:
\`\`\`
Day 1: Buy 10 Treasury futures at 110.00
Day 2: Futures fall to 109.50
MTM loss: 10 contracts × $100,000 × 0.005 = $5,000
Must post additional $5,000 margin or position closed
\`\`\`

### Margin

**Initial margin** = Collateral required to open position.

**Maintenance margin** = Minimum balance required.

**Margin call** = Demand to add funds if below maintenance margin.

**Example**:
\`\`\`
Treasury futures: $100,000 notional
Initial margin: $4,000 (4%)
Maintenance margin: $3,000

Position loses $1,500 → Margin = $2,500
Below maintenance → Margin call for $1,500
\`\`\`

---

## Python: Derivatives Pricing Foundations

\`\`\`python
"""
Derivatives Pricing Core Library
"""
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
import numpy as np
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)


class OptionType(Enum):
    """Option types"""
    CALL = "call"
    PUT = "put"


class ContractType(Enum):
    """Derivative contract types"""
    FORWARD = "forward"
    FUTURE = "future"
    SWAP = "swap"
    OPTION = "option"


@dataclass
class ForwardContract:
    """
    Forward contract pricing
    
    Example:
        >>> forward = ForwardContract(
        ...     spot_price=110.0,
        ...     forward_price=112.0,
        ...     maturity_years=0.5,
        ...     risk_free_rate=0.04
        ... )
        >>> fair_value = forward.fair_forward_price()
        >>> pnl = forward.mark_to_market(current_forward=113.0)
    """
    spot_price: float
    forward_price: float  # Agreed forward price
    maturity_years: float
    risk_free_rate: float
    notional: float = 1000000.0  # $1M default
    position: str = "long"  # "long" or "short"
    
    def fair_forward_price(self) -> float:
        """
        Calculate theoretical fair forward price
        
        F = S × e^(rT)
        
        Where:
        - S = spot price
        - r = risk-free rate
        - T = time to maturity
        """
        fair_price = self.spot_price * np.exp(
            self.risk_free_rate * self.maturity_years
        )
        
        logger.debug(f"Fair forward price: {fair_price:.2f}")
        
        return fair_price
    
    def mark_to_market(self, current_forward: float) -> float:
        """
        Calculate current MTM value
        
        MTM = (F_current - F_agreed) × Notional × DF
        
        Where:
        - F_current = current forward price
        - F_agreed = originally agreed forward price
        - DF = discount factor
        """
        # Discount factor
        df = np.exp(-self.risk_free_rate * self.maturity_years)
        
        # MTM value
        if self.position == "long":
            mtm = (current_forward - self.forward_price) * self.notional * df
        else:  # short
            mtm = (self.forward_price - current_forward) * self.notional * df
        
        logger.info(f"MTM: \${mtm:,.2f}")
        
        return mtm


@dataclass
class FuturesContract:
    """
    Futures contract with daily settlement
    
    Example:
        >>> futures = FuturesContract(
        ...     futures_price=110.0,
        ...     tick_size=0.015625,  # 1/64 for Treasury futures
        ...     tick_value=15.625,    # $15.625 per tick
        ...     contracts=10
        ... )
        >>> daily_pnl = futures.daily_settlement(new_price=110.5)
    """
    futures_price: float
    tick_size: float  # Minimum price movement
    tick_value: float  # Dollar value per tick
    contracts: int
    position: str = "long"  # "long" or "short"
    
    def daily_settlement(self, new_price: float) -> float:
        """
        Calculate daily P&L (marked-to-market)
        
        P&L = (New_price - Old_price) / Tick_size × Tick_value × Contracts
        """
        price_change = new_price - self.futures_price
        
        num_ticks = price_change / self.tick_size
        
        if self.position == "long":
            pnl = num_ticks * self.tick_value * self.contracts
        else:  # short
            pnl = -num_ticks * self.tick_value * self.contracts
        
        logger.info(f"Daily P&L: \${pnl:,.2f}
}")
        
        # Update futures price for next day
        self.futures_price = new_price
        
        return pnl
    
    def margin_requirement(
    self,
    initial_margin_per_contract: float
) -> float:
"""Calculate total margin required"""
return initial_margin_per_contract * self.contracts


class BlackScholesOption:
"""
Black - Scholes option pricing

Example:
        >>> option = BlackScholesOption(
    ...spot_price = 110.0,
    ...strike_price = 112.0,
    ...time_to_expiry = 0.25,  # 3 months
        ...risk_free_rate = 0.04,
    ...volatility = 0.15
        ... )
    >>> call_price = option.price(OptionType.CALL)
        >>> greeks = option.greeks(OptionType.CALL)
"""
    
    def __init__(
    self,
    spot_price: float,
    strike_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float
):
self.S = spot_price
self.K = strike_price
self.T = time_to_expiry
self.r = risk_free_rate
self.sigma = volatility
        
        # Calculate d1 and d2
self.d1 = (
    (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) /
    (self.sigma * np.sqrt(self.T))
)
self.d2 = self.d1 - self.sigma * np.sqrt(self.T)
    
    def price(self, option_type: OptionType) -> float:
"""
        Calculate option price

Call: S×N(d1) - K×e ^ (-rT)×N(d2)
Put: K×e ^ (-rT)×N(-d2) - S×N(-d1)
"""
if option_type == OptionType.CALL:
    price = (
        self.S * norm.cdf(self.d1) -
        self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
    )
else:  # PUT
price = (
    self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) -
    self.S * norm.cdf(-self.d1)
)

logger.info(f"{option_type.value} option price: \${price:.2f}")

return price
    
    def greeks(self, option_type: OptionType) -> Dict[str, float]:
"""
        Calculate option Greeks

Returns:
            Dict with delta, gamma, vega, theta, rho
        """
        # Delta
if option_type == OptionType.CALL:
    delta = norm.cdf(self.d1)
else:
delta = norm.cdf(self.d1) - 1
        
        # Gamma(same for call and put)
    gamma = norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
        
        # Vega(same for call and put)
    vega = self.S * norm.pdf(self.d1) * np.sqrt(self.T) / 100  # Per 1 % vol change
        
        # Theta
if option_type == OptionType.CALL:
    theta = (
        -(self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T)) -
        self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
    ) / 365  # Per day
        else:
theta = (
    -(self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T)) +
    self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
) / 365  # Per day
        
        # Rho
if option_type == OptionType.CALL:
    rho = (
        self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2) / 100
    )  # Per 1 % rate change
        else:
rho = (
    -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2) / 100
)

return {
    'delta': delta,
    'gamma': gamma,
    'vega': vega,
    'theta': theta,
    'rho': rho
}


# Example usage
if __name__ == "__main__":
    print("=== Forward Contract ===\\n")

forward = ForwardContract(
    spot_price = 110.0,
    forward_price = 112.0,
    maturity_years = 0.5,
    risk_free_rate = 0.04,
    notional = 10_000_000
)

fair_price = forward.fair_forward_price()
print(f"Fair forward price: {fair_price:.2f}")
print(f"Agreed forward price: {forward.forward_price:.2f}")
    
    # Mark - to - market
mtm = forward.mark_to_market(current_forward = 113.0)
print(f"MTM value: \${mtm:,.2f}")

print("\\n=== Futures Contract ===\\n")

futures = FuturesContract(
    futures_price = 110.0,
    tick_size = 0.015625,
    tick_value = 15.625,
    contracts = 10
)
    
    # Daily settlement
pnl_day1 = futures.daily_settlement(new_price = 110.5)
print(f"Day 1 P&L: \${pnl_day1:,.2f}")

pnl_day2 = futures.daily_settlement(new_price = 110.25)
print(f"Day 2 P&L: \${pnl_day2:,.2f}")

print("\\n=== Options ===\\n")

option = BlackScholesOption(
    spot_price = 110.0,
    strike_price = 112.0,
    time_to_expiry = 0.25,
    risk_free_rate = 0.04,
    volatility = 0.15
)

call_price = option.price(OptionType.CALL)
put_price = option.price(OptionType.PUT)

print(f"Call option price: \${call_price:.2f}")
print(f"Put option price: \${put_price:.2f}")
    
    # Greeks
call_greeks = option.greeks(OptionType.CALL)

print("\\nCall Option Greeks:")
print(f"  Delta: {call_greeks['delta']:.4f}")
print(f"  Gamma: {call_greeks['gamma']:.4f}")
print(f"  Vega: \${call_greeks['vega']:.2f}")
print(f"  Theta: \${call_greeks['theta']:.2f}/day")
print(f"  Rho: \${call_greeks['rho']:.2f}")
\`\`\`

---

## Derivatives Use Cases

### 1. Hedging

**Purpose**: Reduce risk exposure.

**Example**: Airline hedges jet fuel costs
- Buy crude oil futures
- Locks in fuel costs
- Protects from price spikes

### 2. Speculation

**Purpose**: Profit from price movements.

**Example**: Trader expects bond yields to rise
- Short Treasury futures
- If yields rise, futures fall, profit
- High leverage amplifies gains/losses

### 3. Arbitrage

**Purpose**: Exploit pricing inefficiencies.

**Example**: Basis trading
- Bond trades at 110.50
- Futures imply 110.00 bond price
- Buy futures, short bond
- Convergence at expiration = profit

---

## Key Takeaways

1. **Four main types**: Forwards (OTC custom), Futures (exchange standardized), Swaps (cash flow exchange), Options (rights not obligations)
2. **Notional vs Market**: $600T notional != $600T at risk (leverage effect)
3. **Mark-to-market**: Daily revaluation, futures settled daily
4. **Margin**: Collateral requirement, initial + maintenance, margin calls
5. **Use cases**: Hedging (risk reduction), speculation (directional), arbitrage (inefficiencies)
6. **Leverage**: Control large notional with small margin (amplifies risk + return)

**Next Section**: Forward and Futures Contracts - detailed mechanics, basis, carry, convergence.
`,
};

