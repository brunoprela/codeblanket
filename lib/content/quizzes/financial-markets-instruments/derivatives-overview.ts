export const derivativesOverviewQuiz = [
  {
    id: 'fm-1-3-q-1',
    question:
      'A company will receive €10M in 6 months and wants to hedge FX risk. Current spot: $1.10/€. 6-month forward: $1.08/€. Design: (1) Forward hedge, (2) Options hedge, (3) Compare costs/benefits, (4) When to use each strategy.',
    sampleAnswer: `**Scenario:** Company receives €10M in 6 months, worried EUR will weaken vs USD.

**Forward Hedge:**
- Action: Sell €10M forward at $1.08/€
- Locked-in USD: €10M × $1.08 = $10.8M
- Cost: Implicit in forward rate (2% discount from spot)
- Benefit: Certainty - know exactly what you'll receive
- Risk: If EUR strengthens to $1.15, miss upside

**Options Hedge:**
- Action: Buy €10M PUT option, strike $1.08, 6-month expiry
- Premium: ~2-3% = $200K-300K
- Locked-in minimum: $10.8M (if EUR falls below $1.08)
- Benefit: Participate if EUR strengthens (unlimited upside)
- Cost: Premium paid upfront

**Comparison:**
\`\`\`
Scenario 1: EUR falls to $1.00
- Forward: Receive $10.8M (hedged)
- Option: Exercise put, receive $10.8M - $250K premium = $10.55M
- Winner: Forward (no premium)

Scenario 2: EUR rises to $1.15
- Forward: Receive $10.8M (locked in, miss $700K upside)
- Option: Let expire, sell at spot = $11.5M - $250K = $11.25M
- Winner: Option (participate in upside)

Scenario 3: EUR stays at $1.08
- Forward: Receive $10.8M
- Option: Breakeven, receive $10.8M - $250K = $10.55M
- Winner: Forward
\`\`\`

**Decision Framework:**
- Use FORWARD if: High certainty EUR will weaken, need budget certainty, low risk tolerance
- Use OPTION if: Possible EUR strength, want upside participation, can afford premium

**Python Implementation:**
\`\`\`python
def compare_hedging_strategies (notional_eur, spot_rate, forward_rate, option_strike, option_premium_pct, future_spot_scenarios):
    results = []
    for future_spot in future_spot_scenarios:
        # Forward hedge
        forward_usd = notional_eur * forward_rate
        
        # Option hedge
        option_premium = notional_eur * spot_rate * option_premium_pct
        if future_spot < option_strike:
            option_usd = notional_eur * option_strike - option_premium
        else:
            option_usd = notional_eur * future_spot - option_premium
        
        # Unhedged
        unhedged_usd = notional_eur * future_spot
        
        results.append({
            'future_spot': future_spot,
            'forward': forward_usd,
            'option': option_usd,
            'unhedged': unhedged_usd,
            'best_strategy': max([('forward', forward_usd), ('option', option_usd), ('unhedged', unhedged_usd)], key=lambda x: x[1])[0]
        })
    return results
\`\`\`

**Key Insight:** Forwards lock in rate (certainty), options provide insurance (flexibility). Choose based on risk appetite and view on currency direction.`,
    keyPoints: [
      'Forward: Lock in $1.08/€, certainty but no upside',
      'Option: Floor at $1.08/€, pay premium but keep upside if EUR strengthens',
      'Forward wins if EUR weakens or stays flat (no premium cost)',
      'Option wins if EUR strengthens significantly (upside > premium)',
      'Decision: Forward for certainty, option for flexibility',
    ],
  },
  {
    id: 'fm-1-3-q-2',
    question:
      'Explain how futures differ from forwards, focusing on: (1) Standardization vs customization, (2) Clearing house vs bilateral, (3) Mark-to-market vs settlement at maturity, (4) Why futures are preferred for speculation and forwards for hedging.',
    sampleAnswer: `**Futures vs Forwards Comparison:**

**1. Standardization:**
- Futures: Standard contracts (size, expiry, underlying)
  - Example: CME S&P 500 futures = $50 × index, quarterly expiry
  - Benefit: Liquid, easy to trade
  - Limitation: May not match exact hedging need

- Forwards: Customized to needs
  - Example: €7.3M in 87 days
  - Benefit: Perfect hedge match
  - Limitation: Illiquid, hard to exit

**2. Clearing House vs Bilateral:**
- Futures: Cleared through exchange (CME, ICE)
  - Counterparty risk eliminated (clearinghouse guarantees)
  - Collateral (margin) required
  - Daily mark-to-market

- Forwards: Bilateral OTC contract
  - Counterparty risk (bank might default)
  - Collateral negotiated
  - Settlement at maturity only

**3. Mark-to-Market:**
- Futures: Daily settlement
  - Each day: P&L settled in cash
  - Example: Long 1 S&P futures at 4000, closes at 4010 → Receive $500 ($50 × 10 points)
  - Next day starts at 4010 (zero P&L)
  - Benefit: No credit risk accumulation
  - Risk: Need cash for margin calls

- Forwards: Settlement at maturity
  - No cash flows until expiry
  - Credit risk builds over time
  - Example: Forward at $1.10, matures at $1.05 → Counterparty owes you 5 cents per euro
  - Risk: Counterparty might default if owes large sum

**4. Speculation vs Hedging:**

Futures Better for SPECULATION:
\`\`\`python
# Speculation: Bet on direction, don't need underlying
# Futures advantages:
advantages = {
    'Liquidity': 'Easy to enter/exit',
    'Leverage': '5-10% margin vs 100% cash',
    'Transparency': 'Exchange prices, no negotiation',
    'Low_cost': 'Tight bid-ask spreads'
}

# Example: Speculate on oil rising
# Buy 1 crude oil futures contract (1000 barrels)
# Margin: $5,000 (5% of $100K notional)
# If oil rises $1 → Profit $1,000 (20% return on margin)
\`\`\`

Forwards Better for HEDGING:
\`\`\`python
# Hedging: Exact match to underlying exposure
# Forwards advantages:
advantages = {
    'Customization': 'Exact amount, exact date',
    'No_margin': 'No cash outflows until maturity',
    'Simplicity': 'One contract, set and forget',
    'Accounting': 'Hedge accounting treatment easier'
}

# Example: Hedge €10M receivable in 83 days
# Forward contract: Exactly €10M, exactly 83 days
# Futures would be: Multiple €125K contracts, nearest expiry (imperfect)
\`\`\`

**Real-World Usage:**
- Speculators: 99% use futures (liquidity, leverage, transparency)
- Corporates hedging: 70% use forwards (exact match, no margin hassle)
- Banks: Use both (futures for trading, forwards for clients)

**Bottom Line:**
Futures = standardized, liquid, daily settlement, better for trading
Forwards = customized, OTC, maturity settlement, better for hedging`,
    keyPoints: [
      'Futures: standardized, exchange-traded, daily mark-to-market, liquid',
      'Forwards: customized, OTC, settlement at maturity, counterparty risk',
      'Futures eliminate counterparty risk via clearinghouse guarantee',
      'Futures better for speculation (liquidity, leverage, transparency)',
      'Forwards better for hedging (exact match to underlying exposure)',
    ],
  },
  {
    id: 'fm-1-3-q-3',
    question:
      'Design an options pricing system that implements Black-Scholes, calculates Greeks (delta, gamma, vega, theta), simulates option P&L scenarios, and provides hedging recommendations. Include pricing model, risk metrics, and hedge ratios.',
    sampleAnswer: `**Options Pricing & Risk Management System:**

\`\`\`python
import numpy as np
from scipy.stats import norm
from typing import Dict

class BlackScholesOption:
    def __init__(self, S, K, T, r, sigma, option_type='call'):
        """
        S: Spot price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.type = option_type
    
    def d1(self):
        return (np.log (self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt (self.T))
    
    def d2(self):
        return self.d1() - self.sigma * np.sqrt (self.T)
    
    def price (self):
        """Black-Scholes option price"""
        d1, d2 = self.d1(), self.d2()
        
        if self.type == 'call':
            price = self.S * norm.cdf (d1) - self.K * np.exp(-self.r * self.T) * norm.cdf (d2)
        else:  # put
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        
        return price
    
    def delta (self):
        """
        Delta: ∂V/∂S - Price sensitivity to underlying
        For hedging: Short delta shares to neutralize
        """
        d1 = self.d1()
        if self.type == 'call':
            return norm.cdf (d1)
        else:
            return norm.cdf (d1) - 1
    
    def gamma (self):
        """
        Gamma: ∂²V/∂S² - How delta changes
        High gamma = delta changes rapidly (need frequent rehedging)
        """
        d1 = self.d1()
        return norm.pdf (d1) / (self.S * self.sigma * np.sqrt (self.T))
    
    def vega (self):
        """
        Vega: ∂V/∂σ - Sensitivity to volatility
        Long options = positive vega (benefit from vol increase)
        """
        d1 = self.d1()
        return self.S * norm.pdf (d1) * np.sqrt (self.T) / 100  # Per 1% vol change
    
    def theta (self):
        """
        Theta: ∂V/∂t - Time decay
        Long options = negative theta (lose value over time)
        """
        d1, d2 = self.d1(), self.d2()
        
        term1 = -(self.S * norm.pdf (d1) * self.sigma) / (2 * np.sqrt (self.T))
        
        if self.type == 'call':
            term2 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf (d2)
        else:
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
        
        return (term1 + term2) / 365  # Per day
    
    def rho (self):
        """
        Rho: ∂V/∂r - Sensitivity to interest rates
        Usually smallest Greek (rates change slowly)
        """
        d2 = self.d2()
        
        if self.type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf (d2) / 100
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2) / 100

class OptionPortfolio:
    """Manage portfolio of options"""
    
    def __init__(self):
        self.positions = []  # List of (option, quantity)
    
    def add_position (self, option: BlackScholesOption, quantity: int):
        self.positions.append((option, quantity))
    
    def portfolio_greeks (self) -> Dict:
        """Calculate portfolio-level Greeks"""
        total_delta = sum (opt.delta() * qty for opt, qty in self.positions)
        total_gamma = sum (opt.gamma() * qty for opt, qty in self.positions)
        total_vega = sum (opt.vega() * qty for opt, qty in self.positions)
        total_theta = sum (opt.theta() * qty for opt, qty in self.positions)
        
        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'vega': total_vega,
            'theta': total_theta
        }
    
    def calculate_hedge_ratio (self, underlying_price: float) -> Dict:
        """
        Calculate hedge ratio to delta-neutral portfolio
        """
        greeks = self.portfolio_greeks()
        
        # Shares to short to neutralize delta
        shares_to_short = greeks['delta']
        
        # Notional hedge
        notional_hedge = shares_to_short * underlying_price
        
        return {
            'portfolio_delta': greeks['delta'],
            'shares_to_short': shares_to_short,
            'notional_hedge': notional_hedge,
            'interpretation': f'Short {shares_to_short:.0f} shares to neutralize delta'
        }
    
    def simulate_pnl (self, spot_scenarios, vol_scenarios, days_forward=1):
        """
        Simulate P&L under different market scenarios
        """
        results = []
        
        for new_spot in spot_scenarios:
            for new_vol in vol_scenarios:
                pnl = 0
                
                for option, qty in self.positions:
                    # Create new option with updated parameters
                    new_option = BlackScholesOption(
                        S=new_spot,
                        K=option.K,
                        T=max(0, option.T - days_forward/365),
                        r=option.r,
                        sigma=new_vol,
                        option_type=option.type
                    )
                    
                    # P&L = (New Value - Old Value) × Quantity
                    if new_option.T > 0:
                        pnl += (new_option.price() - option.price()) * qty
                    else:
                        # Expired
                        if option.type == 'call':
                            intrinsic = max(0, new_spot - option.K)
                        else:
                            intrinsic = max(0, option.K - new_spot)
                        pnl += (intrinsic - option.price()) * qty
                
                results.append({
                    'spot': new_spot,
                    'vol': new_vol,
                    'pnl': pnl
                })
        
        return results

# Example Usage
print("=== Options Pricing & Risk Management ===\\n")

# Create option
option = BlackScholesOption(
    S=100,      # Stock at $100
    K=105,      # Strike $105 (5% OTM)
    T=0.25,     # 3 months
    r=0.05,     # 5% risk-free rate
    sigma=0.30,  # 30% volatility
    option_type='call'
)

print(f"Call Option: Strike \${option.K}, {option.T*12:.0f} months")
print(f"Underlying: \${option.S}, Vol: {option.sigma*100:.0f}%\\n")

# Price
price = option.price()
print(f"Fair Value: \${price:.2f}")

# Greeks
print(f"\\nGreeks:")
print(f"  Delta: {option.delta():.4f} (hedge by shorting {option.delta():.2f} shares)")
print(f"  Gamma: {option.gamma():.4f} (delta changes {option.gamma():.4f} per $1 move)")
print(f"  Vega: {option.vega():.2f} (gains \${option.vega():.2f} per 1% vol increase)")
print(f"  Theta: \${option.theta():.2f}/day (loses \${abs (option.theta()):.2f} per day)")

# Portfolio example
portfolio = OptionPortfolio()
portfolio.add_position (option, 100)  # Long 100 calls

greeks = portfolio.portfolio_greeks()
print(f"\\nPortfolio (100 calls):")
print(f"  Total Delta: {greeks['delta']:.0f}")
print(f"  Total Vega: \${greeks['vega']:.0f}")
print(f"  Total Theta: \${greeks['theta']:.0f}/day")

# Hedge recommendation
hedge = portfolio.calculate_hedge_ratio (option.S)
print(f"\\nHedge: {hedge['interpretation']}")
print(f"  Notional: \${hedge['notional_hedge']:,.0f}")

# Simulate scenarios
scenarios = portfolio.simulate_pnl(
        spot_scenarios = [95, 100, 105],
        vol_scenarios = [0.25, 0.30, 0.35],
        days_forward = 30
    )

print(f"\\nP&L Scenarios (30 days forward):")
for result in scenarios[: 3]:
print(f"  Spot=\${result['spot']}, Vol={result['vol']*100:.0f}%: P&L=\${result['pnl']:,.0f}")
\`\`\`

**Key Components:**
1. Black-Scholes pricing engine
2. Greeks calculation (delta, gamma, vega, theta, rho)
3. Portfolio aggregation
4. Delta hedging recommendations
5. Scenario analysis

**Production Enhancements:**
- Implied volatility solver (given price, solve for vol)
- American options (early exercise, use binomial trees)
- Volatility smile (different vols by strike)
- Real-time market data integration`,
    keyPoints: [
      'Black-Scholes: Closed-form solution for European options',
      'Delta: Hedge ratio (short delta shares to neutralize)',
      'Gamma: How delta changes (high gamma = frequent rehedging needed)',
      'Vega: Volatility sensitivity (long options = positive vega)',
      'Theta: Time decay (long options lose value daily)',
    ],
  },
];
