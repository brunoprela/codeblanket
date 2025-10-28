export const marketImpactSlippageModels = {
  title: 'Market Impact and Slippage Models',
  id: 'market-impact-slippage-models',
  content: `
# Market Impact and Slippage Models

## Introduction

**Market impact** is the effect that a trade has on the price of a security. When you execute a large order, the very act of trading can move the market against you, increasing your transaction costs. **Slippage** is the difference between the expected price of a trade and the actual price at which it is executed.

Understanding and minimizing market impact is crucial for:
- **Institutional traders:** Managing large orders worth millions or billions of dollars.
- **Algorithmic traders:** Optimizing execution strategies to reduce costs.
- **Portfolio managers:** Estimating realistic transaction costs for performance attribution.
- **Quantitative researchers:** Backtesting strategies with realistic execution assumptions.

Market impact can significantly erode trading profits. A strategy that looks profitable on paper might be unprofitable after accounting for realistic market impact costs. This section explores the mathematical models, empirical findings, and practical algorithms used to understand and minimize market impact.

---

## Deep Technical Explanation: Market Impact Components

### 1. Types of Market Impact

**Temporary Impact (Transitory Impact)**:
- **Definition:** The immediate price movement caused by demanding liquidity, which partially reverts after the order is completed.
- **Cause:** Order book depletion. When a large market order consumes the best available quotes, it temporarily widens the spread and moves the price. As new limit orders arrive, liquidity is replenished, and the price partially recovers.
- **Time Horizon:** Seconds to minutes.
- **Example:** A 10,000 share market buy order pushes the price from $100.00 to $100.05 due to walking the book. After 5 minutes, new sell orders arrive, and the price settles back to $100.02.

**Permanent Impact (Information Impact)**:
- **Definition:** The lasting price movement that persists after the trade, reflecting information revealed by the order.
- **Cause:** Information signaling. The market interprets a large buy order as a signal that the buyer has positive information or strong demand, leading to a sustained price increase.
- **Time Horizon:** Minutes to hours, potentially days.
- **Example:** After the large buy order, the price remains elevated at $100.02 because the market believes the stock is undervalued or that demand has increased.

**Total Impact**:
- **Formula:** Total Impact = Temporary Impact + Permanent Impact
- **Execution Price:** The price at which the order is filled reflects both components.
- **Post-Trade:** After the trade, the temporary component reverts, but the permanent component remains.

### 2. Square-Root Law of Market Impact

The **square-root law** is one of the most empirically robust findings in market microstructure. It states that market impact is proportional to the square root of the order size relative to the average daily volume.

**Formula:**
\`\`\`
Impact = γ × σ × sqrt(Q / V)
\`\`\`

Where:
- **Impact:** Price movement as a percentage (or in basis points).
- **γ (gamma):** Market impact coefficient (typically 0.2-0.5 for equities).
- **σ (sigma):** Daily volatility (annualized, e.g., 20% = 0.20).
- **Q:** Order size (number of shares).
- **V:** Average daily trading volume (shares).

**Intuition:**
- **Square-root relationship:** Doubling the order size increases impact by √2 ≈ 1.41 (sub-linear).
- **Volatility:** Higher volatility stocks have higher impact (wider bid-ask spreads, more price sensitivity).
- **Volume:** More liquid stocks (higher V) have lower impact per share traded.

**Example Calculation:**
- Stock: ABC Corp, σ = 25% (0.25), V = 1,000,000 shares/day
- Order: Q = 10,000 shares (1% of daily volume)
- γ = 0.3 (typical for mid-cap equity)
- Impact = 0.3 × 0.25 × sqrt(10,000 / 1,000,000) = 0.3 × 0.25 × sqrt(0.01) = 0.3 × 0.25 × 0.1 = 0.0075 = 0.75% = 75 bps
- If stock price is $100, impact = $0.75 per share.

**Empirical Validation:**
- **Estudies:** Analyzed by Hasbrouck (1991), Almgren & Chriss (2000), and many others.
- **Findings:** Square-root law holds across equities, futures, and even cryptocurrencies.
- **Deviations:** For very large orders (>10% of ADV), impact can be super-linear. For small orders (<0.1% of ADV), impact is nearly negligible.

### 3. Almgren-Chriss Optimal Execution Model

The **Almgren-Chriss model** (2000) provides a rigorous framework for optimal trade execution, balancing **market impact** against **price risk** (volatility during execution).

**Problem Setup:**
- **Goal:** Execute a total order of **X** shares over a time horizon **T**.
- **Trade-off:**
  - **Fast execution:** Minimizes price risk (less exposure to volatility) but increases market impact (aggressive orders).
  - **Slow execution:** Minimizes market impact (spreads order out) but increases price risk (longer exposure to price movements).

**Model Components:**

**a) Temporary Impact Function:**
\`\`\`
h(v) = ε × v
\`\`\`
- **ε (epsilon):** Temporary impact parameter (cost per share traded per unit time).
- **v:** Trading rate (shares per unit time).
- **Interpretation:** Trading faster incurs higher temporary impact.

**b) Permanent Impact Function:**
\`\`\`
g(v) = θ × v
\`\`\`
- **θ (theta):** Permanent impact parameter.
- **Interpretation:** Each share traded moves the price permanently by θ.

**c) Price Dynamics:**
- The price evolves due to:
  1. **Drift:** μ (expected return, often assumed 0 for short-term execution).
  2. **Volatility:** σ (random price movements).
  3. **Permanent impact:** -θ × (cumulative shares traded).

**d) Cost Function:**
The trader seeks to minimize the expected cost, which includes:
- **Market Impact Cost:** Due to temporary and permanent impact.
- **Volatility Cost:** Risk penalty from holding inventory during execution.

**Optimal Strategy:**
Almgren-Chriss derives the optimal trading trajectory:
\`\`\`
n(t) = sinh(κ(T - t)) / sinh(κT)
\`\`\`
Where:
- **n(t):** Fraction of the order remaining at time t.
- **κ (kappa):** Urgency parameter, κ = sqrt(λ / η), where λ is risk aversion and η relates to temporary impact.
- **Behavior:**
  - **High κ (aggressive):** Trade more at the beginning and end (front-loaded and back-loaded).
  - **Low κ (passive):** Trade more uniformly (closer to TWAP).

**Practical Implementation:**
- **Slice the order:** Divide the total order X into N smaller child orders.
- **Schedule:** Release child orders according to the optimal trajectory n(t).
- **Adapt:** Monitor market conditions and adjust if liquidity or volatility changes.

### 4. Implementation Shortfall

**Implementation shortfall** (also called "slippage" or "transaction cost analysis") measures the difference between the portfolio's paper return and the actual return after accounting for execution costs.

**Formula:**
\`\`\`
Implementation Shortfall = (Decision Price - Execution Price) × Shares Traded
\`\`\`

**Components:**1. **Delay Cost (Timing Cost):**
   - Price movement between the decision time (when you decide to trade) and when the order is submitted.
   - Example: Decide to buy at $100, but by the time the order is placed, the price is $100.05.
   - Delay Cost = ($100.05 - $100.00) × 10,000 shares = $500.

2. **Market Impact Cost:**
   - Price movement caused by executing the order.
   - Example: Order execution pushes the average fill price to $100.10.
   - Impact Cost = ($100.10 - $100.05) × 10,000 shares = $500.

3. **Opportunity Cost:**
   - Cost of not executing the entire order (unfilled portion).
   - Example: Only 8,000 shares filled before a sudden price spike to $101.00.
   - Opportunity Cost = ($101.00 - $100.05) × 2,000 unfilled shares = $1,900.

**Total Implementation Shortfall:**
- Sum of all components: Delay + Impact + Opportunity = $500 + $500 + $1,900 = $2,900.
- Per-share: $2,900 / 10,000 = $0.29 per share = 29 bps (if decision price was $100).

**Use in Practice:**
- **Performance attribution:** Compare execution performance across algorithms, venues, or traders.
- **Broker evaluation:** Assess which broker provides the best execution quality.
- **Algorithm tuning:** Optimize execution strategies to minimize shortfall.

---

## Code Implementation: Square-Root Law and Optimal Execution

### Square-Root Law Market Impact Estimator

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SquareRootImpactModel:
    """
    Estimates market impact using the square-root law.
    
    Impact = gamma * sigma * sqrt(Q / V)
    """
    def __init__(self, gamma: float = 0.3):
        """
        Initialize the model.
        
        Parameters:
        - gamma: Market impact coefficient (typically 0.2-0.5 for equities)
        """
        self.gamma = gamma
    
    def estimate_impact(self, order_size: float, daily_volume: float, 
                       volatility: float) -> float:
        """
        Estimate market impact in percentage terms.
        
        Parameters:
        - order_size: Number of shares to trade
        - daily_volume: Average daily volume (shares)
        - volatility: Daily volatility (annualized, e.g., 0.20 for 20%)
        
        Returns:
        - impact: Expected price impact as a percentage
        """
        if daily_volume <= 0:
            raise ValueError("Daily volume must be positive")
        
        participation_rate = order_size / daily_volume
        impact = self.gamma * volatility * np.sqrt(participation_rate)
        return impact
    
    def estimate_impact_bps(self, order_size: float, daily_volume: float, 
                           volatility: float) -> float:
        """Estimate impact in basis points (bps)."""
        return self.estimate_impact(order_size, daily_volume, volatility) * 10000
    
    def estimate_cost(self, order_size: float, daily_volume: float, 
                     volatility: float, price: float) -> float:
        """
        Estimate total market impact cost in dollars.
        
        Parameters:
        - price: Current stock price
        
        Returns:
        - cost: Total cost in dollars
        """
        impact_pct = self.estimate_impact(order_size, daily_volume, volatility)
        cost_per_share = price * impact_pct
        total_cost = cost_per_share * order_size
        return total_cost

# Example usage
model = SquareRootImpactModel(gamma=0.3)

# Stock parameters
price = 100.0
daily_volume = 1_000_000  # 1M shares/day
volatility = 0.25  # 25% annualized

# Order scenarios
order_sizes = [1000, 5000, 10000, 25000, 50000, 100000]

print("Market Impact Analysis (Square-Root Law)")
print("=" * 70)
print(f"Stock Price: \\$\{price:.2f}")
print(f"Daily Volume: {daily_volume:,} shares")
print(f"Volatility: {volatility*100:.1f}%")
print(f"Gamma: {model.gamma}")
print()
print(f"{'Order Size':>12} {'% of ADV':>10} {'Impact (%)':>12} {'Impact (bps)':>14} {'Total Cost ($)':>15}")
print("-" * 70)

for size in order_sizes:
    pct_adv = (size / daily_volume) * 100
impact_pct = model.estimate_impact(size, daily_volume, volatility)
impact_bps = model.estimate_impact_bps(size, daily_volume, volatility)
total_cost = model.estimate_cost(size, daily_volume, volatility, price)

print(f"{size:>12,} {pct_adv:>9.2f}% {impact_pct*100:>11.3f}% {impact_bps:>13.1f} {total_cost:>14,.2f}")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 5))

# Impact vs Order Size
sizes_range = np.linspace(1000, 100000, 100)
impacts = [model.estimate_impact(s, daily_volume, volatility) * 100 for s in sizes_range]
ax1.plot(sizes_range / 1000, impacts, linewidth = 2)
ax1.set_xlabel('Order Size (thousands of shares)')
ax1.set_ylabel('Market Impact (%)')
ax1.set_title('Market Impact vs Order Size (Square-Root Law)')
ax1.grid(True, alpha = 0.3)

# Total Cost vs Order Size
costs = [model.estimate_cost(s, daily_volume, volatility, price) for s in sizes_range]
ax2.plot(sizes_range / 1000, np.array(costs) / 1000, linewidth = 2, color = 'red')
ax2.set_xlabel('Order Size (thousands of shares)')
ax2.set_ylabel('Total Market Impact Cost ($1000s)')
ax2.set_title('Total Cost vs Order Size')
ax2.grid(True, alpha = 0.3)

plt.tight_layout()
plt.show()
\`\`\`

### Almgren-Chriss Optimal Execution

\`\`\`python
class AlmgrenChrissExecutor:
    """
    Implements Almgren-Chriss optimal execution strategy.
    """
    def __init__(self, total_shares: float, total_time: float, 
                 epsilon: float, theta: float, sigma: float, 
                 risk_aversion: float = 1e-6):
        """
        Initialize the executor.
        
        Parameters:
        - total_shares: Total order size
        - total_time: Execution time horizon (e.g., in minutes)
        - epsilon: Temporary impact parameter
        - theta: Permanent impact parameter
        - sigma: Price volatility (per sqrt(time))
        - risk_aversion: Risk aversion parameter (lambda)
        """
        self.X = total_shares
        self.T = total_time
        self.epsilon = epsilon
        self.theta = theta
        self.sigma = sigma
        self.lam = risk_aversion
        
        # Calculate urgency parameter kappa
        self.eta = epsilon  # Simplification: eta = epsilon
        self.kappa = np.sqrt(self.lam * self.sigma**2 / self.eta)
    
    def optimal_trajectory(self, t: float) -> float:
        """
        Calculate the optimal number of shares remaining at time t.
        
        Parameters:
        - t: Current time (0 <= t <= T)
        
        Returns:
        - shares: Number of shares remaining to be traded
        """
        if t >= self.T:
            return 0.0
        
        # Almgren-Chriss formula
        shares_remaining = self.X * np.sinh(self.kappa * (self.T - t)) / np.sinh(self.kappa * self.T)
        return shares_remaining
    
    def generate_schedule(self, num_slices: int) -> pd.DataFrame:
        """
        Generate a trading schedule with N slices.
        
        Parameters:
        - num_slices: Number of child orders to split the parent into
        
        Returns:
        - schedule: DataFrame with columns [time, shares_remaining, shares_to_trade]
        """
        times = np.linspace(0, self.T, num_slices + 1)
        schedule = []
        
        for i in range(len(times)):
            t = times[i]
            shares_remaining = self.optimal_trajectory(t)
            
            if i > 0:
                shares_to_trade = schedule[-1]['shares_remaining'] - shares_remaining
            else:
                shares_to_trade = 0
            
            schedule.append({
                'time': t,
                'shares_remaining': shares_remaining,
                'shares_to_trade': shares_to_trade
            })
        
        return pd.DataFrame(schedule)
    
    def compare_strategies(self, num_slices: int = 10):
        """
        Compare optimal strategy against TWAP and aggressive execution.
        """
        times = np.linspace(0, self.T, 100)
        
        # Optimal (Almgren-Chriss)
        optimal = [self.optimal_trajectory(t) for t in times]
        
        # TWAP (linear)
        twap = [self.X * (1 - t / self.T) for t in times]
        
        # Aggressive (front-loaded exponential decay)
        aggressive = [self.X * np.exp(-5 * t / self.T) for t in times]
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(times, optimal, label='Optimal (Almgren-Chriss)', linewidth=2)
        plt.plot(times, twap, label='TWAP (Uniform)', linestyle='--', linewidth=2)
        plt.plot(times, aggressive, label='Aggressive (Front-loaded)', linestyle=':', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Shares Remaining')
        plt.title('Execution Strategy Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# Example: Execute 100,000 shares over 60 minutes
executor = AlmgrenChrissExecutor(
    total_shares=100000,
    total_time=60,  # minutes
    epsilon=0.0001,  # temporary impact
    theta=0.00005,   # permanent impact
    sigma=0.02,      # volatility per sqrt(minute)
    risk_aversion=1e-6
)

# Generate trading schedule
schedule = executor.generate_schedule(num_slices=12)
print("\\nAlmgren-Chriss Optimal Execution Schedule")
print("=" * 60)
print(schedule.to_string(index=False))

# Compare strategies
executor.compare_strategies()
\`\`\`

### Implementation Shortfall Calculator

\`\`\`python
class ImplementationShortfallCalculator:
    """
    Calculates implementation shortfall for trade execution.
    """
    def __init__(self, decision_price: float, decision_time: float):
        """
        Initialize with decision price and time.
        
        Parameters:
        - decision_price: Price at the time of decision
        - decision_time: Timestamp of decision (for logging)
        """
        self.decision_price = decision_price
        self.decision_time = decision_time
        self.executions = []
    
    def add_execution(self, shares: float, price: float, timestamp: float):
        """Record an execution."""
        self.executions.append({
            'shares': shares,
            'price': price,
            'timestamp': timestamp
        })
    
    def calculate_shortfall(self, final_price: float = None) -> dict:
        """
        Calculate implementation shortfall components.
        
        Parameters:
        - final_price: Market price at end of execution period (for opportunity cost)
        
        Returns:
        - metrics: Dictionary with shortfall components
        """
        if not self.executions:
            return {'error': 'No executions recorded'}
        
        total_shares_executed = sum(e['shares'] for e in self.executions)
        
        # Calculate VWAP (Volume-Weighted Average Price)
        vwap = sum(e['shares'] * e['price'] for e in self.executions) / total_shares_executed
        
        # Market impact cost (VWAP - decision price)
        market_impact = (vwap - self.decision_price) * total_shares_executed
        
        # Delay cost (first execution price - decision price)
        first_exec_price = self.executions[0]['price']
        delay_cost = (first_exec_price - self.decision_price) * total_shares_executed
        
        # Trading cost (VWAP - first execution price)
        trading_cost = (vwap - first_exec_price) * total_shares_executed
        
        metrics = {
            'decision_price': self.decision_price,
            'vwap': vwap,
            'first_exec_price': first_exec_price,
            'total_shares': total_shares_executed,
            'market_impact_$': market_impact,
            'market_impact_bps': (market_impact / (self.decision_price * total_shares_executed)) * 10000,
            'delay_cost_$': delay_cost,
            'trading_cost_$': trading_cost,
        }
        
        return metrics

# Example usage
calc = ImplementationShortfallCalculator(decision_price=100.00, decision_time=0)

# Simulate executions
calc.add_execution(shares=2000, price=100.02, timestamp=1)
calc.add_execution(shares=3000, price=100.05, timestamp=2)
calc.add_execution(shares=2500, price=100.08, timestamp=3)
calc.add_execution(shares=2000, price=100.06, timestamp=4)
calc.add_execution(shares=500, price=100.04, timestamp=5)

# Calculate shortfall
metrics = calc.calculate_shortfall()
print("\\nImplementation Shortfall Analysis")
print("=" * 60)
for key, value in metrics.items():
    if isinstance(value, float):
        print(f"{key:>25}: {value:>12.4f}")
    else:
        print(f"{key:>25}: {value:>12}")
\`\`\`

---

## Real-World Example: Institutional Execution at BlackRock

BlackRock, the world's largest asset manager, executes billions of dollars in trades daily. Their Aladdin platform incorporates sophisticated market impact models to optimize execution.

**Scenario:** A portfolio manager decides to buy $50 million of Apple stock (AAPL).

**Execution Strategy:**1. **Pre-Trade Analysis:**
   - Estimate market impact using square-root law: ~15-20 bps for a $50M order.
   - Decide on execution horizon: 2 hours (balance impact vs. price risk).
   - Choose algorithm: Almgren-Chriss optimal execution (adaptive).

2. **Smart Order Routing:**
   - Split order across multiple venues: NASDAQ, NYSE Arca, IEX, dark pools.
   - Use VWAP algorithm as benchmark.
   - Monitor real-time liquidity and adjust pacing.

3. **During Execution:**
   - Continuous monitoring of order book depth and market conditions.
   - If volatility spikes, slow down to avoid higher impact.
   - If liquidity improves, accelerate to reduce price risk.

4. **Post-Trade Analysis:**
   - Calculate implementation shortfall: Actual VWAP vs. decision price.
   - Compare to benchmark: VWAP over the day.
   - Attribution: Break down costs into delay, impact, opportunity.

**Results:**
- Decision price: $150.00
- VWAP achieved: $150.12 (12 bps slippage)
- Expected impact (model): 15 bps
- Actual cost: Better than expected (good execution)

**Key Takeaways:**
- Large institutions use quantitative models to guide execution.
- Real-time adaptation is crucial (markets are dynamic).
- Post-trade analysis provides feedback for continuous improvement.

---

## Hands-on Exercise: Build a VWAP Algorithm

**Task:** Implement a simple Volume-Weighted Average Price (VWAP) execution algorithm.

**Requirements:**1. Accept total order size and execution time horizon.
2. Use historical volume profile to schedule child orders.
3. Aim to match the volume profile (trade more when market is more active).
4. Calculate expected market impact using square-root law.

**Historical Volume Profile (example):**
- 9:30-10:00: 15% of daily volume
- 10:00-11:00: 20%
- 11:00-12:00: 15%
- 12:00-13:00: 10%
- 13:00-14:00: 12%
- 14:00-15:00: 15%
- 15:00-16:00: 13%

**Hints:**
- Slice the total order proportionally to the volume profile.
- For each slice, estimate market impact given the expected volume in that period.
- Compare your VWAP strategy to a simple TWAP (uniform distribution).

\`\`\`python
# Your implementation here
class VWAPAlgorithm:
    def __init__(self, total_shares, volume_profile):
        # volume_profile: dict with time_period -> % of daily volume
        pass
    
    def generate_schedule(self):
        # Return DataFrame with time periods and shares to trade
        pass
    
    def estimate_total_impact(self, daily_volume, volatility):
        # Use square-root law for each slice
        pass
\`\`\`

---

## Common Pitfalls

1. **Ignoring Market Impact in Backtests:** Assuming all orders fill at the midpoint without considering impact. This leads to over-optimistic backtesting results, especially for larger strategies.

2. **Linear Extrapolation:** Assuming impact scales linearly with order size (e.g., 2× size = 2× impact). In reality, the square-root law is more accurate (2× size = 1.41× impact).

3. **Static Execution:** Using a fixed algorithm (e.g., TWAP) without adapting to changing market conditions (volatility spikes, liquidity droughts).

4. **Ignoring Permanent Impact:** Focusing only on temporary impact and not accounting for the information signal your order sends to the market.

5. **Over-Optimization:** Tuning execution algorithms on in-sample data, leading to poor out-of-sample performance when market conditions change.

6. **Neglecting Opportunity Cost:** Not accounting for the cost of unfilled orders when execution is too passive.

---

## Production Checklist

1. **Real-Time Market Data:** Integrate Level 2 order book data to assess current liquidity before placing child orders.

2. **Dynamic Impact Models:** Update market impact parameters (γ, σ, ε, θ) based on recent trading activity and current market conditions.

3. **Smart Order Routing:** Implement logic to route orders to the best venue (exchange, dark pool, internalized) based on liquidity, fees, and fill probability.

4. **Adaptive Algorithms:** Build feedback loops to adjust pacing in real-time. If impact is higher than expected, slow down; if lower, speed up.

5. **Risk Controls:** Set maximum participation rate (e.g., never exceed 20% of volume in any 5-minute interval) to avoid excessive impact.

6. **Post-Trade Analytics:** Log all executions with timestamps, venues, and prices. Calculate implementation shortfall for every parent order to evaluate performance.

7. **Latency Management:** Ensure execution algorithms can process market data and submit orders with minimal latency (< 100 microseconds for HFT, < 1 millisecond for institutional).

8. **Compliance Monitoring:** Ensure execution strategies comply with regulations (e.g., Reg NMS best execution, MiFID II).

---

## Regulatory Considerations

1. **Best Execution (Reg NMS, MiFID II):** Brokers and traders are legally required to seek the best execution for client orders, considering price, speed, and likelihood of execution. Market impact models help demonstrate compliance.

2. **Algorithmic Trading Disclosure (MiFID II):** Firms using algorithmic execution strategies must disclose their methodologies to clients and regulators.

3. **Market Manipulation:** Aggressive execution that creates artificial volatility or attempts to "paint the tape" can be considered manipulative. Ensure algorithms don't inadvertently trigger manipulative patterns.

4. **Transaction Cost Analysis (TCA):** Many institutional investors require brokers to provide TCA reports showing implementation shortfall and execution quality. This is increasingly a regulatory expectation.

5. **Circuit Breakers:** Be aware of exchange rules that halt trading during extreme volatility. Execution algorithms must handle halts gracefully and not exacerbate volatility.

---

## Further Reading

1. **Almgren, R., & Chriss, N. (2000).** "Optimal execution of portfolio transactions." *Journal of Risk*, 3, 5-40.
   - Foundational paper on optimal execution theory.

2. **Kissell, R. (2013).** *The Science of Algorithmic Trading and Portfolio Management.*
   - Comprehensive practitioner guide to execution algorithms.

3. **Hasbrouck, J. (2007).** *Empirical Market Microstructure.*
   - Detailed treatment of market impact estimation.

4. **Grinold, R., & Kahn, R. (1999).** *Active Portfolio Management.*
   - Chapter on transaction costs and implementation shortfall.
`,
};
