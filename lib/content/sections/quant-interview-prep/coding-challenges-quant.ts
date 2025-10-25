export const codingChallenges = {
  title: 'Coding Challenges (Quant-Focused)',
  id: 'coding-challenges-quant',
  content: `
# Coding Challenges (Quant-Focused)

## Introduction

Coding interviews at quantitative trading firms (Citadel, Jane Street, Two Sigma, Jump Trading, HRT, SIG, DRW, etc.) test different skills than typical software engineering interviews. While FAANG interviews focus on general algorithms and data structures, quant interviews emphasize:

**Core Focus Areas:**
1. **Financial algorithms** - Option pricers, portfolio optimization, backtesting
2. **Numerical computation** - Precision, stability, efficiency
3. **Data structures for trading** - Order books, tick data, position tracking
4. **Mathematical correctness** - Exact solutions, edge cases, numerical issues
5. **Clean, readable code** - Production-ready, testable, maintainable
6. **Time/space complexity** - Efficient solutions for real-time trading

**Key Differences from FAANG:**
- More emphasis on mathematical correctness
- Financial domain knowledge expected
- Numerical precision matters (floating point issues)
- Real-world constraints (latency, memory)
- Often require statistics/probability knowledge

**Typical Interview Structure:**
- 2-3 coding problems in 60-90 minutes
- 10-15 minutes per problem for simpler ones
- 30-45 minutes for complex problems
- Live coding with interviewer watching
- Expected to test your own code

This section covers 15+ problems across all difficulty levels, organized by category, with complete solutions, test cases, and complexity analysis.

---

## Category 1: Option Pricing & Greeks

### Problem 1.1: Black-Scholes Call Option Pricer (Medium, 30 min)

**Task:** Implement Black-Scholes formula for European call options. Must handle edge cases and be numerically stable.

**Requirements:**
- Accurate to within 0.01
- Handle edge cases (T=0, sigma=0, S=0)
- Clean, documented code
- Test cases

\`\`\`python
"""
Black-Scholes Option Pricer
Difficulty: Medium
Time: 30 minutes
"""

import math
from typing import Optional

def norm_cdf(x: float) -> float:
    """Cumulative distribution function for standard normal."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> Optional[float]:
    """
    Black-Scholes call option price.
    
    Args:
        S: Current stock price (must be > 0)
        K: Strike price (must be > 0)
        T: Time to expiration in years (must be >= 0)
        r: Risk-free interest rate
        sigma: Volatility (must be >= 0)
        
    Returns:
        Call option price, or None if inputs invalid
    """
    # Input validation
    if S <= 0 or K <= 0 or T < 0 or sigma < 0:
        return None
    
    # Edge case: At expiration
    if T == 0:
        return max(S - K, 0)
    
    # Edge case: Zero volatility
    if sigma == 0:
        future_value = S * math.exp(r * T)
        return max(future_value - K, 0) * math.exp(-r * T)
    
    # Calculate d1 and d2
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    # Calculate option price
    call_price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    
    return call_price

def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> Optional[float]:
    """Black-Scholes put option price using put-call parity."""
    call_price = black_scholes_call(S, K, T, r, sigma)
    if call_price is None:
        return None
    return call_price - S + K * math.exp(-r * T)

# Test cases
def test_black_scholes():
    # ATM option
    price = black_scholes_call(100, 100, 1.0, 0.05, 0.2)
    assert abs(price - 10.45) < 0.1
    
    # OTM option
    price = black_scholes_call(100, 110, 1.0, 0.05, 0.2)
    assert abs(price - 5.79) < 0.1
    
    # At expiration ITM
    price = black_scholes_call(110, 100, 0, 0.05, 0.2)
    assert abs(price - 10) < 0.01
    
    # Put-call parity
    call = black_scholes_call(100, 100, 1.0, 0.05, 0.2)
    put = black_scholes_put(100, 100, 1.0, 0.05, 0.2)
    parity = call - put - (100 - 100 * math.exp(-0.05))
    assert abs(parity) < 0.01
    
    print("✓ All tests passed!")

test_black_scholes()
\`\`\`

**Time Complexity:** O(1)
**Space Complexity:** O(1)

---

### Problem 1.2: Implied Volatility Calculator (Hard, 45 min)

**Task:** Find implied volatility using Newton-Raphson method.

\`\`\`python
"""
Implied Volatility Calculator
Difficulty: Hard
Time: 45 minutes
"""

def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = 'call',
    tolerance: float = 1e-6,
    max_iter: int = 100
) -> Optional[float]:
    """Calculate implied volatility using Newton-Raphson."""
    
    sigma = 0.2  # Initial guess
    
    for i in range(max_iter):
        # Calculate price and vega at current sigma
        if option_type == 'call':
            price = black_scholes_call(S, K, T, r, sigma)
        else:
            price = black_scholes_put(S, K, T, r, sigma)
        
        if price is None:
            return None
        
        diff = price - market_price
        
        if abs(diff) < tolerance:
            return sigma
        
        # Calculate vega
        d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        vega = S * math.exp(-0.5*d1**2) / math.sqrt(2*math.pi) * math.sqrt(T)
        
        if vega < 1e-10:
            return None
        
        # Newton-Raphson update
        sigma = sigma - diff / vega
        sigma = max(sigma, 0.001)  # Keep positive
    
    return None

# Test
true_sigma = 0.25
market_price = black_scholes_call(100, 100, 1.0, 0.05, true_sigma)
implied = implied_volatility(market_price, 100, 100, 1.0, 0.05, 'call')
assert abs(implied - true_sigma) < 1e-4
print("✓ Implied volatility test passed!")
\`\`\`

---

## Category 2: Order Books & Market Microstructure

### Problem 2.1: Limit Order Book (Hard, 45 min)

**Task:** Implement efficient order book with O(log n) operations.

\`\`\`python
"""
Limit Order Book Implementation
Difficulty: Hard
Time: 45 minutes
"""

import heapq
from collections import defaultdict
from typing import Dict, List, Optional

class OrderBook:
    """High-performance limit order book using heaps."""
    
    def __init__(self):
        self.bids = []  # Max-heap: (-price, timestamp, order_id, quantity)
        self.asks = []  # Min-heap: (price, timestamp, order_id, quantity)
        self.orders = {}  # order_id -> (price, quantity, side)
        self.timestamp = 0
    
    def add_order(self, order_id: int, side: str, price: float, quantity: int) -> bool:
        """Add limit order. Time: O(log n)"""
        if order_id in self.orders or quantity <= 0:
            return False
        
        self.timestamp += 1
        self.orders[order_id] = (price, quantity, side)
        
        if side == 'BUY':
            heapq.heappush(self.bids, (-price, self.timestamp, order_id, quantity))
        else:
            heapq.heappush(self.asks, (price, self.timestamp, order_id, quantity))
        
        return True
    
    def cancel_order(self, order_id: int) -> bool:
        """Cancel order. Time: O(1) amortized"""
        if order_id not in self.orders:
            return False
        del self.orders[order_id]
        return True
    
    def _clean_heap(self, heap: List):
        """Remove canceled orders from heap top."""
        while heap and heap[0][2] not in self.orders:
            heapq.heappop(heap)
    
    def get_best_bid(self) -> Optional[float]:
        """Get best bid price. Time: O(log n) worst case"""
        self._clean_heap(self.bids)
        return -self.bids[0][0] if self.bids else None
    
    def get_best_ask(self) -> Optional[float]:
        """Get best ask price."""
        self._clean_heap(self.asks)
        return self.asks[0][0] if self.asks else None
    
    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        bid, ask = self.get_best_bid(), self.get_best_ask()
        return (ask - bid) if (bid and ask) else None

# Test
book = OrderBook()
book.add_order(1, 'BUY', 100.0, 10)
book.add_order(2, 'SELL', 101.0, 5)
assert book.get_best_bid() == 100.0
assert book.get_best_ask() == 101.0
assert abs(book.get_spread() - 1.0) < 0.01
print("✓ Order book tests passed!")
\`\`\`

---

## Category 3: Backtesting & Strategy Simulation

### Problem 3.1: Moving Average Crossover (Medium, 30 min)

**Task:** Backtest MA crossover strategy with transaction costs.

\`\`\`python
"""
Moving Average Crossover Backtest
Difficulty: Medium
Time: 30 minutes
"""

import numpy as np
import pandas as pd

def backtest_ma_crossover(
    prices: np.ndarray,
    fast_period: int = 20,
    slow_period: int = 50,
    transaction_cost: float = 0.001
) -> dict:
    """
    Backtest moving average crossover strategy.
    
    Returns:
        Dictionary with performance metrics
    """
    df = pd.DataFrame({'price': prices})
    
    # Calculate MAs
    df['fast_ma'] = df['price'].rolling(fast_period).mean()
    df['slow_ma'] = df['price'].rolling(slow_period).mean()
    
    # Generate signals
    df['signal'] = (df['fast_ma'] > df['slow_ma']).astype(int)
    
    # Calculate returns
    df['returns'] = df['price'].pct_change()
    df['strategy_returns'] = df['signal'].shift(1) * df['returns']
    
    # Apply transaction costs
    df['position_change'] = df['signal'].diff().abs()
    df['costs'] = df['position_change'] * transaction_cost
    df['net_returns'] = df['strategy_returns'] - df['costs']
    
    # Performance metrics
    total_return = (1 + df['net_returns'].dropna()).prod() - 1
    sharpe = df['net_returns'].mean() / df['net_returns'].std() * np.sqrt(252)
    num_trades = df['position_change'].sum() / 2
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'num_trades': int(num_trades)
    }

# Test
np.random.seed(42)
prices = 100 * (1 + np.random.randn(500) * 0.02).cumprod()
results = backtest_ma_crossover(prices)
print(f"Total return: {results['total_return']:.2%}")
print(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
print(f"Trades: {results['num_trades']}")
\`\`\`

---

## Category 4: Portfolio Optimization

### Problem 4.1: Mean-Variance Optimization (Medium, 30 min)

**Task:** Find optimal portfolio weights given returns and covariance.

\`\`\`python
"""
Portfolio Optimization
Difficulty: Medium
Time: 30 minutes
"""

import numpy as np
from scipy.optimize import minimize

def optimize_portfolio(returns: np.ndarray, cov_matrix: np.ndarray, target_return: float = None) -> dict:
    """
    Find minimum variance portfolio.
    
    Args:
        returns: Expected returns
        cov_matrix: Covariance matrix
        target_return: Optional target return constraint
        
    Returns:
        Dictionary with optimal weights and stats
    """
    n = len(returns)
    
    # Objective: minimize variance
    def objective(weights):
        return weights @ cov_matrix @ weights
    
    # Constraints: weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    
    if target_return is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda w: w @ returns - target_return
        })
    
    # Bounds: long only
    bounds = tuple((0, 1) for _ in range(n))
    
    # Initial guess
    w0 = np.ones(n) / n
    
    # Optimize
    result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    weights = result.x
    port_return = weights @ returns
    port_vol = np.sqrt(weights @ cov_matrix @ weights)
    
    return {
        'weights': weights,
        'return': port_return,
        'volatility': port_vol,
        'sharpe': port_return / port_vol if port_vol > 0 else 0
    }

# Test
returns = np.array([0.10, 0.15, 0.12])
cov = np.array([[0.04, 0.01, 0.02],
                [0.01, 0.09, 0.03],
                [0.02, 0.03, 0.05]])

result = optimize_portfolio(returns, cov)
print(f"Optimal weights: {result['weights']}")
print(f"Expected return: {result['return']:.2%}")
print(f"Volatility: {result['volatility']:.2%}")
\`\`\`

---

## Category 5: Algorithmic Problems with Financial Flavor

### Problem 5.1: Maximum Profit with K Transactions (Hard, 30 min)

**Task:** Given stock prices, find maximum profit with at most k transactions.

\`\`\`python
"""
Maximum Profit with K Transactions
Difficulty: Hard
Time: 30 minutes
"""

def max_profit_k_transactions(prices: list, k: int) -> int:
    """
    Find maximum profit with at most k buy-sell transactions.
    
    DP approach:
    buy[i][j] = max profit after at most i transactions with stock in hand on day j
    sell[i][j] = max profit after at most i transactions without stock on day j
    
    Time: O(n*k)
    Space: O(k)
    """
    if not prices or k == 0:
        return 0
    
    n = len(prices)
    
    # Optimization: if k >= n/2, can do unlimited transactions
    if k >= n // 2:
        return sum(max(prices[i] - prices[i-1], 0) for i in range(1, n))
    
    # DP arrays
    buy = [-float('inf')] * (k + 1)
    sell = [0] * (k + 1)
    
    for price in prices:
        for j in range(k, 0, -1):
            sell[j] = max(sell[j], buy[j] + price)
            buy[j] = max(buy[j], sell[j-1] - price)
    
    return sell[k]

# Test cases
assert max_profit_k_transactions([3,2,6,5,0,3], 2) == 7  # Buy at 2, sell at 6, buy at 0, sell at 3
assert max_profit_k_transactions([2,4,1], 2) == 2  # Buy at 2, sell at 4
assert max_profit_k_transactions([3,3,5,0,0,3,1,4], 2) == 6  # Buy at 0, sell at 3, buy at 1, sell at 4
print("✓ All max profit tests passed!")
\`\`\`

---

### Problem 5.2: Calendar Spread Arbitrage Detection (Medium, 25 min)

**Task:** Detect arbitrage opportunities in calendar spreads.

\`\`\`python
"""
Calendar Spread Arbitrage Detection
Difficulty: Medium
Time: 25 minutes
"""

def detect_calendar_arbitrage(
    short_term_call: float,
    long_term_call: float,
    S: float,
    K: float,
    T1: float,
    T2: float,
    r: float
) -> dict:
    """
    Detect arbitrage in calendar spread.
    
    Calendar spread: Buy long-term call, sell short-term call (same strike).
    
    No-arbitrage bounds:
    - Calendar spread cost should be >= 0
    - Calendar spread cost should be <= K*(e^(-rT1) - e^(-rT2))
    
    Returns:
        Dictionary with arbitrage info
    """
    spread_cost = long_term_call - short_term_call
    
    # Lower bound: should be non-negative (time value)
    lower_bound = 0
    
    # Upper bound: from put-call parity
    upper_bound = K * (math.exp(-r * T1) - math.exp(-r * T2))
    
    arbitrage_detected = False
    arbitrage_type = None
    profit_potential = 0
    
    if spread_cost < lower_bound - 1e-6:
        # Negative spread - buy the spread
        arbitrage_detected = True
        arbitrage_type = 'Buy calendar spread (negative cost)'
        profit_potential = -spread_cost
    
    if spread_cost > upper_bound + 1e-6:
        # Overpriced spread - sell the spread
        arbitrage_detected = True
        arbitrage_type = 'Sell calendar spread (overpriced)'
        profit_potential = spread_cost - upper_bound
    
    return {
        'arbitrage_detected': arbitrage_detected,
        'arbitrage_type': arbitrage_type,
        'spread_cost': spread_cost,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'profit_potential': profit_potential
    }

# Test
result = detect_calendar_arbitrage(
    short_term_call=5.0,
    long_term_call=8.0,
    S=100,
    K=100,
    T1=0.25,
    T2=0.5,
    r=0.05
)
print(f"Arbitrage detected: {result['arbitrage_detected']}")
print(f"Spread cost: \${result['spread_cost']:.2f}")
\`\`\`

---

## Common Patterns & Data Structures

### 1. Binary Search
- Finding strikes in sorted arrays
- Bisection for root finding (implied vol)

### 2. Dynamic Programming
- Optimal execution problems
- Transaction limit problems
- Path-dependent options

### 3. Heaps/Priority Queues
- Order books (best bid/ask)
- Event-driven simulations
- Top-K problems

### 4. Hash Maps
- Order ID lookup
- Position tracking
- Price level aggregation

### 5. Sliding Windows
- Moving averages
- Rolling statistics
- Time-series analysis

---

## Interview Tips & Strategy

### Time Management (45-minute problem)
- **0-5 min:** Clarify requirements, discuss approach
- **5-35 min:** Code solution incrementally
- **35-40 min:** Test thoroughly
- **40-45 min:** Discuss complexity, optimizations, edge cases

### Code Quality Checklist
- ✓ Descriptive variable names (strike_price not k)
- ✓ Type hints (when using Python 3.6+)
- ✓ Docstrings for functions
- ✓ Input validation
- ✓ Edge case handling
- ✓ Test cases with assertions
- ✓ Comments for complex logic

### Communication Tips
1. **Think out loud** - Explain your reasoning
2. **Ask questions** - Clarify ambiguities
3. **Discuss tradeoffs** - Time vs space, accuracy vs speed
4. **Mention alternatives** - "I'm using X because Y, but Z would also work"
5. **Test incrementally** - Don't wait until the end

### Common Pitfalls to Avoid
1. **Jumping into coding** without clarifying requirements
2. **Ignoring edge cases** (T=0, sigma=0, division by zero)
3. **Poor variable naming** (single letters for complex concepts)
4. **No test cases** (always include them!)
5. **Forgetting complexity analysis** (state time/space)
6. **Numerical issues** (floating point precision, overflow)

### Complexity Analysis
Always state:
- **Time complexity:** O(?) - Be specific about variables
- **Space complexity:** O(?) - Consider auxiliary space
- **Tradeoffs:** "Could optimize X at cost of Y"

---

## Practice Plan

### Week 1: Fundamentals
- Black-Scholes (variants)
- Basic order book
- Simple backtests

### Week 2: Advanced
- Implied volatility
- Full order book with matching
- Portfolio optimization

### Week 3: Speed
- Practice under time pressure
- Mock interviews
- Code review

### Resources
- LeetCode (finance-tagged problems)
- QuantStart articles
- "Heard on the Street" book
- Project Euler (mathematical problems)

**Remember:** Quant firms care more about correctness and clarity than perfect optimization. Get it working, then improve it!

Good luck with your interviews!
`,
};
