export const codingChallengesQuiz = [
  {
    id: 'ccq-q-1',
    question:
      'Jane Street: "Implement a function to compute implied volatility given option price using Newton-Raphson method. Include: (1) complete implementation with convergence criteria, (2) initial guess strategy, (3) handling edge cases (extreme prices, near expiration). Time limit: 30 minutes. Then explain why Newton-Raphson is preferred over binary search here."',
    sampleAnswer:
      "Implied volatility Newton-Raphson: (1) Implementation: def implied_vol_newton(S, K, T, r, option_price, is_call=True): sigma = 0.3  # initial guess, max_iter = 100, tol = 1e-6. Loop: compute BS price at sigma, compute vega = S × N'(d1) × √T, error = BS_price - option_price, if |error| < tol: return sigma, sigma -= error / vega (Newton step). (2) Initial guess: Use ATM approximation sigma ≈ option_price / (0.4 × S × √T) for better starting point. (3) Edge cases: If T < 0.001, return large sigma; if price < intrinsic, return None (arbitrage); if vega near 0, use binary search fallback. (4) Newton-Raphson vs binary search: Newton converges quadratically (doubles digits per iteration) vs linear for binary search. Typical: Newton needs 3-5 iterations, binary search needs 20-30. Vega (derivative) is easy to compute, making Newton natural choice. However, binary search more robust for extreme cases where vega vanishes.",
    keyPoints: [
      'Newton step: σ_new = σ_old - (BS_price - market_price) / vega',
      'Initial guess: σ ≈ price / (0.4 × S × √T) from ATM formula',
      'Convergence: typically 3-5 iterations with tol=1e-6',
      'Edge cases: T near 0, price < intrinsic, vega near 0',
      'Newton quadratic convergence vs binary search linear convergence',
    ],
  },
  {
    id: 'ccq-q-2',
    question:
      'Citadel: "Design and implement an order matching engine. Requirements: (1) support limit orders (buy/sell), (2) price-time priority, (3) O(log n) add, O(log n) match operations, (4) handle order cancellations efficiently. Explain data structure choices and trade-offs. Time: 45 minutes."',
    sampleAnswer:
      'Order matching engine design: (1) Data structures: Use two heaps (priority queues) for bids (max-heap by price, then time) and asks (min-heap by price, then time). Use hash map for O(1) order lookup by ID. (2) Add order: If limit buy at price p, add to bids heap with (-p, timestamp, order_id). Check if crosses spread (best_ask ≤ p), if yes, match immediately. O(log n) heap insert. (3) Match: While bids non-empty and asks non-empty and best_bid ≥ best_ask: pop both, match quantity (partial fill if needed), emit trade. Continue until no more matches. Each match is O(log n) due to heap operations. (4) Cancel: Mark order as canceled in hash map (O(1)). Lazy deletion: remove from heap when popped to top. Alternative: use balanced BST (TreeMap) for O(log n) true deletion but more complex. (5) Trade-offs: Heap + lazy deletion: Simple, fast average case, but heap may grow with canceled orders. BST: True O(log n) deletion, but more code complexity, slightly slower constants. For high-cancel scenarios (many order modifications), BST better. For typical trading (fewer cancels), heap sufficient.',
    keyPoints: [
      'Heaps: max-heap for bids, min-heap for asks, O(log n) operations',
      'Hash map: O(1) order lookup for cancellations',
      'Lazy deletion: mark canceled in map, remove when popped from heap',
      'Price-time priority: heap sorted by (-price, timestamp) for bids',
      'Trade-off: heap simplicity vs BST true deletion efficiency',
    ],
  },
  {
    id: 'ccq-q-3',
    question:
      'Two Sigma: "Write a function to detect arbitrage in a graph of currency exchange rates. Given n currencies and exchange rates between pairs, find if arbitrage cycle exists and return the cycle. Explain algorithm, complexity, and how to handle floating point precision issues."',
    sampleAnswer:
      'Currency arbitrage detection: (1) Graph modeling: Vertices = currencies, edges = exchange rates. Weight of edge from currency i to j is log (rate_ij). Arbitrage exists if negative cycle exists (because log (a×b×c) = log (a)+log (b)+log (c), and profitable cycle has product > 1, so sum of logs > 0, negated < 0). (2) Algorithm: Use Bellman-Ford to detect negative cycles. Initialize dist[start] = 0, others = inf. Relax edges n-1 times: for edge (u,v) with weight w, if dist[u] + w < dist[v], update dist[v]. Check: if any edge can be relaxed further, negative cycle exists. To find cycle: track predecessors, do one more relaxation, backtrack from updated vertex. (3) Complexity: O(V×E) = O(n³) worst case if complete graph. For sparse graphs (few currency pairs), O(n×m) where m = number of pairs. (4) Floating point: Use epsilon tolerance for comparisons (if dist[u] + w < dist[v] - eps). Alternatively, work with integer arithmetic by multiplying rates by large constant and using logs scaled to integers. Check cycle product: if within 1-eps of 1.0, not true arbitrage (due to precision/transaction costs). (5) Return cycle: trace predecessors from detected vertex to find cycle vertices, then extract exchange rates along cycle.',
    keyPoints: [
      'Model as graph: vertices=currencies, edge weights=log (exchange_rate)',
      'Arbitrage ⟺ negative cycle (product > 1 ⟹ sum of logs > 0)',
      'Bellman-Ford: O(V×E) to detect negative cycles',
      'Handle precision: epsilon tolerance for comparisons, scale rates',
      'Return cycle: backtrack predecessors from updated vertex',
    ],
  },
];
