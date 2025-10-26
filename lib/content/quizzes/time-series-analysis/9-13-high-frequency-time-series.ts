export const highFrequencyTimeSeriesQuiz = [
  {
    id: 1,
    question:
      "Design a real-time market-making algorithm using high-frequency order book data (bid/ask prices, depths, trade flow). Address: optimal bid-ask spread placement, inventory risk management, adverse selection detection, latency requirements, and how to detect when to shut down (toxic flow periods).",
    answer: `[Framework: Place bids/asks around mid-price ± spread/2; Spread = fixed_component + inventory_penalty × |position| + volatility_component; Inventory management: target neutral, widen quotes when far from target; Adverse selection: detect via order flow imbalance (OFI), widen or pull quotes if OFI extreme; Latency: need <1ms response for competitive; Shutdown criteria: realized spread negative for 15min, OFI > 3σ consistently, volatility > 2× daily average; Profit from bid-ask spread minus adverse selection costs and inventory risk.]`,
  },
  {
    id: 2,
    question:
      "Compare realized volatility estimators: (1) 5-minute returns, (2) 1-minute returns, (3) Two-scale realized volatility (TSRV) robust to microstructure noise. Under what conditions does each perform best? Design experiment using simulated tick data with known microstructure noise to evaluate.",
    answer: `[Analysis: 5-min: Less noise but fewer observations (less efficient); 1-min: More observations but noise dominates at very high freq; TSRV: Optimal trade-off using two timescales; Simulation: True vol = 20%, add bid-ask bounce ±1bp; Find: 5-min RMSE = 0.5%, 1-min biased upward (noise), TSRV closest to truth; Best: TSRV for liquid stocks, 5-min for illiquid; Practical: Use 5-min for simplicity, TSRV for precision when noise significant.]`,
  },
  {
    id: 3,
    question:
      "Your HFT firm's VWAP execution algorithm consistently underperforms benchmark by 2bps on large orders (>5% ADV). Investigate using tick data: order placement timing, price impact estimation, information leakage detection, and propose improvements.",
    answer: `[Investigation: Analyze with Lee-Ready algorithm for signed trades, measure temporary vs permanent impact; Likely causes: (1) Predictable timing (e.g., always trade first 30min), (2) Too aggressive (moving market), (3) Information leakage (other algos detect pattern); Solutions: (1) Randomize start times, (2) Dynamic participation rate based on volume profile, (3) Use dark pools for large chunks, (4) Incorporate order book resilience measures; Expected improvement: 1-1.5bps reduction possible with optimized timing and dark pool utilization.]`,
  },
];

