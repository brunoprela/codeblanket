export const fixedIncomePortfolioManagementQuiz = [
  {
    id: 'fipm-q-1',
    question:
      'Build a fixed income portfolio optimization system that: (1) Constructs optimal portfolios given constraints (target duration, sector limits, minimum rating), (2) Minimizes tracking error vs benchmark while maximizing alpha, (3) Rebalances automatically when drift exceeds thresholds, (4) Performs attribution analysis (duration, sector, selection effects), (5) Generates trade lists for rebalancing. Include: transaction cost modeling, tax-loss harvesting, liquidity constraints. How do you handle: benchmark changes (index rebalancing)? Corporate actions (calls, tenders)? Cash flows (coupons reinvestment)?',
    sampleAnswer:
      'Portfolio optimization: Use mean-variance optimization or tracking error minimization, Constraints: duration match ±0.5yr, sector weights ±5%, min BBB rating, Objective: min(tracking_error) or max(alpha - λ×tracking_error), Rebalancing: trigger when |drift| > threshold, Transaction costs: model bid-ask + market impact, Attribution: decompose returns into duration/curve/sector/selection/carry components, Trade generation: calculate buy/sell orders to achieve target weights.',
    keyPoints: [
      'Optimization with constraints',
      'Tracking error minimization',
      'Attribution analysis',
      'Automated rebalancing',
      'Transaction cost modeling',
    ],
  },
  {
    id: 'fipm-q-2',
    question:
      'Design a benchmark tracking system for passive fixed income management: (1) Implements stratified sampling (hold subset of benchmark bonds), (2) Minimizes tracking error through cell-matching, (3) Handles index rebalancing events, (4) Optimizes transaction costs vs tracking error trade-off, (5) Reports tracking error and deviations. Include: sampling strategies (market-cap weighted, duration-matched), replication quality metrics. How do you handle: illiquid bonds in benchmark? Large benchmark turnover? Cost of full replication vs sampling?',
    sampleAnswer:
      'Stratified sampling: Divide benchmark into cells (sector × duration × rating), Hold representative bonds from each cell, Market-cap weight within cells, Target tracking error <0.25%, Rebalancing: Match index reconstitution dates, Trade only necessary changes, Transaction costs: 5-20bp per trade, optimize frequency, Quality metrics: R² vs benchmark (>0.95 target), tracking error volatility, holdings overlap.',
    keyPoints: [
      'Stratified sampling',
      'Cell matching',
      'Index rebalancing',
      'Tracking error <0.25%',
      'Cost-error trade-off',
    ],
  },
  {
    id: 'fipm-q-3',
    question:
      'Implement a yield curve positioning system: (1) Analyzes current curve shape (steep, flat, inverted), (2) Recommends barbell vs bullet vs ladder based on forecast, (3) Calculates convexity advantage of different strategies, (4) Simulates P&L under various curve scenarios, (5) Manages rolldown return. Include: butterfly trades, curve flatteners/steepeners. How do you handle: execution (building positions gradually)? Risk limits (DV01 budgets)? Historical analysis (curve regime identification)?',
    sampleAnswer:
      'Curve analysis: Calculate 2s10s, 5s30s spreads, historical percentiles, identify regime (steep >150bp, flat <100bp, inverted <0), Strategy: Steep curve → barbell (convexity), Flat curve → bullet (carry), Inversion → short duration, Butterfly: Long wings + short body, profit from curve changes, Rolldown: Bonds age, move down curve (positive carry if upward-sloping), Risk: Set DV01 limits by maturity bucket.',
    keyPoints: [
      'Curve shape analysis',
      'Barbell vs bullet',
      'Convexity advantage',
      'Rolldown return',
      'Butterfly trades',
    ],
  },
];
