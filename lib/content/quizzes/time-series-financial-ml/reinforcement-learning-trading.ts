export const reinforcementLearningTradingQuiz = [
  {
    id: 'rlt-q-1',
    question:
      'Formulate stock trading as reinforcement learning problem: states, actions, rewards.',
    sampleAnswer:
      'States: [price, volume, technical indicators, position, balance, portfolio value] → 10-20 features. Actions: {Buy, Sell, Hold} or continuous position size. Rewards: (1) PnL per trade, (2) Sharpe ratio, (3) Risk-adjusted return. Episode: One trading period (e.g., 1 month). Goal: Maximize cumulative reward. Challenges: Sparse rewards (profit only on close), non-stationary (markets change), sample efficiency (limited historical data).',
    keyPoints: [
      'States: price + indicators + position + balance',
      'Actions: Buy/Sell/Hold or continuous sizing',
      'Rewards: PnL, Sharpe, or risk-adjusted return',
      'Episode: Trading period (days/weeks/months)',
      'Challenges: sparse rewards, non-stationary, sample efficiency',
    ],
  },
  {
    id: 'rlt-q-2',
    question: 'Compare RL vs supervised learning for trading. Which is better?',
    sampleAnswer:
      'Supervised: Predict direction/return, simpler, faster training, proven (55-60% accuracy). RL: Learn full policy (when to enter/exit/size), handles sequential decisions, but harder to train, requires more data, less stable. Verdict: Start with supervised (reliable baseline). Use RL for: (1) Complex multi-step strategies, (2) Continuous position sizing, (3) Market making. Best: Hybrid—supervised for signals, RL for execution/sizing. Industry mostly uses supervised + rules.',
    keyPoints: [
      'Supervised: simpler, proven, 55-60% accuracy',
      'RL: handles sequential decisions, harder to train',
      'RL advantages: continuous actions, full policy',
      'RL challenges: sample efficiency, reward engineering',
      'Best: hybrid—supervised signals + RL execution',
    ],
  },
  {
    id: 'rlt-q-3',
    question:
      'Design reward function for RL trading agent. What works and what fails?',
    sampleAnswer:
      'Good rewards: (1) Risk-adjusted: Sharpe ratio (best), (2) Incremental: reward each step, not only end. (3) Shaped: small reward for reducing loss. Bad rewards: (1) Only final PnL (sparse, hard to learn), (2) Raw return (ignores risk), (3) Win rate (ignores magnitude). Best: Reward = (return - risk_free) / volatility at each step. Add penalties: large drawdowns (-10), excessive trading (-0.1 per trade). Balance exploration and risk. Test in simulation first.',
    keyPoints: [
      'Good: Sharpe ratio (risk-adjusted), incremental rewards',
      'Bad: sparse rewards (only final PnL), ignores risk',
      'Best: reward = return / volatility per step',
      'Penalties: drawdowns, excessive trading',
      'Test: simulation, ensure agent learns stable policy',
    ],
  },
];
