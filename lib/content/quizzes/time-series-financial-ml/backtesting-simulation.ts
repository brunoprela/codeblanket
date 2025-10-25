export const backtestingSimulationQuiz = [
  {
    id: 'bs-q-1',
    question: 'What is lookahead bias and how do you prevent it?',
    sampleAnswer:
      'Lookahead bias: Using future information to make past decisions. Example: Calculate RSI using data up to day 100, but RSI formula uses day 101 (future). Prevention: (1) Always use .shift(1) for features, (2) Walk-forward validation, (3) Strict temporal ordering, (4) Check correlation of shifted features. Very common bug—can inflate backtest returns 20-50%. Always verify no future data leakage.',
    keyPoints: [
      'Using future information in past decisions',
      'Example: RSI calculated with future data',
      'Prevention: shift features, walk-forward validation',
      'Can inflate returns 20-50% (major bug)',
      'Check: corr (feature.shift(-1), target) > corr (feature, target)',
    ],
  },
  {
    id: 'bs-q-2',
    question: 'What is survivorship bias? How does it affect backtests?',
    sampleAnswer:
      'Survivorship bias: Only testing on stocks still trading (survivors), ignoring delisted/bankrupt stocks. Example: Backtest 2000-2020 using current S&P 500 members—excludes Enron, Lehman Bros (bankruptcies). Effect: Inflates returns 2-5% annually. Fix: Use point-in-time universe (stocks actually in index each date), include delisted stocks. Critical for long-term backtests (>5 years).',
    keyPoints: [
      'Testing only on surviving stocks (ignores bankruptcies)',
      'Example: Backtest on current S&P 500 → excludes failures',
      'Inflates returns 2-5% per year',
      'Fix: point-in-time universe, include delisted stocks',
      'Critical for long-term backtests',
    ],
  },
  {
    id: 'bs-q-3',
    question: 'Design Monte Carlo simulation to validate strategy robustness.',
    sampleAnswer:
      'Monte Carlo: (1) Record all backtest trades (entry, exit, PnL), (2) Resample trades randomly with replacement 1000 times, (3) Calculate return distribution, (4) Check: Mean > 0, Std < Mean/2, 5th percentile > -10%, Prob (positive) > 60%. Robust strategy: consistent across resamples. Fragile: Few large wins (lottery-like). Also test: shuffle entry dates, bootstrap returns, vary parameters. Good strategy survives all tests.',
    keyPoints: [
      'Resample trades randomly 1000 times',
      'Calculate return distribution (mean, std, percentiles)',
      'Check: 5th percentile > -10%, Prob(+) > 60%',
      'Robust: consistent, Fragile: few large wins',
      'Also test: shuffle dates, bootstrap, parameter sensitivity',
    ],
  },
];
