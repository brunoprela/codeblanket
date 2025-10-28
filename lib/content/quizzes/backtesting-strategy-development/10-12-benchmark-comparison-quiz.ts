import { MultipleChoiceQuestion } from '@/lib/types';

const benchmarkComparisonQuiz: MultipleChoiceQuestion[] = [
  {
    id: 'bench-1',
    question:
      'A market-neutral equity strategy reports a Sharpe ratio of 1.2. Which benchmark is most appropriate for comparison?',
    options: [
      'S&P 500 Index (SPY)',
      'HFRI Equity Market Neutral Index',
      '60/40 stock/bond portfolio',
      'Treasury bills (risk-free rate)',
    ],
    correctAnswer: 1,
    explanation:
      "Market-neutral strategies target zero market beta and absolute returns, so they should be compared to market-neutral benchmarks like the HFRI Equity Market Neutral Index. Comparing to the S&P 500 (Option A) is inappropriate because market-neutral strategies explicitly avoid market beta—a 20% S&P gain shouldn't make the market-neutral strategy look bad. The 60/40 portfolio (Option C) contains directional market exposure. Treasury bills (Option D) are too conservative—market-neutral strategies should outperform risk-free rates significantly. The benchmark must match the strategy's risk/return profile and investment mandate.",
    difficulty: 'intermediate',
  },
  {
    id: 'bench-2',
    question:
      'Your strategy returned 15% with 20% volatility. The benchmark returned 12% with 18% volatility. Both have the same Sharpe ratio. What does this indicate?',
    options: [
      'Your strategy outperformed on a risk-adjusted basis',
      'Your strategy underperformed on a risk-adjusted basis',
      'Your strategy has identical risk-adjusted performance to the benchmark',
      'The comparison is invalid without knowing the beta',
    ],
    correctAnswer: 2,
    explanation:
      'When Sharpe ratios are equal, risk-adjusted performance is identical. Sharpe measures return per unit of volatility, so (15% - Rf) / 20% = (12% - Rf) / 18% means both strategies generate the same excess return per unit of risk. Option A is wrong (no outperformance). Option B is wrong (no underperformance). Option D is misleading—while beta provides additional insight, equal Sharpe ratios definitively indicate equivalent risk-adjusted returns. Your strategy simply achieved higher absolute returns by taking proportionally more risk. An investor who could leverage the benchmark to 20% volatility would achieve the same 15% return.',
    difficulty: 'intermediate',
  },
  {
    id: 'bench-3',
    question:
      "Your strategy has alpha = 3%, beta = 1.5, and the benchmark returned 10%. What was your strategy's expected return according to CAPM?",
    options: [
      '13% (10% × 1.5 - 3%)',
      '15% (10% × 1.5)',
      '18% (3% + 10% × 1.5)',
      '13% (3% + 10%)',
    ],
    correctAnswer: 2,
    explanation:
      'CAPM formula: E[R] = Rf + β × (Rm - Rf) + α. Simplifying when using excess returns: E[R] = α + β × Rm = 3% + 1.5 × 10% = 3% + 15% = 18%. Alpha (3%) is the excess return ABOVE what beta exposure would predict. Beta (1.5) means the strategy moves 1.5x the market. If the market returned 10%, beta alone would predict 15% return. Adding 3% alpha gives 18% total expected return. Option A incorrectly subtracts alpha. Option B ignores alpha entirely. Option D misapplies the formula. Alpha represents skill/edge; beta represents passive market exposure.',
    difficulty: 'advanced',
  },
  {
    id: 'bench-4',
    question: 'Information Ratio (IR) measures what aspect of performance?',
    options: [
      'Total return divided by total risk',
      'Excess return over benchmark divided by tracking error',
      'Sharpe ratio minus benchmark Sharpe ratio',
      'Alpha divided by beta',
    ],
    correctAnswer: 1,
    explanation:
      'Information Ratio = (Portfolio Return - Benchmark Return) / Tracking Error, where tracking error is the standard deviation of excess returns. IR measures how much excess return you generate per unit of active risk (deviation from benchmark). Option A describes Sharpe ratio, not IR. Option C is conceptually related but not the definition. Option D has no meaning. IR answers: "For each percentage point of tracking error I take, how much outperformance do I achieve?" IR > 0.5 is good, IR > 1.0 is excellent. Unlike Sharpe ratio, IR specifically evaluates active management skill relative to a benchmark.',
    difficulty: 'easy',
  },
  {
    id: 'bench-5',
    question:
      'A strategy has 8% tracking error but only 0.5% alpha. What does this suggest?',
    options: [
      'The strategy is highly skilled with excellent risk-adjusted outperformance',
      'The strategy takes substantial active risk but generates minimal alpha (poor IR)',
      'Tracking error is too high to calculate Information Ratio',
      'The strategy is perfectly correlated with the benchmark',
    ],
    correctAnswer: 1,
    explanation:
      'Information Ratio = 0.5% / 8% = 0.0625, which is extremely poor. The strategy deviates significantly from the benchmark (8% tracking error means substantial active bets) but generates almost no excess return (0.5% alpha). This is the worst of both worlds—high active risk, minimal reward. Option A is backwards (this is terrible risk-adjusted performance). Option C is wrong (you can always calculate IR with these inputs). Option D is contradicted by the 8% tracking error (perfect correlation would have zero tracking error). This pattern often indicates a poorly constructed strategy that makes random active bets without skill. Investors would be better off in passive index funds.',
    difficulty: 'advanced',
  },
];

export default benchmarkComparisonQuiz;
