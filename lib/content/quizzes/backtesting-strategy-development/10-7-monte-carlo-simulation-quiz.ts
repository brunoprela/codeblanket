import { MultipleChoiceQuestion } from '@/lib/types';

const monteCarloSimulationQuiz: MultipleChoiceQuestion[] = [
  {
    id: 'mc-sim-1',
    question:
      "A trading strategy shows a Sharpe ratio of 1.8 on the original backtest. After running 10,000 Monte Carlo simulations randomizing trade order, the strategy's Sharpe ratio ranks at the 23rd percentile. What does this indicate?",
    options: [
      'The strategy has a genuine edge and is robust',
      "The strategy's performance is likely due to luck and favorable trade sequencing",
      'Monte Carlo simulation is not applicable to this strategy',
      'The strategy needs more trades for statistical significance',
    ],
    correctAnswer: 1,
    explanation:
      "A 23rd percentile ranking means the original strategy performed worse than 77% of random trade orderings. This indicates the strategy's performance was highly dependent on the specific sequence of trades (path dependency) and likely benefited from lucky timing. A robust strategy should rank in the 75th+ percentile, showing it performs better than random. This is a critical warning sign that the strategy's edge may be illusory.",
    difficulty: 'intermediate',
  },
  {
    id: 'mc-sim-2',
    question:
      'What is the primary advantage of using block bootstrap (resampling consecutive blocks of returns) instead of standard bootstrap (random resampling with replacement) for time series data?',
    options: [
      'Block bootstrap is computationally faster',
      'Block bootstrap preserves autocorrelation structure in the returns',
      'Block bootstrap provides wider confidence intervals',
      'Block bootstrap eliminates the need for out-of-sample testing',
    ],
    correctAnswer: 1,
    explanation:
      "Block bootstrap preserves the time series structure and autocorrelation in returns by keeping consecutive observations together. Standard bootstrap treats observations as independent, which violates the reality of financial time series where returns often exhibit serial correlation. This preservation is crucial for strategies that depend on momentum or mean reversion. Block bootstrap is NOT faster (actually slower due to block management), doesn't inherently provide wider intervals, and definitely doesn't eliminate the need for out-of-sample testing.",
    difficulty: 'advanced',
  },
  {
    id: 'mc-sim-3',
    question:
      "A Monte Carlo simulation shows that a strategy's 95% confidence interval for annual return is [-5%, +25%], with the original backtest showing +18%. What is the correct interpretation?",
    options: [
      'The strategy has a 95% probability of returning between -5% and +25% next year',
      'If we could repeat the historical period many times with resampled returns, 95% of the time the return would fall between -5% and +25%',
      "The strategy's true expected return is 10% (the midpoint of the confidence interval)",
      'The strategy is too risky and should be rejected',
    ],
    correctAnswer: 1,
    explanation:
      "The confidence interval represents the range of outcomes if we could repeat the historical period many times with different return sequences. It's a statement about estimation uncertainty, not a probability forecast for future performance. Option A is a common misinterpretation—CI's don't directly predict future returns. Option C incorrectly assumes the midpoint is the expected return (the original backtest value of 18% is the point estimate). Option D makes a value judgment without context—wide CI's indicate uncertainty but don't automatically make a strategy 'too risky' if the risk-adjusted returns are acceptable.",
    difficulty: 'advanced',
  },
  {
    id: 'mc-sim-4',
    question:
      'When stress testing a trading strategy using Monte Carlo simulation, which approach is most appropriate for simulating a market crash scenario?',
    options: [
      'Multiply all historical returns by a constant factor',
      'Insert extreme negative returns at random positions while preserving the original distribution shape',
      'Replace all positive returns with zero',
      'Use parametric simulation from a fitted normal distribution',
    ],
    correctAnswer: 1,
    explanation:
      "Inserting extreme negative returns at random positions simulates realistic crash scenarios while maintaining the strategy's overall characteristics. This approach tests how the strategy performs when hit by tail events. Option A (constant scaling) doesn't create crashes, just increases volatility uniformly. Option C (removing all gains) is unrealistic and extreme. Option D (normal distribution) fails to capture fat tails and extreme events that characterize real market crashes. Professional risk management requires testing strategies against realized crash scenarios (e.g., 1987, 2008, 2020) and synthetic tail events.",
    difficulty: 'intermediate',
  },
  {
    id: 'mc-sim-5',
    question:
      'How many Monte Carlo simulations are typically needed to get stable confidence interval estimates for trading strategy metrics?',
    options: [
      '100-500 simulations are sufficient for most strategies',
      '1,000-5,000 simulations provide reasonable estimates',
      '10,000+ simulations are recommended for reliable estimates',
      "The number doesn't matter as long as you use bootstrap",
    ],
    correctAnswer: 2,
    explanation:
      "10,000+ simulations are generally recommended for stable, reliable confidence interval estimates. With fewer simulations, the CI boundaries themselves become unstable—run the analysis twice and you'll get noticeably different results. Academic literature and professional practice typically use 10,000-50,000 simulations. While 1,000 simulations might give a rough estimate, the standard error of the confidence interval endpoints is still too large for production use. The computational cost is usually acceptable with modern hardware (seconds to minutes for most strategies). Option D is wrong—the number of simulations matters regardless of the resampling method used.",
    difficulty: 'beginner',
  },
];

export default monteCarloSimulationQuiz;
