import { MultipleChoiceQuestion } from '@/lib/types';

const statisticalSignificanceTestingQuiz: MultipleChoiceQuestion[] = [
  {
    id: 'stat-sig-1',
    question:
      'A trading strategy has a Sharpe ratio of 1.2 over 250 trading days with a p-value of 0.08. What is the correct interpretation?',
    options: [
      'The strategy has an 8% probability of having no edge',
      "There's an 8% chance the observed Sharpe ratio occurred by luck, so we reject the strategy",
      "We cannot reject the null hypothesis at the 5% significance level, but this doesn't prove the strategy has no edge",
      'The strategy is 92% likely to be profitable',
    ],
    correctAnswer: 2,
    explanation:
      "A p-value of 0.08 means that if the strategy truly had no edge, there would be an 8% probability of observing a Sharpe ratio of 1.2 or better purely by chance. Since 0.08 > 0.05 (common significance level), we cannot reject the null hypothesis at the 5% level. However, this doesn't mean the strategy has no edge—it means we don't have strong enough statistical evidence to conclude it does. The strategy might still have genuine edge; we just need more data or a larger effect size to prove it statistically. Options A and D misinterpret what p-values represent.",
    difficulty: 'intermediate',
  },
  {
    id: 'stat-sig-2',
    question:
      'You test 100 different trading strategies at a 5% significance level (α=0.05). How many strategies would you expect to appear statistically significant purely by chance?',
    options: [
      "None, because they're independent tests",
      '5 strategies (5% of 100)',
      '1 strategy (the winner)',
      'It depends on the correlation between strategies',
    ],
    correctAnswer: 1,
    explanation:
      "This is the multiple testing problem. With α=0.05, each test has a 5% chance of Type I error (false positive). Testing 100 strategies, you expect 100 × 0.05 = 5 strategies to appear significant purely by chance, even if none have genuine edge. This is why multiple testing corrections (Bonferroni, FDR) are crucial when screening many strategies. Without correction, you're almost guaranteed to find 'significant' strategies that are actually just luck. To control for this, use Bonferroni correction: α_corrected = 0.05/100 = 0.0005.",
    difficulty: 'advanced',
  },
  {
    id: 'stat-sig-3',
    question:
      'What is the main advantage of using a permutation test over a parametric t-test for testing trading strategy significance?',
    options: [
      'Permutation tests are always more powerful',
      "Permutation tests don't require assumptions about return distributions",
      'Permutation tests are faster to compute',
      'Permutation tests always give lower p-values',
    ],
    correctAnswer: 1,
    explanation:
      "Permutation tests are non-parametric, meaning they don't require assumptions about the underlying distribution of returns (like normality). They work by randomly shuffling the data many times and comparing the observed statistic to the distribution of shuffled statistics. This is particularly valuable in finance where returns often have fat tails, skewness, and other departures from normality that violate t-test assumptions. However, permutation tests are NOT faster (they're much slower due to resampling), don't always give lower p-values, and aren't necessarily more powerful than parametric tests when parametric assumptions hold.",
    difficulty: 'intermediate',
  },
  {
    id: 'stat-sig-4',
    question:
      'A strategy has a 95% confidence interval for Sharpe ratio of [0.3, 1.9]. What does this tell you?',
    options: [
      'The true Sharpe ratio is between 0.3 and 1.9 with 95% probability',
      "There's significant uncertainty about the strategy's true performance",
      "The strategy is statistically significant because the interval doesn't include 0",
      'Both B and C are correct',
    ],
    correctAnswer: 3,
    explanation:
      "Both B and C are correct. The wide confidence interval (0.3 to 1.9, a range of 1.6) indicates substantial uncertainty about the true Sharpe ratio. However, since the interval doesn't include 0, the strategy IS statistically significant at the 5% level—we can reject the hypothesis that it has no edge. Option A is a common misinterpretation: the confidence interval is about the estimation procedure, not a probability statement about the parameter. The correct interpretation is: if we repeated this process many times, 95% of the intervals would contain the true Sharpe ratio.",
    difficulty: 'advanced',
  },
  {
    id: 'stat-sig-5',
    question:
      'Why might a trading strategy be statistically significant but not economically significant?',
    options: [
      "This can't happen—statistical significance implies economic significance",
      "The strategy's returns are statistically positive but too small to cover transaction costs",
      'The p-value is below 0.05',
      'The confidence interval is too narrow',
    ],
    correctAnswer: 1,
    explanation:
      'Statistical significance only means the results are unlikely due to chance—it says nothing about whether the strategy is profitable in practice. Example: A strategy with annualized Sharpe of 0.3 might be statistically significant with enough data (large sample size), but after accounting for transaction costs (commission, slippage, market impact), it loses money. Statistical significance is about detecting an effect; economic significance is about whether that effect is large enough to matter. Professional trading firms require BOTH: strategies must be statistically significant AND generate sufficient risk-adjusted returns after all costs to justify capital allocation.',
    difficulty: 'easy',
  },
];

export default statisticalSignificanceTestingQuiz;
