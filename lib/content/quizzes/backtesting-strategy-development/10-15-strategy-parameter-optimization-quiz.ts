import { MultipleChoiceQuestion } from '@/lib/types';

const strategyParameterOptimizationQuiz: MultipleChoiceQuestion[] = [
  {
    id: 'opt-1',
    question:
      'You have 3 parameters to optimize, each with 10 possible values. How many combinations must grid search evaluate?',
    options: [
      '30 combinations (10 + 10 + 10)',
      '100 combinations (10 × 10)',
      '1,000 combinations (10 × 10 × 10)',
      '300 combinations (10 × 10 + 10 × 10)',
    ],
    correctAnswer: 2,
    explanation:
      "Grid search evaluates ALL possible combinations, which grows exponentially: 10 × 10 × 10 = 1,000 combinations. This is why grid search doesn't scale beyond 4-5 parameters. With 5 parameters and 10 values each, you'd need 100,000 evaluations. Option A incorrectly adds instead of multiplying. Options B and D miscalculate. This exponential growth is called the 'curse of dimensionality.' Professional solutions: Bayesian optimization (intelligently samples ~50-100 points), genetic algorithms (evolutionary search), or random search (surprisingly effective—testing 100 random combinations often finds near-optimal solutions).",
    difficulty: 'beginner',
  },
  {
    id: 'opt-2',
    question:
      'What is the main advantage of Bayesian optimization over grid search for parameter tuning?',
    options: [
      'Bayesian optimization always finds the global optimum',
      'Bayesian optimization uses past evaluations to intelligently choose which parameters to test next',
      'Bayesian optimization is easier to implement',
      'Bayesian optimization works better with discrete parameters',
    ],
    correctAnswer: 1,
    explanation:
      'Bayesian optimization builds a probabilistic model (usually Gaussian Process) of the objective function and uses it to decide where to sample next, balancing exploration (uncertain regions) vs exploitation (promising regions). This typically finds near-optimal solutions with 10-100x fewer evaluations than grid search. Option A is wrong—no optimization guarantees global optimum in non-convex spaces. Option C is backwards (Bayesian optimization is more complex). Option D is incorrect—it actually works better with continuous parameters. Renaissance Technologies uses Bayesian methods extensively because with 1000+ parameters, exhaustive search is impossible.',
    difficulty: 'intermediate',
  },
  {
    id: 'opt-3',
    question:
      'You optimize parameters on 5 years of data and achieve a Sharpe of 3.5. When tested on the next 6 months (out-of-sample), the Sharpe drops to 0.8. What likely happened?',
    options: [
      'The out-of-sample period had unusually poor market conditions',
      'The optimization process overfit to the training data',
      'The parameters need further tuning to improve OOS performance',
      "6 months isn't enough time to evaluate the strategy",
    ],
    correctAnswer: 1,
    explanation:
      "A 77% degradation (3.5 to 0.8 Sharpe) is classic overfitting. The optimization found parameters that worked exceptionally well on the specific 5-year sample but don't generalize. Option A is possible but unlikely to cause such dramatic degradation. Option C is dangerous—further tuning on the OOS data would contaminate it. Option D misses the point—dramatic degradation indicates a fundamental problem, not insufficient testing time. Proper approach: Use walk-forward optimization (optimize on window 1, test on window 2, roll forward), limit parameter search space, prefer simpler models, and expect 20-40% degradation as normal.",
    difficulty: 'advanced',
  },
  {
    id: 'opt-4',
    question:
      "In genetic algorithm optimization, what is the purpose of the 'mutation' step?",
    options: [
      'To ensure every parameter combination is eventually tested',
      'To introduce randomness and prevent getting stuck in local optima',
      'To remove poor-performing strategies from the population',
      'To combine the best features of two parent strategies',
    ],
    correctAnswer: 1,
    explanation:
      "Mutation introduces random changes to offspring, maintaining genetic diversity and allowing exploration of new parameter regions. Without mutation, the algorithm could converge prematurely to a local optimum. Option A describes exhaustive search, not mutation. Option C describes selection (survival of fittest). Option D describes crossover (combining parents). The mutation rate is typically low (5-15%)—too high causes random search, too low risks premature convergence. Think of evolution: most offspring inherit parents' genes (crossover), but occasional mutations enable adaptation to new environments.",
    difficulty: 'intermediate',
  },
  {
    id: 'opt-5',
    question:
      'Walk-forward optimization uses what approach to prevent overfitting?',
    options: [
      'Optimizes on entire dataset, then validates on separate dataset',
      'Optimizes on rolling training windows and tests on sequential out-of-sample windows',
      'Averages parameters from multiple optimization runs',
      'Uses regularization to penalize complex parameter sets',
    ],
    correctAnswer: 1,
    explanation:
      "Walk-forward optimization divides data into sequential train/test periods. Optimize on Period 1, test on Period 2. Roll forward: optimize on Period 2, test on Period 3, etc. This mimics real-world deployment where you periodically re-optimize on recent data and trade the next period. Option A describes single train/test split (less robust). Option C is a good addition but not walk-forward's core mechanism. Option D describes regularization (different technique). Walk-forward is gold standard because it validates parameters generalize across multiple time periods and different market regimes. Two Sigma re-optimizes monthly and requires strategies to maintain performance across rolling windows.",
    difficulty: 'advanced',
  },
];

export default strategyParameterOptimizationQuiz;
