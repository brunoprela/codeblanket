import { MultipleChoiceQuestion } from '@/types/curriculum';

export const abTestingForMlQuestions: MultipleChoiceQuestion[] = [
  {
    id: 'abt-mc-1',
    question:
      "You're A/B testing a new recommendation model (variant B) against the current model (control A). After one week with 50,000 users per variant, variant B shows 2% higher click-through rate (CTR: 5.1% vs 5.0%) but the difference is not statistically significant (p=0.15). What is the most appropriate action?",
    options: [
      "Conclude the new model doesn't improve performance and abandon it",
      'Calculate required sample size for desired statistical power and extend the test if feasible',
      'Deploy the new model since it shows positive directional improvement',
      'Reduce the significance threshold to p < 0.20 to make the result significant',
    ],
    correctAnswer: 1,
    explanation:
      'Calculate the required sample size for adequate statistical power (typically 80%) to detect a 2% lift. The test may be underpowered—if the effect size is real but small, you need more samples. Use power analysis to determine if extending the test would reach significance. Option A prematurely abandons a potentially valuable model. Option C risks deploying a model that may not actually be better (p=0.15 means 15% chance the observed difference is random). Option D (changing significance threshold post-hoc) is p-hacking and invalidates the test.',
    difficulty: 'advanced',
    topic: 'A/B Testing for ML',
  },
  {
    id: 'abt-mc-2',
    question:
      'Your A/B test compares model versions across 10 different metrics (CTR, conversion rate, revenue, engagement time, etc.). Variant B shows significant improvement (p < 0.05) in 2 metrics but no significant differences in the other 8. How should you interpret these results?',
    options: [
      'The variant is successful since it improved 2 key metrics',
      'Apply Bonferroni correction: use adjusted significance level of 0.05/10 = 0.005 to account for multiple comparisons',
      'Use a composite metric combining all 10 metrics to make a single decision',
      'Test only the primary metric and ignore secondary metrics',
    ],
    correctAnswer: 1,
    explanation:
      "When testing multiple metrics, you increase the probability of false positives (Type I error). Bonferroni correction adjusts the significance threshold by dividing by the number of comparisons (0.05/10 = 0.005) to maintain the family-wise error rate. Other methods like Benjamini-Hochberg (FDR control) can also be used. Option A ignores multiple testing problems—with 10 metrics and p<0.05, you'd expect ~0.5 false positives by chance. Option C (composite metric) is good but doesn't address the multiple testing issue directly. Option D (single primary metric) is valid but loses information from secondary metrics.",
    difficulty: 'advanced',
    topic: 'A/B Testing for ML',
  },
  {
    id: 'abt-mc-3',
    question:
      "You're testing a new credit scoring model using A/B testing. Users in variant B (new model) receive different loan approval decisions than they would have in control A. This raises concerns about fairness and long-term user impact. Which experimentation approach best addresses these concerns?",
    options: [
      'Standard A/B test but monitor for adverse effects on protected groups',
      'Interleaving: show predictions from both models to users and measure engagement with each',
      'Backtest using historical data instead of online experimentation',
      "Shadowing: run new model in parallel, log decisions, but serve control model's decisions to all users",
    ],
    correctAnswer: 3,
    explanation:
      "Shadowing (shadow mode deployment) allows you to evaluate the new model's predictions on real production data without actually affecting user outcomes. You can analyze performance, fairness metrics, and business impact risk-free, then make an informed deployment decision. This is crucial for high-stakes applications like credit scoring. Option A exposes users to potentially harmful decisions. Option B (interleaving) doesn't work for binary decisions like loan approval. Option C (backtesting) doesn't capture online behavior or distribution shift.",
    difficulty: 'advanced',
    topic: 'A/B Testing for ML',
  },
  {
    id: 'abt-mc-4',
    question:
      'Your A/B test shows variant B has 5% higher revenue per user (RPU) but 10% lower user retention after 30 days. Both results are statistically significant. How should you make the deployment decision?',
    options: [
      'Deploy variant B since revenue is the primary business metric',
      'Keep control A since retention is more important for long-term business health',
      'Calculate the long-term value (LTV) impact considering both metrics and their time horizons',
      'Run a longer test to see if retention improves over time',
    ],
    correctAnswer: 2,
    explanation:
      'Calculate lifetime value (LTV) impact by modeling both effects: variant B increases short-term revenue but decreases retention, which reduces future revenue. Use LTV framework: LTV = RPU × (retention rate) × (customer lifetime). The 10% retention drop may outweigh 5% RPU gain when projected over time. This quantifies the tradeoff objectively. Option A focuses only on short-term gains. Option B makes assumptions without quantification. Option D delays the decision without addressing the fundamental tradeoff analysis needed.',
    difficulty: 'advanced',
    topic: 'A/B Testing for ML',
  },
  {
    id: 'abt-mc-5',
    question:
      "You're implementing a multi-armed bandit (MAB) algorithm to dynamically allocate traffic between model variants instead of fixed A/B testing. Which MAB algorithm is most appropriate for a recommendation system where rewards (clicks) are immediately observable?",
    options: [
      'Epsilon-greedy with ε = 0.1 (90% exploit best arm, 10% explore others)',
      'Upper Confidence Bound (UCB) with confidence parameter tuned for the reward variance',
      'Thompson Sampling with Beta priors for click-through rate',
      'EXP3 (Exponential-weight algorithm for Exploration and Exploitation)',
    ],
    correctAnswer: 2,
    explanation:
      'Thompson Sampling with Beta priors is ideal for binary rewards (click/no-click) with immediate feedback. It naturally balances exploration/exploitation, handles multiple arms well, and provides Bayesian uncertainty estimates. It often outperforms other MAB algorithms in practice for CTR optimization. Epsilon-greedy (option A) is simple but explores inefficiently (uniform exploration). UCB (option B) works well but Thompson Sampling typically performs better for binary rewards. EXP3 (option D) is designed for adversarial settings where reward distributions can change adversarially, which is overkill here.',
    difficulty: 'advanced',
    topic: 'A/B Testing for ML',
  },
];
