/**
 * Multiple choice questions for Bivariate Analysis section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const bivariateanalysisMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Two features have Pearson correlation = 0.3 but Spearman correlation = 0.8. What does this suggest?',
    options: [
      'The features have a strong linear relationship',
      'The features have a non-linear but monotonic relationship',
      'The features are completely independent',
      'There is a data error in the calculation',
    ],
    correctAnswer: 1,
    explanation:
      'When Spearman (rank-based) is much higher than Pearson (linear), it indicates a non-linear but monotonic relationship. The variables increase together but not in a straight line (e.g., exponential or logarithmic relationship).',
  },
  {
    id: 'mc2',
    question:
      'You have three product categories and want to test if average sales differ significantly. Normality tests fail for all groups. Which test should you use?',
    options: [
      'One-way ANOVA',
      'T-test',
      'Kruskal-Wallis test',
      'Chi-square test',
    ],
    correctAnswer: 2,
    explanation:
      "When normality assumption is violated, use the Kruskal-Wallis test - the non-parametric alternative to one-way ANOVA. It uses ranks instead of raw values and doesn't assume normality.",
  },
  {
    id: 'mc3',
    question:
      'Feature X and Feature Y have correlation r = -0.92. What does this mean?',
    options: [
      'As X increases, Y increases strongly',
      'As X increases, Y decreases strongly (strong negative relationship)',
      'X causes Y to decrease',
      'X and Y are independent',
    ],
    correctAnswer: 1,
    explanation:
      'A correlation of -0.92 indicates a strong negative (inverse) linear relationship: as X increases, Y decreases. The negative sign indicates direction, the magnitude (0.92) indicates strength. Note: this is correlation, not causation.',
  },
  {
    id: 'mc4',
    question: 'What does a p-value of 0.03 from an ANOVA test tell you?',
    options: [
      'The effect size is large',
      'The groups are definitely different',
      'There is a 3% probability the observed differences occurred by chance if groups are actually equal',
      'The differences are practically important',
    ],
    correctAnswer: 2,
    explanation:
      "P-value = 0.03 means there's a 3% probability of observing these differences (or more extreme) if the null hypothesis (groups are equal) is true. It doesn't indicate effect size or practical importance - need to check effect size measures like eta-squared.",
  },
  {
    id: 'mc5',
    question:
      'Chi-square test of independence between two categorical variables gives p-value = 0.45. What can you conclude?',
    options: [
      'The variables are strongly associated',
      'The variables are completely independent',
      'There is insufficient evidence to conclude the variables are dependent',
      'One variable causes changes in the other',
    ],
    correctAnswer: 2,
    explanation:
      "P-value = 0.45 (>> 0.05) means we fail to reject the null hypothesis of independence. This means there's insufficient evidence to conclude the variables are dependent. It does NOT prove independence, and never implies causation.",
  },
];
