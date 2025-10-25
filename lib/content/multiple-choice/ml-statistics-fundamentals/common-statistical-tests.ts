/**
 * Multiple choice questions for Common Statistical Tests section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const commonstatisticaltestsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Which test should you use to compare model accuracy before and after adding a feature, using the same cross-validation folds?',
    options: ['Two-sample t-test', 'Paired t-test', 'Chi-square test', 'ANOVA'],
    correctAnswer: 1,
    explanation:
      'Use a paired t-test because the measurements are related - they come from the same CV folds. The paired test accounts for within-fold correlation and has more power than treating them as independent samples.',
  },
  {
    id: 'mc2',
    question:
      'You have highly skewed accuracy data with clear outliers and a small sample (n=15). What test should you use to compare two models?',
    options: [
      'Two-sample t-test',
      'Paired t-test',
      'Mann-Whitney U test',
      'Chi-square test',
    ],
    correctAnswer: 2,
    explanation:
      "With small sample, skewed data, and outliers, parametric tests (t-tests) assumptions are violated. Mann-Whitney U test is the non-parametric alternative to the two-sample t-test and doesn't assume normality or require large samples.",
  },
  {
    id: 'mc3',
    question:
      'ANOVA shows p=0.01 when comparing four models. What should you do next?',
    options: [
      'Conclude all models are significantly different',
      'Conclude at least one model is different, then perform post-hoc pairwise tests with correction',
      'Stop - ANOVA tells you which models differ',
      'Reject all models',
    ],
    correctAnswer: 1,
    explanation:
      'ANOVA only tells you that at least one mean is different, not which ones. You need post-hoc pairwise comparisons (e.g., Tukey HSD) with multiple testing correction to determine which specific models differ from each other.',
  },
  {
    id: 'mc4',
    question:
      "What does the Levene\'s test check for before performing a t-test?",
    options: [
      'Normality of distributions',
      'Equality of variances',
      'Independence of samples',
      'Sample size adequacy',
    ],
    correctAnswer: 1,
    explanation:
      "Levene\'s test checks for equality of variances (homoscedasticity) between groups. If variances are significantly different (p < 0.05 in Levene's), you should use Welch's t-test which doesn't assume equal variances.",
  },
  {
    id: 'mc5',
    question:
      'Which test is appropriate for comparing proportions of users who clicked on recommendations across three different algorithms?',
    options: [
      'ANOVA',
      'Chi-square test',
      'Mann-Whitney U test',
      'Kruskal-Wallis test',
    ],
    correctAnswer: 1,
    explanation:
      "When comparing categorical outcomes (clicked/didn't click) across multiple groups, use chi-square test of independence. ANOVA is for continuous data, Mann-Whitney is for two groups, and Kruskal-Wallis is for continuous (but non-parametric) data.",
  },
];
