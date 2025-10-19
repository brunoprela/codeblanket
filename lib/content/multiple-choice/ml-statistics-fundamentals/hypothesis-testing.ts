/**
 * Multiple choice questions for Hypothesis Testing section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const hypothesistestingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What does a p-value of 0.03 mean?',
    options: [
      'There is a 3% chance that the null hypothesis is true',
      'There is a 3% chance that the result is due to chance',
      'If the null hypothesis is true, there is a 3% probability of observing data as extreme or more extreme than what we got',
      'The effect size is 0.03',
    ],
    correctAnswer: 2,
    explanation:
      'The p-value is the probability of observing data at least as extreme as what we obtained, ASSUMING the null hypothesis is true. It is NOT the probability that H₀ is true, nor the probability the result is due to chance.',
  },
  {
    id: 'mc2',
    question:
      'In hypothesis testing for ML model comparison, which type of error occurs when you deploy a new model that is actually no better than the baseline?',
    options: [
      'Type I error (False Positive)',
      'Type II error (False Negative)',
      'Standard error',
      'Sampling error',
    ],
    correctAnswer: 0,
    explanation:
      "Deploying a model that isn't actually better is a Type I error - rejecting H₀ (models are the same) when H₀ is actually true. This is a false positive: you conclude there's an improvement when there isn't one.",
  },
  {
    id: 'mc3',
    question:
      'You test 10 features and find that 1 has p=0.04. With α=0.05, what should you do?',
    options: [
      'Include the feature since p < 0.05',
      'Reject the feature since this is likely a false positive from multiple testing',
      'Apply Bonferroni correction: use α/10 = 0.005 as threshold',
      'Run more experiments',
    ],
    correctAnswer: 2,
    explanation:
      'When performing multiple tests, you should apply a correction like Bonferroni (α_corrected = α/n_tests). Here, α/10 = 0.05/10 = 0.005. Since p=0.04 > 0.005, you should not conclude significance. With 10 tests, finding one p<0.05 by chance is expected even if all null hypotheses are true.',
  },
  {
    id: 'mc4',
    question: 'What is statistical power in hypothesis testing?',
    options: [
      'The probability of making a Type I error',
      'The probability of correctly rejecting a false null hypothesis',
      'The same as the p-value',
      'The significance level (α)',
    ],
    correctAnswer: 1,
    explanation:
      "Statistical power = 1 - β = P(Reject H₀ | H₀ is false). It's the probability of correctly detecting a real effect when one exists. Higher power means fewer Type II errors (false negatives). Power depends on sample size, effect size, and significance level.",
  },
  {
    id: 'mc5',
    question:
      'A test shows p=0.001 (highly significant) but the improvement is from 85.00% to 85.05% accuracy. What should you conclude?',
    options: [
      'Deploy immediately since p < 0.05',
      'The result is both statistically and practically significant',
      'The result is statistically significant but may not be practically significant',
      'The test is invalid',
    ],
    correctAnswer: 2,
    explanation:
      'With large sample sizes, even tiny differences can be statistically significant (very small p-values). However, a 0.05 percentage point improvement may not justify the cost and effort of deploying a new model. Always assess practical significance (effect size) alongside statistical significance.',
  },
];
