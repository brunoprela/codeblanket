import { MultipleChoiceQuestion } from '@/lib/types';

export const statisticsInferenceMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'si-mc-1',
    question:
      'A sample of 100 returns has mean 0.1% and std dev 2%. What is the 95% confidence interval for the true mean?',
    options: [
      '[-0.292%, 0.492%]',
      '[0.06%, 0.14%]',
      '[-0.29%, 0.49%]',
      '[0.1%, 0.1%]',
    ],
    correctAnswer: 2,
    explanation:
      'SE = s/√n = 2%/√100 = 0.2%. 95% CI: 0.1% ± 1.96(0.2%) = 0.1% ± 0.392% = [-0.292%, 0.492%]. Rounding to 2 decimals: [-0.29%, 0.49%]. Note that zero is included in the interval, so we cannot conclude the mean is positive at 95% confidence.',
  },
  {
    id: 'si-mc-2',
    question:
      'You test 20 strategies at α=0.05. What is the probability of at least one false positive (Type I error)?',
    options: ['5%', '36%', '64%', '95%'],
    correctAnswer: 2,
    explanation:
      "P(at least one false positive) = 1 - P(no false positives) = 1 - (1 - α)^k = 1 - (0.95)^20 ≈ 1 - 0.358 ≈ 0.642 = 64%. With 20 tests at α=0.05, there's a 64% chance of at least one false positive. This illustrates why multiple testing correction (Bonferroni, FDR) is essential.",
  },
  {
    id: 'si-mc-3',
    question: 'A t-test gives p-value = 0.03. Which statement is correct?',
    options: [
      'There is a 3% probability the null hypothesis is true',
      'There is a 97% probability the alternative is true',
      'If H₀ is true, there is a 3% chance of seeing data this extreme',
      'The result is practically significant',
    ],
    correctAnswer: 2,
    explanation:
      'P-value is P(data or more extreme | H₀ true). It is NOT the probability H₀ is true or H₁ is true (those require Bayesian analysis with priors). P-value of 0.03 means: if the null hypothesis were true, we would see data this extreme only 3% of the time. Statistical significance (p<0.05) does not imply practical significance.',
  },
  {
    id: 'si-mc-4',
    question: 'To achieve 80% power, you need to ensure:',
    options: [
      'α = 0.20',
      'β = 0.20',
      'P(Type I error) = 0.80',
      'Sample size n = 80',
    ],
    correctAnswer: 1,
    explanation:
      'Power = 1 - β = P(correctly rejecting false H₀). For 80% power, β = 0.20 (Type II error rate = 20%). α is Type I error rate (typically 0.05), not related to power. Power depends on α, effect size, sample size n, and standard deviation - not just one fixed value.',
  },
  {
    id: 'si-mc-5',
    question: 'Which requires the LARGEST sample size to achieve 80% power?',
    options: [
      'Detecting effect size = 0.5σ',
      'Detecting effect size = 1.0σ',
      'Detecting effect size = 2.0σ',
      'All require same sample size',
    ],
    correctAnswer: 0,
    explanation:
      'Sample size n ∝ (σ/δ)² where δ is effect size to detect. Smaller effects require larger samples. For δ = 0.5σ: n ∝ 1/0.25 = 4. For δ = 1.0σ: n ∝ 1/1 = 1. For δ = 2.0σ: n ∝ 1/4 = 0.25. Detecting small effects (0.5σ) requires 16× more data than large effects (2σ).',
  },
];
