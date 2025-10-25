/**
 * Multiple choice questions for Experimental Design section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const experimentaldesignMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the primary purpose of randomization in experiments?',
    options: [
      'To make the experiment faster',
      'To balance both observed and unobserved confounding variables across groups',
      'To increase sample size',
      'To make statistical tests easier',
    ],
    correctAnswer: 1,
    explanation:
      'Randomization balances all confounding variables (both measured and unmeasured) across treatment and control groups, allowing causal inferences. Without randomization, group differences could be due to pre-existing differences rather than the treatment.',
  },
  {
    id: 'mc2',
    question:
      'You need 80% power to detect a 2% improvement with α=0.05. What happens if you only achieve 50% power?',
    options: [
      'Type I error rate increases',
      'Type II error rate increases (higher chance of missing real effect)',
      'The test becomes invalid',
      'Sample size automatically adjusts',
    ],
    correctAnswer: 1,
    explanation:
      'Power = 1 - β (Type II error). Lower power (50% instead of 80%) means higher β (Type II error), so you have a greater chance of failing to detect a real effect. Type I error rate (α) remains at 0.05.',
  },
  {
    id: 'mc3',
    question: 'What is "p-hacking" or "peeking" in A/B testing?',
    options: [
      'Looking at results multiple times before reaching predetermined sample size',
      'Calculating p-values incorrectly',
      'Using different statistical tests',
      'Removing outliers',
    ],
    correctAnswer: 0,
    explanation:
      'P-hacking/peeking is checking test results multiple times during data collection and stopping when you find significance. This inflates the false positive rate because each look is effectively a separate test. The solution is to pre-specify sample size and test only once, or use sequential testing methods with appropriate corrections.',
  },
  {
    id: 'mc4',
    question: 'In A/B testing, what does "statistical power" represent?',
    options: [
      'The probability of making a Type I error',
      'The probability of correctly detecting a real effect (1 - Type II error)',
      'The sample size',
      'The effect size',
    ],
    correctAnswer: 1,
    explanation:
      "Statistical power = P(reject H₀ | H₁ is true) = 1 - β (where β is Type II error rate). It\'s the probability of correctly detecting a real effect when one exists. Higher power means lower chance of missing real improvements.",
  },
  {
    id: 'mc5',
    question:
      'Why should sample size be determined before running an experiment?',
    options: [
      'To reduce computation time',
      'To ensure adequate statistical power and prevent p-hacking',
      'To make randomization easier',
      'It is not necessary to determine sample size beforehand',
    ],
    correctAnswer: 1,
    explanation:
      'Pre-determining sample size through power analysis ensures you have adequate power to detect meaningful effects and prevents p-hacking from repeatedly checking results. It also helps with resource planning and maintains the validity of hypothesis tests.',
  },
];
