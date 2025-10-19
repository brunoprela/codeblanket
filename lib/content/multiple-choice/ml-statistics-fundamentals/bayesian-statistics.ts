/**
 * Multiple choice questions for Bayesian Statistics section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const bayesianstatisticsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'In Bayesian statistics, what does the posterior distribution represent?',
    options: [
      'The probability of the data',
      'Our belief about parameters before seeing data',
      'Our updated belief about parameters after seeing data',
      'The likelihood function',
    ],
    correctAnswer: 2,
    explanation:
      'The posterior P(θ|data) represents our updated belief about the parameters after incorporating both the prior belief and the observed data through Bayes theorem.',
  },
  {
    id: 'mc2',
    question: 'How should you interpret a 95% Bayesian credible interval?',
    options: [
      'If we repeat the experiment, 95% of intervals will contain the true parameter',
      'There is a 95% probability that the true parameter is in this interval',
      'The procedure is correct 95% of the time',
      'It has the same interpretation as a confidence interval',
    ],
    correctAnswer: 1,
    explanation:
      'Bayesian credible intervals allow direct probability statements: there is a 95% probability the parameter is in the interval. This is different from frequentist confidence intervals which describe the long-run behavior of the procedure.',
  },
  {
    id: 'mc3',
    question: 'What happens to the posterior as you collect more data?',
    options: [
      'It becomes identical to the prior',
      'It increasingly reflects the likelihood, with prior influence diminishing',
      'It stays the same',
      'It becomes less certain',
    ],
    correctAnswer: 1,
    explanation:
      'As more data is collected, the likelihood dominates and the posterior increasingly reflects the data rather than the prior. With infinite data, the posterior converges to the maximum likelihood estimate regardless of the prior (assuming the prior gives non-zero probability to the true value).',
  },
  {
    id: 'mc4',
    question: 'What is a "non-informative" or "weak" prior?',
    options: [
      'A prior that gives all parameter values equal probability',
      'A prior with high variance that expresses little prior knowledge',
      'A prior based on previous data',
      'The likelihood function',
    ],
    correctAnswer: 1,
    explanation:
      'A non-informative or weak prior has high variance and expresses little prior knowledge about the parameter. It allows the data (likelihood) to dominate the posterior. Uniform priors are one type, but there are other ways to construct non-informative priors.',
  },
  {
    id: 'mc5',
    question:
      'In Bayes theorem P(θ|data) ∝ P(data|θ) × P(θ), what does each term represent?',
    options: [
      'Posterior ∝ Prior × Evidence',
      'Prior ∝ Likelihood × Posterior',
      'Posterior ∝ Likelihood × Prior',
      'Likelihood ∝ Posterior × Prior',
    ],
    correctAnswer: 2,
    explanation:
      'P(θ|data) is the posterior, P(data|θ) is the likelihood, and P(θ) is the prior. The posterior is proportional to the likelihood times the prior. The proportionality constant is the evidence P(data), which normalizes the distribution.',
  },
];
