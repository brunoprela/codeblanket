/**
 * Multiple choice questions for Stochastic Calculus Fundamentals section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const stochasticcalculusMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'stoch-1',
    question: 'Brownian motion W_t has the property that W_t - W_s follows:',
    options: [
      'Uniform distribution',
      'Normal distribution N(0, t-s)',
      'Exponential distribution',
      'Poisson distribution',
    ],
    correctAnswer: 1,
    explanation:
      'Brownian motion increments W_t - W_s are normally distributed with mean 0 and variance t-s.',
    difficulty: 'medium',
  },
  {
    id: 'stoch-2',
    question: "Itô's lemma differs from the standard chain rule by:",
    options: [
      'Having no difference',
      'Including an extra term (1/2)σ²·∂²f/∂x² from quadratic variation',
      'Only applying to linear functions',
      'Not requiring derivatives',
    ],
    correctAnswer: 1,
    explanation:
      "Itô's lemma includes an additional second-order term (1/2)σ²·∂²f/∂x² because Brownian motion has non-zero quadratic variation: (dW_t)² = dt.",
    difficulty: 'hard',
  },
  {
    id: 'stoch-3',
    question:
      'Stochastic Gradient Langevin Dynamics (SGLD) adds noise to gradient descent to:',
    options: [
      'Make optimization slower',
      'Enable exploration and sampling from the posterior distribution',
      'Increase computational cost',
      'Remove the need for gradients',
    ],
    correctAnswer: 1,
    explanation:
      'SGLD adds carefully calibrated noise to enable exploration of the optimization landscape and asymptotically samples from the posterior distribution p(x) ∝ exp(-f(x)).',
    difficulty: 'medium',
  },
  {
    id: 'stoch-4',
    question: 'In a diffusion model, the forward process:',
    options: [
      'Removes noise from data',
      'Gradually adds noise to transform data into pure noise',
      'Trains the neural network',
      'Generates new samples',
    ],
    correctAnswer: 1,
    explanation:
      'The forward diffusion process gradually adds Gaussian noise to the data until it becomes indistinguishable from pure noise. The reverse process (learned) then denoises to generate samples.',
    difficulty: 'medium',
  },
  {
    id: 'stoch-5',
    question: 'The Euler-Maruyama method is:',
    options: [
      'An analytical solution method',
      'A numerical scheme for simulating stochastic differential equations',
      'Only for deterministic ODEs',
      'A deep learning architecture',
    ],
    correctAnswer: 1,
    explanation:
      'Euler-Maruyama is a numerical method for simulating SDEs, discretizing dX_t = μ dt + σ dW_t into finite time steps with Gaussian increments.',
    difficulty: 'easy',
  },
];
