/**
 * Multiple choice questions for Integration Basics section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const integrationbasicsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'integration-1',
    question: 'The Fundamental Theorem of Calculus states that:',
    options: [
      'Differentiation and integration are unrelated',
      "∫ₐᵇ f(x)dx = F(b) - F(a) where F'(x) = f(x)",
      'All functions can be integrated analytically',
      'Integration always gives a constant',
    ],
    correctAnswer: 1,
    explanation:
      'The Fundamental Theorem connects differentiation and integration: the definite integral of f from a to b equals the antiderivative evaluated at the endpoints.',
  },
  {
    id: 'integration-2',
    question: 'What is ∫ x³ dx?',
    options: ['x⁴ + C', 'x⁴/4 + C', '3x² + C', 'x²/2 + C'],
    correctAnswer: 1,
    explanation:
      'Power rule for integration: ∫ xⁿ dx = xⁿ⁺¹/(n+1) + C. For n=3: x⁴/4 + C.',
  },
  {
    id: 'integration-3',
    question:
      "Simpson's rule is more accurate than the trapezoidal rule because:",
    options: [
      'It uses more function evaluations',
      'It approximates the function with parabolas instead of straight lines',
      'It uses random sampling',
      'It only works for polynomials',
    ],
    correctAnswer: 1,
    explanation:
      "Simpson's rule uses quadratic (parabolic) interpolation between points, providing better approximation than linear (trapezoidal) interpolation.",
  },
  {
    id: 'integration-4',
    question:
      'The expectation E[X] of a continuous random variable is computed as:',
    options: ['∫ f(x) dx', '∫ x·f(x) dx', '∫ x² ·f(x) dx', '∫ log(f(x)) dx'],
    correctAnswer: 1,
    explanation:
      'Expectation is the weighted average: E[X] = ∫ x·f(x) dx, where f(x) is the probability density function.',
  },
  {
    id: 'integration-5',
    question: 'Why is numerical integration necessary in machine learning?',
    options: [
      'Most ML functions have simple analytical integrals',
      'Many probability distributions and loss functions have integrals without closed-form solutions',
      'Analytical integration is always less accurate',
      'Computers cannot do analytical integration',
    ],
    correctAnswer: 1,
    explanation:
      'Many ML problems involve complex integrals (KL divergence, marginal likelihoods, expectations) that lack closed-form solutions, requiring numerical methods.',
  },
];
