/**
 * Multiple choice questions for Convex Optimization section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const convexoptimizationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'convex-1',
    question: 'A function f is convex if:',
    options: [
      'It has a unique minimum',
      'The line segment between any two points on the graph lies above the graph',
      'Its gradient is always positive',
      'It is differentiable everywhere',
    ],
    correctAnswer: 1,
    explanation:
      'Convexity means the chord (line segment) between any two points on the graph lies above or on the graph: f(θx + (1-θ)y) ≤ θf(x) + (1-θ)f(y).',
  },
  {
    id: 'convex-2',
    question:
      'For a twice-differentiable function, convexity can be checked by:',
    options: [
      'Verifying the gradient is zero',
      'Checking if the Hessian is positive semidefinite everywhere',
      'Ensuring the function is monotonic',
      'Computing the Jacobian',
    ],
    correctAnswer: 1,
    explanation:
      'Second-order condition: f is convex iff ∇²f(x) ⪰ 0 (Hessian positive semidefinite) for all x.',
  },
  {
    id: 'convex-3',
    question: 'In convex optimization, every local minimum is:',
    options: [
      'Also a saddle point',
      'Also a global minimum',
      'Not necessarily optimal',
      'A local maximum',
    ],
    correctAnswer: 1,
    explanation:
      'Key property of convex optimization: every local minimum is automatically a global minimum. This is why convex problems are tractable.',
  },
  {
    id: 'convex-4',
    question: 'Which of these ML problems is NOT convex?',
    options: [
      'Linear regression with MSE loss',
      'Logistic regression with cross-entropy loss',
      'Training a deep neural network',
      'Support Vector Machine (SVM)',
    ],
    correctAnswer: 2,
    explanation:
      'Deep neural networks have non-convex loss surfaces due to the composition of non-linear activation functions. Linear regression, logistic regression, and SVMs are all convex.',
  },
  {
    id: 'convex-5',
    question: 'The KKT conditions for constrained optimization are:',
    options: [
      'Only necessary for optimality',
      'Only sufficient for optimality',
      'Both necessary and sufficient for convex problems',
      'Unrelated to convexity',
    ],
    correctAnswer: 2,
    explanation:
      'For convex optimization problems, KKT conditions are both necessary and sufficient for a point to be optimal. For non-convex problems, they are only necessary.',
  },
];
