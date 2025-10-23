/**
 * Multiple choice questions for Gradient & Directional Derivatives section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const gradientdirectionalderivativesMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'grad-dir-1',
      question: 'The directional derivative D_v f equals:',
      options: [
        'Just the gradient',
        '∇f · v̂ (dot product with unit direction)',
        'The magnitude of the gradient',
        'The second derivative',
      ],
      correctAnswer: 1,
      explanation:
        'The directional derivative equals the dot product of the gradient with the unit direction vector: D_v f = ∇f · v̂.',
    },
    {
      id: 'grad-dir-2',
      question: 'The gradient ∇f points in the direction of:',
      options: [
        'Steepest descent',
        'Steepest ascent (maximum increase)',
        'Zero change',
        'Any arbitrary direction',
      ],
      correctAnswer: 1,
      explanation:
        'The gradient points in the direction of maximum rate of increase (steepest ascent). We use -∇f for steepest descent.',
    },
    {
      id: 'grad-dir-3',
      question: 'What is the maximum directional derivative of f at point a?',
      options: [
        'Always 1',
        'The gradient magnitude ||∇f(a)||',
        'Infinity',
        'The minimum eigenvalue',
      ],
      correctAnswer: 1,
      explanation:
        'The maximum directional derivative equals the gradient magnitude. This maximum is achieved when moving in the gradient direction.',
    },
    {
      id: 'grad-dir-4',
      question: 'In gradient descent with momentum, the velocity accumulates:',
      options: [
        'Only the current gradient',
        'Exponential moving average of past gradients',
        'The sum of all gradients',
        'Random directions',
      ],
      correctAnswer: 1,
      explanation:
        'Momentum maintains an exponential moving average of past gradients: v = βv + (1-β)∇L, which helps accelerate convergence.',
    },
    {
      id: 'grad-dir-5',
      question:
        'For constrained optimization on a manifold, projected gradient descent:',
      options: [
        'Ignores constraints',
        'Takes gradient step then projects onto constraint set',
        'Only moves along constraints',
        "Doesn't use gradients",
      ],
      correctAnswer: 1,
      explanation:
        'Projected GD alternates between: (1) taking a gradient step, (2) projecting back onto the constraint set. This ensures constraints are satisfied.',
    },
  ];
