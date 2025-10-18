/**
 * Multiple choice questions for Chain Rule for Multiple Variables section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const chainrulemultivariableMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'chain-multi-1',
    question: 'For z = f(x,y) where x = g(t), y = h(t), what is dz/dt?',
    options: [
      '(∂f/∂x) + (∂f/∂y)',
      '(∂f/∂x)(dx/dt) + (∂f/∂y)(dy/dt)',
      '(∂f/∂x)(∂f/∂y)',
      'dx/dt + dy/dt',
    ],
    correctAnswer: 1,
    explanation:
      'Multivariable chain rule sums contributions from all paths: dz/dt = (∂f/∂x)(dx/dt) + (∂f/∂y)(dy/dt).',
    difficulty: 'medium',
  },
  {
    id: 'chain-multi-2',
    question: 'Backpropagation in neural networks is:',
    options: [
      'Random weight updates',
      'Repeated application of the chain rule',
      'Only forward propagation',
      'A heuristic without mathematical foundation',
    ],
    correctAnswer: 1,
    explanation:
      'Backpropagation is the systematic application of the chain rule to compute gradients through composed functions (layers).',
    difficulty: 'easy',
  },
  {
    id: 'chain-multi-3',
    question: 'In a computational graph, gradients flow:',
    options: [
      'Only forward',
      'Backward from output to inputs via chain rule',
      'Randomly',
      'Only to the first layer',
    ],
    correctAnswer: 1,
    explanation:
      'Gradients flow backward through the computational graph, with each node applying the chain rule to propagate gradients to its inputs.',
    difficulty: 'easy',
  },
  {
    id: 'chain-multi-4',
    question:
      'For a 3-layer network X→Layer1→Layer2→Layer3→Loss, how many chain rule applications are needed for ∂L/∂W1?',
    options: ['1', '2', '3 (through all layers)', '0'],
    correctAnswer: 2,
    explanation:
      'Computing ∂L/∂W1 requires chain rule through all layers: Layer3→Layer2→Layer1, three applications total.',
    difficulty: 'medium',
  },
  {
    id: 'chain-multi-5',
    question: 'The Jacobian matrix in the chain rule represents:',
    options: [
      'All partial derivatives of outputs with respect to inputs',
      'Only first derivatives',
      'Second derivatives',
      'The loss function',
    ],
    correctAnswer: 0,
    explanation:
      'The Jacobian matrix contains all first-order partial derivatives, capturing how each output component depends on each input component.',
    difficulty: 'medium',
  },
];
