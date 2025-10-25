/**
 * Multiple choice questions for Functions & Relations section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const functionsrelationsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1-function-basics',
    question: 'For a function to be valid, which property must hold?',
    options: [
      'Every input must have at least two outputs',
      'Every input must have exactly one output',
      'Every output must have exactly one input',
      'The domain must equal the range',
    ],
    correctAnswer: 1,
    explanation:
      'A function must assign exactly ONE output to each input. This is the fundamental definition of a function. However, multiple inputs can map to the same output (many-to-one is allowed), but one input cannot map to multiple outputs.',
  },
  {
    id: 'mc2-composition',
    question: 'If f (x) = x + 3 and g (x) = 2x, what is (f ∘ g)(4)?',
    options: ['11', '14', '8', '10'],
    correctAnswer: 0,
    explanation:
      '(f ∘ g)(4) = f (g(4)). First apply g: g(4) = 2(4) = 8. Then apply f: f(8) = 8 + 3 = 11. Note: This is different from (g ∘ f)(4) = g (f(4)) = g(7) = 14.',
  },
  {
    id: 'mc3-sigmoid',
    question: 'What is the range of the sigmoid function σ(x) = 1/(1 + e⁻ˣ)?',
    options: ['(-∞, ∞)', '[-1, 1]', '(0, 1)', '[0, 1]'],
    correctAnswer: 2,
    explanation:
      'The sigmoid function has range (0, 1) - open interval, meaning it approaches but never reaches 0 or 1. As x → -∞, σ(x) → 0 and as x → ∞, σ(x) → 1. This makes sigmoid useful for binary classification where outputs are interpreted as probabilities.',
  },
  {
    id: 'mc4-inverse',
    question: 'If f (x) = 3x - 6, what is the inverse function f⁻¹(x)?',
    options: [
      'f⁻¹(x) = (x + 6)/3',
      'f⁻¹(x) = (x - 6)/3',
      'f⁻¹(x) = 3x + 6',
      'f⁻¹(x) = x/3 + 6',
    ],
    correctAnswer: 0,
    explanation:
      'To find inverse: Start with y = 3x - 6, solve for x: y + 6 = 3x, x = (y + 6)/3. Replace y with x: f⁻¹(x) = (x + 6)/3. Verify: f (f⁻¹(x)) = 3((x+6)/3) - 6 = x + 6 - 6 = x ✓',
  },
  {
    id: 'mc5-neural-network',
    question:
      'In a neural network with 3 layers (input → hidden → output), if each layer applies linear transformation followed by ReLU activation (except output), how many function compositions are involved?',
    options: [
      '3 (one per layer)',
      '5 (linear + activation for hidden, linear for output)',
      '6 (linear + activation for each layer)',
      '2 (just hidden and output)',
    ],
    correctAnswer: 1,
    explanation:
      'The computation is: output = linear2(ReLU(linear1(input))). This involves 3 function compositions: (1) linear1 (input to hidden), (2) ReLU (activation), (3) linear2 (hidden to output). Total: 3 functions composed. However, considering each operation separately: input→linear1→ReLU→linear2→output = 3 transformations, but 5 function applications if counting start point.',
  },
];
