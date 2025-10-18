/**
 * Multiple choice questions for Interview Strategy section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const interviewstrategyMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What keywords signal a math/geometry problem?',
    options: [
      'Data structures',
      'Prime, divisible, GCD, factorial, distance, area, angle, rotate, modulo',
      'Sort, search',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Math/geometry keywords: "prime", "divisible", "GCD/LCM", "factorial", "permutation/combination", "distance", "area", "angle", "rotate", "modulo", "digit". Suggest mathematical formulas over complex algorithms.',
  },
  {
    id: 'mc2',
    question: 'How do you approach a math problem in an interview?',
    options: [
      'Code immediately',
      'Look for patterns, try small examples, derive formula, check for closed-form solution',
      'Random',
      'Use hash map',
    ],
    correctAnswer: 1,
    explanation:
      'Approach: 1) Try small examples (n=1,2,3) to spot pattern, 2) Check if closed-form formula exists, 3) Consider mathematical properties (GCD, modulo), 4) Implement efficiently, 5) Handle overflow/precision.',
  },
  {
    id: 'mc3',
    question: 'What should you clarify in a math/geometry interview?',
    options: [
      'Nothing',
      'Integer overflow? Precision needed? Modulo required? Coordinate system? Edge cases?',
      'Random',
      'Language only',
    ],
    correctAnswer: 1,
    explanation:
      'Clarify: 1) Can numbers overflow (use long/modulo?), 2) Floating point precision concerns, 3) Return modulo 10^9+7?, 4) Coordinate system (axis-aligned?), 5) Edge cases (0, negative, max values).',
  },
  {
    id: 'mc4',
    question: 'What is a common math problem mistake?',
    options: [
      'Using formulas',
      'Integer overflow, floating point precision errors, off-by-one in formulas, not handling mod correctly',
      'Good naming',
      'Comments',
    ],
    correctAnswer: 1,
    explanation:
      'Common mistakes: 1) Integer overflow in factorial/combinations, 2) Floating point precision (use epsilon for comparisons), 3) Off-by-one in formulas, 4) Wrong mod application ((a-b)%m may be negative).',
  },
  {
    id: 'mc5',
    question: 'How should you communicate your math solution?',
    options: [
      'Just code',
      'Explain mathematical insight/formula, why it works, walk through example, mention edge cases',
      'No explanation',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Communication: 1) Key mathematical insight (pattern, formula), 2) Why formula is correct (derive if simple), 3) Walk through small example, 4) Edge cases (overflow, 0, negative), 5) Time/space complexity.',
  },
];
