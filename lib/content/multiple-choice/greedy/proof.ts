/**
 * Multiple choice questions for Proving Greedy Correctness section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const proofMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the exchange argument for proving greedy correctness?',
    options: [
      'Test all cases',
      'Show any non-greedy solution can be converted to greedy without losing quality',
      'Use induction',
      'Random testing',
    ],
    correctAnswer: 1,
    explanation:
      "Exchange argument: take any optimal solution, replace non-greedy choices with greedy ones. If quality doesn't decrease, greedy is optimal. Shows greedy choice is always safe.",
  },
  {
    id: 'mc2',
    question: 'What does "greedy stays ahead" mean?',
    options: [
      'Greedy is faster',
      'At each step, greedy maintains ≥ quality as any other solution',
      'Greedy uses more memory',
      'Random property',
    ],
    correctAnswer: 1,
    explanation:
      'Greedy stays ahead: after k steps, greedy solution has quality ≥ any other solution with same resources. Proves greedy remains optimal throughout.',
  },
  {
    id: 'mc3',
    question: 'Why does greedy fail for 0/1 knapsack?',
    options: [
      'Too slow',
      "Cannot take fractions - greedy by ratio doesn't guarantee optimal",
      'Uses too much memory',
      'Cannot handle knapsacks',
    ],
    correctAnswer: 1,
    explanation:
      '0/1 knapsack: must take whole items. Greedy by ratio fails: might take high-ratio small item instead of optimal large item. Need DP to try all combinations.',
  },
  {
    id: 'mc4',
    question: 'What is structural induction for proving greedy?',
    options: [
      'Test examples',
      'Base case + show greedy choice maintains optimality for n+1',
      'Random testing',
      'Brute force',
    ],
    correctAnswer: 1,
    explanation:
      'Structural induction: prove greedy optimal for base case, then show adding greedy choice maintains optimality for larger inputs. Builds proof incrementally.',
  },
  {
    id: 'mc5',
    question: "How do you know when greedy won't work?",
    options: [
      'Always test first',
      'Cannot prove with exchange/stays-ahead arguments - need DP instead',
      'Random guess',
      'Greedy always works',
    ],
    correctAnswer: 1,
    explanation:
      'If you cannot prove greedy correctness with exchange argument or stays-ahead, greedy likely fails. Use DP for those problems (0/1 knapsack, longest path, etc.).',
  },
];
