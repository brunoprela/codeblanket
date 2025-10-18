/**
 * Multiple choice questions for Problem-Solving Strategy & Interview Tips section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const strategyMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the first thing you should check to recognize a two-pointer problem?',
    options: [
      'If the array is small',
      'If the data is sorted or can be sorted',
      'If recursion is needed',
      'If a hash map would work',
    ],
    correctAnswer: 1,
    explanation:
      'The first signal for two pointers is whether the data is sorted or can be sorted. Sorted data enables predictable pointer movement decisions based on comparisons.',
  },
  {
    id: 'mc2',
    question:
      'Which keywords in a problem statement suggest using two pointers?',
    options: [
      'Recursive, stack, tree',
      'Pair, two numbers, remove, partition, in-place',
      'Hash, frequency, count',
      'Binary, search, logarithmic',
    ],
    correctAnswer: 1,
    explanation:
      'Keywords like "pair", "two numbers", "remove", "partition", and "in-place" strongly suggest two pointers. These indicate pair-finding or in-place modification patterns.',
  },
  {
    id: 'mc3',
    question: 'What is a common mistake when implementing two pointers?',
    options: [
      'Using too much memory',
      'Incorrect loop termination conditions (e.g., left <= right vs left < right)',
      'Making it too fast',
      'Not using recursion',
    ],
    correctAnswer: 1,
    explanation:
      'A common mistake is incorrect loop termination conditions, like using left <= right when you should use left < right, or vice versa. This can cause infinite loops or missed cases.',
  },
  {
    id: 'mc4',
    question:
      'How much time should you spend planning before coding a two-pointer solution in an interview?',
    options: [
      'No time, start coding immediately',
      '2-3 minutes',
      '10-15 minutes',
      'Half the interview time',
    ],
    correctAnswer: 1,
    explanation:
      'You should spend 2-3 minutes planning your approach, choosing the right pattern, and thinking through edge cases before writing code. This prevents costly mistakes.',
  },
  {
    id: 'mc5',
    question:
      'What edge cases should you always test with two-pointer solutions?',
    options: [
      'Only test with large inputs',
      'Empty array, single element, all same values, target at boundaries',
      'Only test the happy path',
      'Only test when there is a bug',
    ],
    correctAnswer: 1,
    explanation:
      "Always test edge cases: empty array (what if there's no input?), single element (do pointers work?), all same values (duplicates), and targets at boundaries (first/last elements).",
  },
];
