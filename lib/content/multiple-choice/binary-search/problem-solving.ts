/**
 * Multiple choice questions for Problem-Solving Strategy & Interview Tips section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const problemsolvingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the first question you should ask in a binary search interview problem?',
    options: [
      'What is the array size?',
      'Is the array sorted?',
      'What programming language to use?',
      'Can I use extra space?',
    ],
    correctAnswer: 1,
    explanation:
      'The most critical question is whether the array is sorted, as binary search only works on sorted data. This is a fundamental precondition that must be verified.',
  },
  {
    id: 'mc2',
    question:
      'How long should you spend coding a medium binary search problem in an interview?',
    options: ['2-3 minutes', '5-7 minutes', '15-20 minutes', '1 minute'],
    correctAnswer: 1,
    explanation:
      'Plan for 5-7 minutes of careful coding. Accuracy is more important than speed. Rush jobs lead to bugs that waste time debugging.',
  },
  {
    id: 'mc3',
    question:
      'What is the recommended approach for a rotated sorted array problem?',
    options: [
      'Sort it first',
      'Use linear search',
      'Determine which half is sorted and decide accordingly',
      'Find rotation point first',
    ],
    correctAnswer: 2,
    explanation:
      'In a rotated sorted array, at least one half is always properly sorted. Compare mid with the edges to determine which half is sorted, then decide where to search based on the target.',
  },
  {
    id: 'mc4',
    question: 'What edge cases should you always test for binary search?',
    options: [
      'Only test the middle element',
      'Empty array, single element, boundaries, target not found',
      'Only test when target is found',
      'No need to test edge cases',
    ],
    correctAnswer: 1,
    explanation:
      'Always test: empty array, single element, target at first/last position, target not in array. These edge cases catch most bugs.',
  },
  {
    id: 'mc5',
    question: 'When debugging binary search, what is the first thing to check?',
    options: [
      'The array contents',
      'Print left, mid, right at each iteration to verify search space is shrinking',
      'Run it on larger inputs',
      'Change to linear search',
    ],
    correctAnswer: 1,
    explanation:
      'Print left, mid, right values at each iteration to verify the search space is properly shrinking. If pointers are not converging, you have a logic error.',
  },
];
