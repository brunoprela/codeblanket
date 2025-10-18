/**
 * Multiple choice questions for Time & Space Complexity Analysis section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const complexityMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the time complexity of the two pointers technique?',
    options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
    correctAnswer: 2,
    explanation:
      'The two pointers technique is O(n) because each pointer traverses the array at most once. Even with two pointers, the combined movement is linear, not quadratic.',
  },
  {
    id: 'mc2',
    question: 'What is the space complexity of most two-pointer algorithms?',
    options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
    correctAnswer: 0,
    explanation:
      'Most two-pointer algorithms use O(1) constant space as they only require a few pointer variables and often modify arrays in-place without additional data structures.',
  },
  {
    id: 'mc3',
    question:
      'How does the time complexity change if you need to sort the array first before using two pointers?',
    options: [
      'Stays O(n)',
      'Becomes O(n log n)',
      'Becomes O(n²)',
      'Becomes O(log n)',
    ],
    correctAnswer: 1,
    explanation:
      'If sorting is required first, the overall time complexity becomes O(n log n) because sorting dominates. The two pointers part is still O(n), but O(n log n) + O(n) = O(n log n).',
  },
  {
    id: 'mc4',
    question: 'Why is two pointers more cache-friendly than nested loops?',
    options: [
      'It uses less memory',
      'It accesses elements sequentially',
      'It runs faster on all computers',
      'It uses recursion',
    ],
    correctAnswer: 1,
    explanation:
      'Two pointers accesses array elements sequentially, which is cache-friendly because it takes advantage of spatial locality. Nested loops may jump around more, leading to more cache misses.',
  },
  {
    id: 'mc5',
    question:
      'When comparing two pointers vs hash map for the two sum problem, what is the key trade-off?',
    options: [
      'Time vs readability',
      'Space (O(1) vs O(n)) vs need for sorted input',
      'Speed vs accuracy',
      'Complexity vs simplicity',
    ],
    correctAnswer: 1,
    explanation:
      'Two pointers uses O(1) space but requires sorted input, while hash map uses O(n) space but works on unsorted input. The trade-off is space efficiency versus the requirement to sort.',
  },
];
