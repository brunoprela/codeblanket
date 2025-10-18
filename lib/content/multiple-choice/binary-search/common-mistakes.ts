/**
 * Multiple choice questions for Common Pitfalls & How to Avoid Them section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const commonmistakesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the main problem with using mid = (left + right) // 2 in languages like Java or C++?',
    options: [
      'It is slower',
      'It can cause integer overflow',
      'It gives the wrong result',
      'It uses more memory',
    ],
    correctAnswer: 1,
    explanation:
      'When left and right are both very large, adding them together can exceed the maximum integer value, causing overflow. Use mid = left + (right - left) // 2 instead.',
  },
  {
    id: 'mc2',
    question: 'What can cause an infinite loop in binary search?',
    options: [
      'Using while left <= right',
      'Setting left = mid or right = mid instead of mid ± 1',
      'Calculating mid incorrectly',
      'Having duplicates in the array',
    ],
    correctAnswer: 1,
    explanation:
      'Setting left = mid or right = mid can cause infinite loops when the search space reduces to 2 elements. Always use left = mid + 1 and right = mid - 1 to properly exclude the checked middle element.',
  },
  {
    id: 'mc3',
    question:
      'What is the most important precondition for binary search to work correctly?',
    options: [
      'Array must have no duplicates',
      'Array must be sorted',
      'Array must be large',
      'Array must have unique elements',
    ],
    correctAnswer: 1,
    explanation:
      'Binary search absolutely requires the array to be sorted. Without sorting, comparisons with the middle element cannot reliably determine which half contains the target.',
  },
  {
    id: 'mc4',
    question:
      'If you must search an unsorted array many times, what is the best approach?',
    options: [
      'Use binary search directly',
      'Build a hash map once for O(1) lookups',
      'Sort before every search',
      'Always use linear search',
    ],
    correctAnswer: 1,
    explanation:
      'For multiple searches on the same data, build a hash map once (O(N)) to get O(1) average lookup time for each subsequent search. Sorting and binary search would be O(N log N) + O(log N) per search.',
  },
  {
    id: 'mc5',
    question:
      'What should you return from binary search when the target is found?',
    options: [
      'The value itself',
      'The index where it was found',
      'True',
      'The array',
    ],
    correctAnswer: 1,
    explanation:
      'Return the index (mid) where the target was found, not the value itself. The caller already knows the value (they provided it as the target), they need to know where it is.',
  },
];
