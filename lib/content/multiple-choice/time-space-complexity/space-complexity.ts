/**
 * Multiple choice questions for Space Complexity Analysis section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const spacecomplexityMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the space complexity of this function?\n\n```python\ndef sum_array (arr):\n    total = 0\n    for num in arr:\n        total += num\n    return total\n```',
    options: ['O(n)', 'O(1)', 'O(log n)', 'O(n²)'],
    correctAnswer: 1,
    explanation:
      'This is O(1) space. We only use a constant amount of extra space (the variable "total") regardless of the input array size. The input array itself is not counted in auxiliary space complexity.',
  },
  {
    id: 'mc2',
    question:
      'What is the space complexity of a recursive function that has a maximum call stack depth of n?',
    options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
    correctAnswer: 2,
    explanation:
      'The space complexity is O(n). Each recursive call adds a frame to the call stack, and if the maximum depth is n, we need O(n) space to store all those stack frames in memory.',
  },
  {
    id: 'mc3',
    question: 'Which statement about space complexity is TRUE?',
    options: [
      'Space complexity always equals time complexity',
      'Recursive functions always use O(1) space',
      'Creating a hash map with n entries uses O(n) space',
      'Space complexity is less important than time complexity',
    ],
    correctAnswer: 2,
    explanation:
      'Creating a hash map with n entries uses O(n) space. Space and time complexity are independent, recursive functions typically use O(depth) stack space, and space complexity is often just as important as time complexity.',
  },
  {
    id: 'mc4',
    question:
      'What is the typical space complexity of merge sort (not in-place)?',
    options: ['O(1)', 'O(log n)', 'O(n)', 'O(n log n)'],
    correctAnswer: 2,
    explanation:
      'Merge sort uses O(n) space for the temporary arrays needed during the merge step. While the recursion depth is O(log n), the dominant space factor is the O(n) auxiliary arrays.',
  },
  {
    id: 'mc5',
    question:
      'In a time-space tradeoff, what does it mean to "trade space for time"?',
    options: [
      'Use less memory to run faster',
      'Use more memory to run faster',
      'Use less memory but run slower',
      'Keep both time and space the same',
    ],
    correctAnswer: 1,
    explanation:
      'Trading space for time means using more memory (extra data structures like caches, hash maps, etc.) to achieve faster execution. A classic example is memoization, where we store computed results to avoid recalculating them.',
  },
];
