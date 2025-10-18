/**
 * Multiple choice questions for Understanding Big O Notation section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const bigonotationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the time complexity of this code?\n\n```python\nfor i in range(n):\n    for j in range(n):\n        print(i, j)\n```',
    options: ['O(n)', 'O(n²)', 'O(2n)', 'O(n log n)'],
    correctAnswer: 1,
    explanation:
      'This is O(n²) because we have nested loops where both iterate n times. The outer loop runs n times, and for each iteration, the inner loop runs n times, resulting in n × n = n² total operations.',
  },
  {
    id: 'mc2',
    question:
      'Which of these has the best (fastest) time complexity for large inputs?',
    options: ['O(2ⁿ)', 'O(n²)', 'O(n log n)', 'O(n³)'],
    correctAnswer: 2,
    explanation:
      'O(n log n) is the fastest among these options. The hierarchy from best to worst is: O(n log n) < O(n²) < O(n³) < O(2ⁿ). Exponential time O(2ⁿ) grows extremely fast and is the slowest.',
  },
  {
    id: 'mc3',
    question: 'What is the time complexity of binary search on a sorted array?',
    options: ['O(1)', 'O(log n)', 'O(n)', 'O(n log n)'],
    correctAnswer: 1,
    explanation:
      'Binary search is O(log n) because it halves the search space with each comparison. After k comparisons, we have n/2^k elements left. When n/2^k = 1, we have k = log₂(n) comparisons.',
  },
  {
    id: 'mc4',
    question:
      'If we simplify O(3n² + 2n + 5) using Big O rules, what do we get?',
    options: ['O(3n²)', 'O(n² + n)', 'O(n²)', 'O(n)'],
    correctAnswer: 2,
    explanation:
      'We drop constants (3, 2, 5) and lower-order terms (2n, 5), keeping only the fastest-growing term. So O(3n² + 2n + 5) simplifies to O(n²).',
  },
  {
    id: 'mc5',
    question:
      'What is the time complexity of this code?\n\n```python\nfor i in range(n):\n    print(i)\nfor j in range(n):\n    print(j)\n```',
    options: ['O(n)', 'O(n²)', 'O(2n)', 'O(log n)'],
    correctAnswer: 0,
    explanation:
      'This is O(n). We have two sequential loops (not nested), each running n times. Total operations: n + n = 2n, which simplifies to O(n) after dropping the constant.',
  },
];
