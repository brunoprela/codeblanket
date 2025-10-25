/**
 * Multiple choice questions for How to Analyze Code Complexity section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const analyzingcodeMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the time complexity of this code?\n\n```python\ndef func (arr):\n    for i in range (len (arr)):\n        for j in range (i):\n            print(arr[i], arr[j])\n```',
    options: ['O(n)', 'O(n log n)', 'O(n²)', 'O(2n)'],
    correctAnswer: 2,
    explanation:
      'This is O(n²). The inner loop runs 0 + 1 + 2 + ... + (n-1) times total, which equals n (n-1)/2. This simplifies to O(n²) after dropping constants and lower-order terms.',
  },
  {
    id: 'mc2',
    question:
      'What is the time complexity of calling sorted() on a Python list?',
    options: ['O(n)', 'O(n log n)', 'O(n²)', 'O(log n)'],
    correctAnswer: 1,
    explanation:
      "Python\'s sorted() function uses Timsort, which has O(n log n) time complexity in the average and worst case. This is an important built-in operation to remember.",
  },
  {
    id: 'mc3',
    question:
      'If you have a loop that runs n times and inside it you sort an array of size n, what is the total time complexity?',
    options: ['O(n)', 'O(n log n)', 'O(n²)', 'O(n² log n)'],
    correctAnswer: 3,
    explanation:
      'The loop runs n times, and each iteration sorts an array of size n (O(n log n)). We multiply these: O(n) × O(n log n) = O(n² log n). This is worse than O(n²)!',
  },
  {
    id: 'mc4',
    question:
      'What is the time complexity of this recursive Fibonacci?\n\n```python\ndef fib (n):\n    if n <= 1: return n\n    return fib (n-1) + fib (n-2)\n```',
    options: ['O(n)', 'O(n²)', 'O(2ⁿ)', 'O(log n)'],
    correctAnswer: 2,
    explanation:
      'This is O(2ⁿ) exponential time. Each call makes two recursive calls, creating a binary tree of depth n. The total number of calls is approximately 2ⁿ.',
  },
  {
    id: 'mc5',
    question: 'Which data structure provides O(1) average case lookup time?',
    options: ['Array', 'Linked List', 'Hash Table', 'Binary Search Tree'],
    correctAnswer: 2,
    explanation:
      'Hash tables (dictionaries/hash maps) provide O(1) average case lookup time. Arrays provide O(1) access by index, but O(n) search. Linked lists are O(n) for lookup. BSTs are O(log n) average case.',
  },
];
