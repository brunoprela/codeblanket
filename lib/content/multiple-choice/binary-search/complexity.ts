/**
 * Multiple choice questions for Time & Space Complexity Analysis section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const complexityMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the time complexity of binary search?',
    options: ['O(N)', 'O(log N)', 'O(N log N)', 'O(1)'],
    correctAnswer: 1,
    explanation:
      'Binary search has O(log N) time complexity because it divides the search space in half with each comparison. For N elements, it takes at most log₂(N) comparisons.',
  },
  {
    id: 'mc2',
    question:
      'How many comparisons are needed to search 1 million elements with binary search?',
    options: ['1,000', 'About 20', '1,000,000', '100'],
    correctAnswer: 1,
    explanation:
      'Binary search needs at most log₂(1,000,000) ≈ 20 comparisons. This is because 2^20 = 1,048,576, which is just over 1 million.',
  },
  {
    id: 'mc3',
    question: 'What is the space complexity of iterative binary search?',
    options: ['O(log N)', 'O(1)', 'O(N)', 'O(N log N)'],
    correctAnswer: 1,
    explanation:
      'Iterative binary search uses O(1) constant space - only a few variables (left, right, mid) are needed regardless of input size.',
  },
  {
    id: 'mc4',
    question: 'What is the space complexity of recursive binary search?',
    options: ['O(1)', 'O(log N)', 'O(N)', 'O(N²)'],
    correctAnswer: 1,
    explanation:
      'Recursive binary search uses O(log N) space for the call stack. Each recursive call adds a frame to the stack, and there are at most log N calls.',
  },
  {
    id: 'mc5',
    question: 'What is the best case time complexity of binary search?',
    options: ['O(log N)', 'O(1)', 'O(N)', 'Best case does not exist'],
    correctAnswer: 1,
    explanation:
      'The best case is O(1) when the target happens to be exactly at the middle position on the first comparison. However, this is rare and average/worst cases are O(log N).',
  },
];
