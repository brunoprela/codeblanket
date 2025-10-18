/**
 * Multiple choice questions for Introduction to Sliding Window section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the primary advantage of the sliding window technique?',
    options: [
      'It uses less memory',
      'It reduces time complexity from O(n²) or O(n×k) to O(n)',
      'It works on unsorted data',
      'It uses recursion',
    ],
    correctAnswer: 1,
    explanation:
      'The sliding window technique reduces time complexity by avoiding recalculation of overlapping subarrays. Instead of computing each window from scratch, it incrementally updates by adding/removing elements.',
  },
  {
    id: 'mc2',
    question: 'What are the two main types of sliding window patterns?',
    options: [
      'Fast and slow',
      'Fixed-size and variable-size',
      'Forward and backward',
      'Recursive and iterative',
    ],
    correctAnswer: 1,
    explanation:
      'The two main patterns are fixed-size (window size is constant k) and variable-size (window size adjusts based on conditions to optimize for longest/shortest sequences).',
  },
  {
    id: 'mc3',
    question: 'When should you consider using a sliding window?',
    options: [
      'When sorting an array',
      'When finding contiguous subarrays/substrings with specific properties',
      'When implementing binary search',
      'When building a tree structure',
    ],
    correctAnswer: 1,
    explanation:
      'Sliding window is ideal for problems involving contiguous sequences with keywords like "longest", "shortest", "maximum", or "minimum" with constraints on consecutive elements.',
  },
  {
    id: 'mc4',
    question:
      'How many times is each element processed in a sliding window algorithm?',
    options: [
      'Once',
      'At most twice (when entering and leaving the window)',
      'n times',
      'log n times',
    ],
    correctAnswer: 1,
    explanation:
      'Each element is processed at most twice: once when the right pointer includes it in the window, and once when the left pointer removes it. This is why sliding window achieves O(n) time.',
  },
  {
    id: 'mc5',
    question:
      'What data structure is commonly used with variable-size sliding windows?',
    options: [
      'Stack',
      'Queue',
      'Hash map or hash set to track window state',
      'Binary tree',
    ],
    correctAnswer: 2,
    explanation:
      'Variable-size windows often use hash maps or sets to track window state, such as character frequencies or checking for duplicates, enabling O(1) condition checks.',
  },
];
