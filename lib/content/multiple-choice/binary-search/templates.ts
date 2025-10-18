/**
 * Multiple choice questions for Code Templates & Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const templatesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'When finding the first occurrence of a target in an array with duplicates, what should you do after finding nums[mid] == target?',
    options: [
      'Return mid immediately',
      'Save mid as result and continue searching left (right = mid - 1)',
      'Set left = mid',
      'Break from the loop',
    ],
    correctAnswer: 1,
    explanation:
      'Save the match as a candidate result, but continue searching left by setting right = mid - 1 to find any earlier occurrences. This ensures you find the leftmost match.',
  },
  {
    id: 'mc2',
    question:
      'When finding the last occurrence of a target with duplicates, what should you do after finding nums[mid] == target?',
    options: [
      'Return mid immediately',
      'Save mid as result and continue searching right (left = mid + 1)',
      'Set right = mid',
      'Set both pointers to mid',
    ],
    correctAnswer: 1,
    explanation:
      'Save the match and continue searching right by setting left = mid + 1 to find any later occurrences. This ensures you find the rightmost match.',
  },
  {
    id: 'mc3',
    question:
      'In the "search insert position" template, what do you return if the target is not found?',
    options: ['-1', 'left (the insertion position)', 'right', 'mid'],
    correctAnswer: 1,
    explanation:
      'Return left, which naturally ends up at the correct position where the target should be inserted to maintain sorted order.',
  },
  {
    id: 'mc4',
    question:
      'When would you use the "find first" template instead of classic binary search?',
    options: [
      'When the array is unsorted',
      'When duplicates exist and you need the leftmost boundary',
      'When you want faster search',
      'When the array is empty',
    ],
    correctAnswer: 1,
    explanation:
      'Use "find first" when duplicates exist and you need to find the leftmost occurrence, such as for range queries or counting occurrences.',
  },
  {
    id: 'mc5',
    question:
      'Which template would you use to count occurrences of a value in a sorted array?',
    options: [
      'Classic binary search',
      'Find first and find last templates',
      'Search insert position',
      'Linear search',
    ],
    correctAnswer: 1,
    explanation:
      'To count occurrences, find the first occurrence and last occurrence, then calculate: last - first + 1. This requires both boundary-finding templates.',
  },
];
