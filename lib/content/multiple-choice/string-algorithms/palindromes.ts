/**
 * Multiple choice questions for Palindrome Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const palindromesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc-pal1',
    question:
      'What is the time complexity of checking if a string is a palindrome using two pointers?',
    options: ['O(log n)', 'O(n)', 'O(n log n)', 'O(n²)'],
    correctAnswer: 1,
    explanation:
      'Two pointers traverse the string from both ends meeting in the middle, comparing n/2 pairs of characters, which is O(n) time.',
  },
  {
    id: 'mc-pal2',
    question:
      'For finding all palindromic substrings, why is expand-around-center better than brute force?',
    options: [
      'It uses less space',
      'It reduces time complexity from O(n³) to O(n²)',
      'It only works for even-length palindromes',
      'It requires preprocessing',
    ],
    correctAnswer: 1,
    explanation:
      'Brute force checks all O(n²) substrings, each taking O(n) to verify (total O(n³)). Expand-around-center has O(n) centers and O(n) expansion per center, giving O(n²).',
  },
  {
    id: 'mc-pal3',
    question:
      'How many potential centers should you check for palindromes in a string of length n?',
    options: ['n centers', '2n-1 centers', 'n² centers', 'n/2 centers'],
    correctAnswer: 1,
    explanation:
      'You need to check n centers for odd-length palindromes (between characters) and n-1 centers for even-length (at characters), totaling 2n-1 centers.',
  },
  {
    id: 'mc-pal4',
    question:
      'What is the best approach to find the longest palindromic subsequence?',
    options: [
      'Two pointers',
      'Expand around center',
      'Dynamic Programming',
      'Sliding window',
    ],
    correctAnswer: 2,
    explanation:
      'Longest palindromic subsequence requires DP with O(n²) time and space, similar to Longest Common Subsequence between s and reverse(s). Two pointers and expand-around-center work for substrings, not subsequences.',
  },
  {
    id: 'mc-pal5',
    question:
      'When checking "A man, a plan, a canal: Panama" as a palindrome, what should you do?',
    options: [
      'Compare as-is',
      'Remove spaces only',
      'Remove spaces and punctuation, compare case-insensitive',
      'Reverse and compare',
    ],
    correctAnswer: 2,
    explanation:
      'For phrase palindromes, you must remove non-alphanumeric characters and compare case-insensitive: "amanaplanacanalpanama" reads the same forward and backward.',
  },
];
