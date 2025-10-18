/**
 * Multiple choice questions for Anagram Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const anagramsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc-ana1',
    question:
      'What is the time complexity of checking if two strings are anagrams using a frequency counter?',
    options: ['O(1)', 'O(n)', 'O(n log n)', 'O(n²)'],
    correctAnswer: 1,
    explanation:
      'Using a hash map or array to count character frequencies requires O(n+m) time where n and m are string lengths. This simplifies to O(n).',
  },
  {
    id: 'mc-ana2',
    question: 'Which data structure is most efficient for grouping anagrams?',
    options: [
      'Array',
      'Hash Map (key = sorted string)',
      'Binary Search Tree',
      'Linked List',
    ],
    correctAnswer: 1,
    explanation:
      'A hash map with sorted string as key groups anagrams in O(n * k log k) time where n is number of strings and k is average string length. All anagrams map to the same key.',
  },
  {
    id: 'mc-ana3',
    question:
      'For finding all anagram starting indices in a string, what is the optimal approach?',
    options: [
      'Check every substring individually',
      'Use a sliding window with frequency counter',
      'Sort all substrings and compare',
      'Use dynamic programming',
    ],
    correctAnswer: 1,
    explanation:
      'Sliding window with frequency counter achieves O(n) time by maintaining character counts as the window slides, avoiding redundant comparisons.',
  },
  {
    id: 'mc-ana4',
    question:
      'What is the space complexity of checking anagrams with a character frequency array for lowercase letters only?',
    options: ['O(1)', 'O(n)', 'O(k) where k is unique characters', 'O(26)'],
    correctAnswer: 0,
    explanation:
      'For lowercase letters only, we use a fixed 26-element array regardless of input size, which is O(1) space.',
  },
  {
    id: 'mc-ana5',
    question:
      'When comparing two strings as anagrams, what must be checked first?',
    options: [
      'If they start with the same character',
      'If they have the same length',
      'If they are already sorted',
      'If they contain only lowercase letters',
    ],
    correctAnswer: 1,
    explanation:
      'Strings of different lengths cannot be anagrams. This is an O(1) check that should be done first to avoid unnecessary processing.',
  },
];
