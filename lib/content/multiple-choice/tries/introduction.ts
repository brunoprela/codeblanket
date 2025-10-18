/**
 * Multiple choice questions for Introduction to Tries section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is a Trie primarily used for?',
    options: [
      'Sorting numbers',
      'Efficient string storage and prefix-based searches',
      'Matrix operations',
      'Graph traversal',
    ],
    correctAnswer: 1,
    explanation:
      'A Trie (prefix tree) is specifically designed for efficient string storage and prefix-based operations like autocomplete, spell checking, and dictionary lookups.',
  },
  {
    id: 'mc2',
    question: 'What is the time complexity of searching for a word in a Trie?',
    options: [
      'O(N) where N is number of words',
      'O(L) where L is the word length',
      'O(log N)',
      'O(N²)',
    ],
    correctAnswer: 1,
    explanation:
      'Searching in a Trie takes O(L) time where L is the length of the word being searched. This is independent of the total number of words stored in the Trie.',
  },
  {
    id: 'mc3',
    question:
      'How do Tries handle words with common prefixes like "cat" and "car"?',
    options: [
      'Store them separately',
      'Share the common prefix "ca" to save space',
      'Use hash tables',
      'Compress all letters',
    ],
    correctAnswer: 1,
    explanation:
      'Tries share common prefixes efficiently. Words "cat" and "car" share the path "c→a" and then branch at node "a" with children "t" and "r". This is a key space advantage.',
  },
  {
    id: 'mc4',
    question: 'What is a major disadvantage of Tries?',
    options: [
      'Slow search time',
      'Space overhead - each node needs storage for all possible children',
      'Cannot handle strings',
      'Limited to small datasets',
    ],
    correctAnswer: 1,
    explanation:
      'Tries can have significant space overhead because each node needs storage for all possible children (e.g., 26 pointers for lowercase English), even if most are unused.',
  },
  {
    id: 'mc5',
    question: 'Why is autocomplete a perfect use case for Tries?',
    options: [
      'Tries are fast at sorting',
      'After traversing to prefix node, can collect all words in that subtree',
      'Tries use hash tables',
      'Autocomplete is random',
    ],
    correctAnswer: 1,
    explanation:
      'For autocomplete, traverse to the prefix node in O(L) time, then collect all words in that subtree - these are all words starting with the prefix. Perfect for type-ahead suggestions.',
  },
];
