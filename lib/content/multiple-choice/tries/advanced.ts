/**
 * Multiple choice questions for Advanced Trie Techniques section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const advancedMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is a Compressed Trie (Radix Tree)?',
    options: [
      'A Trie that uses compression algorithms',
      'A Trie that merges single-child chains into nodes storing edge labels (strings)',
      'A sorted Trie',
      'A hash-based Trie',
    ],
    correctAnswer: 1,
    explanation:
      'Compressed Trie (Radix Tree) merges chains of single-child nodes into one node, storing strings on edges instead of single characters. This saves space, using O(words) instead of O(total characters).',
  },
  {
    id: 'mc2',
    question: 'What does adding a count field to Trie nodes enable?',
    options: [
      'Faster searches',
      'Prefix frequency queries and word popularity tracking',
      'Automatic sorting',
      'Memory compression',
    ],
    correctAnswer: 1,
    explanation:
      'Count fields track how many words pass through each node. This enables: prefix frequency ("how many words start with X?"), word frequency, and popularity-based autocomplete suggestions.',
  },
  {
    id: 'mc3',
    question: 'What is a Suffix Trie used for?',
    options: [
      'Sorting strings',
      'Pattern matching by building a Trie of all suffixes',
      'Prefix searches only',
      'Deleting words',
    ],
    correctAnswer: 1,
    explanation:
      'Suffix Trie contains all suffixes of a text, enabling pattern matching. Check if pattern exists by searching the Trie. All pattern occurrences are found by collecting indices stored at matching nodes.',
  },
  {
    id: 'mc4',
    question: 'How does a binary XOR Trie maximize XOR?',
    options: [
      'By sorting numbers',
      'By traversing bits and trying to take the opposite bit at each level',
      'By using hash maps',
      'By random selection',
    ],
    correctAnswer: 1,
    explanation:
      'Binary XOR Trie stores numbers bit by bit. To find maximum XOR, greedily try the opposite bit at each level (0→try 1, 1→try 0). This maximizes XOR from the most significant bit down.',
  },
  {
    id: 'mc5',
    question: 'When is a Compressed Trie most beneficial?',
    options: [
      'Always',
      'When strings are long with few shared prefixes (sparse dictionary)',
      'When all strings are the same length',
      'Never',
    ],
    correctAnswer: 1,
    explanation:
      'Compressed Tries are most beneficial for sparse dictionaries with long strings and few shared prefixes. They save space by merging long chains of single-child nodes, reducing from O(total chars) to O(words).',
  },
];
