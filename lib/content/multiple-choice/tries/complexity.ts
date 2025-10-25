/**
 * Multiple choice questions for Complexity Analysis section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const complexityMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the time complexity of searching for a word of length m in a Trie?',
    options: [
      'O(n) where n is number of words',
      'O(m) independent of number of words',
      'O(log n)',
      'O(m log n)',
    ],
    correctAnswer: 1,
    explanation:
      'Search takes O(m) time where m is word length. You traverse m nodes (one per character), independent of how many total words are stored in the Trie.',
  },
  {
    id: 'mc2',
    question: 'What is the worst-case space complexity of a Trie?',
    options: ['O(n)', 'O(ALPHABET_SIZE × m × n)', 'O(m)', 'O(log n)'],
    correctAnswer: 1,
    explanation:
      'Worst case (no shared prefixes): n words of length m, each node has ALPHABET_SIZE pointers. Total: ALPHABET_SIZE × m × n space. For English (26 letters), this can be substantial.',
  },
  {
    id: 'mc3',
    question: "What is Trie\'s advantage over Hash Tables?",
    options: [
      'Always uses less memory',
      'Efficient prefix search O(m) vs Hash Table O(n×m)',
      'Faster exact lookups',
      'Simpler implementation',
    ],
    correctAnswer: 1,
    explanation:
      'Trie excels at prefix operations: find all words starting with prefix in O(m + k) time. Hash Tables cannot efficiently do prefix searches and would need O(n×m) to check all n words.',
  },
  {
    id: 'mc4',
    question: 'When should you NOT use a Trie?',
    options: [
      'When you need prefix searches',
      'When only exact lookups are needed and space is critical',
      'When words share prefixes',
      'For autocomplete',
    ],
    correctAnswer: 1,
    explanation:
      "Don't use Trie when only exact lookups are needed (Hash Table is simpler and more space-efficient), space is critical, or there's no prefix sharing. Trie adds complexity without benefits in these cases.",
  },
  {
    id: 'mc5',
    question:
      'How does using HashMap for children nodes improve space efficiency?',
    options: [
      "It doesn't help",
      'Stores only actual children instead of fixed ALPHABET_SIZE array',
      'Makes searches faster',
      'Reduces time complexity',
    ],
    correctAnswer: 1,
    explanation:
      'HashMap stores only actual children (e.g., 2-3 nodes) instead of allocating space for all 26 possible letters. This saves significant space when nodes have few children, which is common.',
  },
];
