/**
 * Multiple choice questions for Common Trie Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const patternsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'How does a Trie improve word search efficiency compared to checking each word individually?',
    options: [
      'Tries are always faster',
      'Build once in O(N×M), then each search is O(M) regardless of N words',
      'Tries use less memory',
      "They don't improve efficiency",
    ],
    correctAnswer: 1,
    explanation:
      'Build the Trie once with all N words (O(N×M) time). Then each search takes only O(M) time where M is word length, independent of N. Shared prefixes avoid redundant checks.',
  },
  {
    id: 'mc2',
    question: 'In the word break pattern, how does a Trie help?',
    options: [
      'Tries break words automatically',
      'Trie validates if substrings are valid dictionary words in O(M) time',
      'Tries sort the words',
      'Tries merge words',
    ],
    correctAnswer: 1,
    explanation:
      'In word break DP, the Trie efficiently validates whether substrings are valid words. Traverse the Trie from each position to find all valid words, enabling O(M) validation per substring.',
  },
  {
    id: 'mc3',
    question: 'What is the key advantage of using Tries for autocomplete?',
    options: [
      'Faster typing',
      'All words with a prefix are in one subtree, collectible with DFS',
      'Uses less memory',
      'Sorts automatically',
    ],
    correctAnswer: 1,
    explanation:
      'After navigating to the prefix node, all words starting with that prefix are in that subtree. A simple DFS collects all of them, making autocomplete very efficient.',
  },
  {
    id: 'mc4',
    question: 'How does a Trie with count nodes help count prefixes?',
    options: [
      'Counts characters',
      'Each node stores count of words passing through it',
      'Counts leaves',
      'Uses hash maps',
    ],
    correctAnswer: 1,
    explanation:
      'Augmenting nodes with a count field that increments during insertion allows O(M) queries for "how many words start with this prefix?" Just navigate to the prefix and return the count.',
  },
  {
    id: 'mc5',
    question: 'Why are Tries used in IP routing for longest prefix matching?',
    options: [
      'IPs are strings',
      'Trie naturally finds longest match by traversing as deep as possible',
      'Tries compress data',
      'Random access',
    ],
    correctAnswer: 1,
    explanation:
      'Build a Trie of IP prefixes (treating bits as characters). When routing, traverse as deep as possible in the Trie while tracking the last valid route seen. This naturally finds the longest matching prefix.',
  },
];
