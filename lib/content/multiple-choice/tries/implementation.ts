/**
 * Multiple choice questions for Trie Implementation section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const implementationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What data structure is typically used to store children in a TrieNode?',
    options: [
      'Array or hash map (dictionary)',
      'Linked list',
      'Stack',
      'Queue',
    ],
    correctAnswer: 0,
    explanation:
      'Children are typically stored in an array (for fixed alphabets like a-z) or hash map/dictionary (for variable characters). Arrays give O(1) access, hash maps handle any character set.',
  },
  {
    id: 'mc2',
    question: 'What does the is_end_of_word flag indicate in a TrieNode?',
    options: [
      'The node is a leaf',
      'A complete word ends at this node',
      'The node has no children',
      'The Trie is full',
    ],
    correctAnswer: 1,
    explanation:
      'The is_end_of_word flag marks nodes where complete words end. This distinguishes actual words from prefixes. For example, if "car" and "carpet" are inserted, the "r" in "car" is marked as end of word.',
  },
  {
    id: 'mc3',
    question:
      'What is the time complexity of inserting a word of length L into a Trie?',
    options: ['O(N)', 'O(L)', 'O(log N)', 'O(L²)'],
    correctAnswer: 1,
    explanation:
      'Inserting a word takes O(L) time where L is the word length. You traverse or create one node per character, independent of how many words are already in the Trie.',
  },
  {
    id: 'mc4',
    question: 'In the autocomplete implementation, what are the two phases?',
    options: [
      'Sort and search',
      'Navigate to prefix node, then DFS to collect all words in that subtree',
      'Insert and delete',
      'Hash and compare',
    ],
    correctAnswer: 1,
    explanation:
      'Autocomplete: 1) Navigate to the prefix node character by character, 2) DFS from that node to collect all words in the subtree. All words with the prefix are in that subtree.',
  },
  {
    id: 'mc5',
    question:
      'When deleting a word from a Trie, when should you delete a node?',
    options: [
      'Always delete all nodes',
      "Only if it's not end of another word and has no other children",
      'Never delete nodes',
      "Delete if it's a leaf",
    ],
    correctAnswer: 1,
    explanation:
      "Delete a node only if it's not the end of another word and has no other children. This preserves other words sharing the prefix. Deletion backtracks from the end of the word.",
  },
];
