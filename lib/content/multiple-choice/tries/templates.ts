/**
 * Multiple choice questions for Code Templates section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const templatesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What are the two essential components of the basic Trie template?',
    options: [
      'Array and list',
      'TrieNode (children, is_end_of_word) and Trie (root, methods)',
      'Stack and queue',
      'Hash and tree',
    ],
    correctAnswer: 1,
    explanation:
      'Basic Trie template has two classes: TrieNode with children (dict/array) and is_end_of_word flag, and Trie with root node and methods (insert, search, startsWith).',
  },
  {
    id: 'mc2',
    question:
      'What is the difference between search() and startsWith() in the Trie template?',
    options: [
      'They are the same',
      'search() checks is_end_of_word flag, startsWith() ignores it',
      'search() is faster',
      'startsWith() uses less memory',
    ],
    correctAnswer: 1,
    explanation:
      "search() returns true only if the exact word exists (is_end_of_word = true). startsWith() returns true if the prefix path exists, ignoring whether it's a complete word.",
  },
  {
    id: 'mc3',
    question: 'In the autocomplete template, what are the two phases?',
    options: [
      'Sort and filter',
      'Navigate to prefix node, then DFS to collect all words in subtree',
      'Insert and delete',
      'Hash and search',
    ],
    correctAnswer: 1,
    explanation:
      'Phase 1: Navigate to the prefix node character by character. Phase 2: DFS from that node to collect all words (all words with the prefix are in that subtree).',
  },
  {
    id: 'mc4',
    question: 'Why is Trie deletion more complex than insertion?',
    options: [
      "It's slower",
      'Must avoid breaking other words - cannot blindly remove nodes',
      'Uses more memory',
      'Requires sorting',
    ],
    correctAnswer: 1,
    explanation:
      'Deletion must preserve other words. Can only remove a node if it has no children and is not the end of another word. Must check bottom-up recursively to avoid breaking shared prefixes.',
  },
  {
    id: 'mc5',
    question: 'What does the is_end_of_word flag distinguish?',
    options: [
      'Parent from child',
      'Complete words from prefixes',
      'Letters from numbers',
      'Valid from invalid',
    ],
    correctAnswer: 1,
    explanation:
      'The is_end_of_word flag marks nodes where complete words end, distinguishing actual words from mere prefixes. Essential for search() to work correctly.',
  },
];
