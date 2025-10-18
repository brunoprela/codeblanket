/**
 * Quiz questions for Introduction to Tries section
 */

export const introductionQuiz = [
  {
    id: 'q1',
    question:
      'Explain what a Trie is and how it differs from other tree structures. What makes it special for string operations?',
    sampleAnswer:
      'A Trie (prefix tree) is a tree where each node represents a character and paths from root to nodes spell words. Unlike binary trees, each node can have up to 26 children (for lowercase English). Special for strings because it stores common prefixes once - if you have "cat" and "car", they share "ca". Each edge represents adding one character. Searching for word takes O(L) where L is word length, independent of how many words stored. Binary search tree would take O(L log N) where N is number of words. Trie trades space for time: uses more memory (26 pointers per node even if unused) but provides fast prefix operations. Perfect for autocomplete, spell check, IP routing.',
    keyPoints: [
      'Tree where nodes represent characters',
      'Paths spell words, shares common prefixes',
      'Search: O(L) independent of word count',
      'vs BST: O(L log N)',
      'Use cases: autocomplete, spell check, prefix search',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through inserting a word into a Trie. What happens with overlapping prefixes?',
    sampleAnswer:
      'Start at root. For each character in word: check if child node for that character exists. If yes, move to that child. If no, create new child node for that character. At end of word, mark node as end-of-word. For overlapping prefixes: if inserting "cat" then "car", first insert creates nodes C→A→T with T marked as end. Second insert reuses C→A, creates R branch at A, marks R as end. The A node now has two children: T and R. This is key efficiency: shared prefix "ca" stored once, not duplicated. Each insert is O(L) where L is word length. The end-of-word marker distinguishes complete words from prefixes.',
    keyPoints: [
      'For each char: use existing child or create new',
      'Mark last node as end-of-word',
      'Overlapping prefixes: reuse existing nodes',
      'Example: "cat" then "car" shares "ca"',
      'O(L) per insert',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe common Trie applications. Why is autocomplete a perfect use case?',
    sampleAnswer:
      'Autocomplete is perfect for Tries because after user types prefix, we just traverse to that prefix node, then DFS/BFS to collect all words in that subtree - all words starting with prefix. Dictionary operations: check if word exists in O(L) time. Spell check: find words within edit distance using Trie traversal with modifications. IP routing: store IP prefixes, match longest prefix efficiently. Word break: check if segments are valid words. Phone T9: map number sequences to possible words. The pattern: problems involving prefix matching, word validation, or collecting words with common prefix. Trie excels when you need fast prefix operations on large dictionary.',
    keyPoints: [
      'Autocomplete: traverse to prefix, collect subtree',
      'Dictionary: O(L) word lookup',
      'Spell check: edit distance traversal',
      'IP routing: longest prefix match',
      'Perfect for: prefix operations on large dictionary',
    ],
  },
];
