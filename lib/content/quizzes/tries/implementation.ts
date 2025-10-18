/**
 * Quiz questions for Trie Implementation section
 */

export const implementationQuiz = [
  {
    id: 'q1',
    question:
      'Compare HashMap vs Array for storing Trie children. When would you choose each?',
    sampleAnswer:
      'HashMap (dict) is flexible: handles any character set (Unicode, symbols), space efficient for sparse data (only stores existing children). Array is fixed: typically size 26 for lowercase letters, uses index = char - ord("a") for O(1) access. Choose HashMap when: character set is large or unknown, data is sparse (few children per node), need flexibility. Choose Array when: character set is small and fixed (lowercase letters), performance critical (array access faster than hash lookup), memory layout matters (cache-friendly). For English words, array is common. For international text or mixed characters, use HashMap. The tradeoff: HashMap flexibility vs Array speed and simplicity.',
    keyPoints: [
      'HashMap: flexible, any char set, space efficient when sparse',
      'Array: fixed size (26), O(1) guaranteed, cache-friendly',
      'HashMap when: large/unknown charset, sparse data',
      'Array when: fixed small charset, performance critical',
      'Tradeoff: flexibility vs speed',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the is_end_of_word flag. Why cannot we just check if node has no children?',
    sampleAnswer:
      'is_end_of_word marks complete words vs prefixes. Cannot check "no children" because words can be prefixes of other words. Example: insert "car" then "carpet". The node at R (end of "car") has children (p→e→t), but "car" is still a complete word. Without the flag, searching "car" would fail because R has children. The flag distinguishes: node in middle of word, node at end of word but also prefix of longer words, node at end of only word (leaf). When searching, we need node to exist AND be marked as end. The flag is independent of having children - a node can be both end of word and have children.',
    keyPoints: [
      'Marks complete words vs prefixes only',
      'Cannot use "no children" check',
      'Words can be prefixes: "car" and "carpet"',
      'Node can be end-of-word AND have children',
      'Search needs: node exists AND is_end_of_word',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through the autocomplete get_words_with_prefix implementation. How does DFS collect all words?',
    sampleAnswer:
      'Autocomplete has two phases. Phase 1: Navigate to prefix node. Traverse character by character from root. If any character missing, return empty list (no words with prefix). If reach end, we have the prefix subtree. Phase 2: DFS from prefix node to collect all words. Recursive DFS: if current node is_end_of_word, add accumulated path to results. Then recurse on all children, adding each child character to path. The DFS explores entire subtree under prefix node. For example, prefix "ca": navigate to node C→A, then DFS finds all paths: R (→car), R→P→E→T (→carpet), T (→cat). Return ["car", "carpet", "cat"]. The beauty: all words starting with prefix are in one subtree.',
    keyPoints: [
      'Phase 1: navigate to prefix node',
      'Phase 2: DFS to collect all words in subtree',
      'DFS: add word if is_end_of_word, recurse on children',
      'All words with prefix are in one subtree',
      'Example: "ca" → DFS finds car, carpet, cat',
    ],
  },
];
