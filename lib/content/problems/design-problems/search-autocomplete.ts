/**
 * Design Search Autocomplete System
 * Problem ID: search-autocomplete
 * Order: 9
 */

import { Problem } from '../../../types';

export const search_autocompleteProblem: Problem = {
  id: 'search-autocomplete',
  title: 'Design Search Autocomplete System',
  difficulty: 'Hard',
  topic: 'Design Problems',
  description: `Design a search autocomplete system for a search engine. Users may input a sentence (at least one word and end with a special character \`'#'\`).

You are given a string array \`sentences\` and an integer array \`times\` both of length \`n\` where \`sentences[i]\` is a previously typed sentence and \`times[i]\` is the corresponding number of times the sentence was typed. For each input character except \`'#'\`, return the top \`3\` historical hot sentences that have the same prefix as the part of the sentence already typed.

Here are the specific rules:

- The hot degree for a sentence is defined as the number of times a user typed the exactly same sentence before.
- The returned top \`3\` hot sentences should be sorted by hot degree (The first is the hottest one). If several sentences have the same hot degree, use ASCII-code order (smaller one appears first).
- If less than 3 hot sentences exist, return as many as you can.
- When the input is a special character, it means the sentence ends, and in this case, you need to return an empty list.

Implement the \`AutocompleteSystem\` class:

- \`AutocompleteSystem(String[] sentences, int[] times)\` Initializes the object with the \`sentences\` and \`times\` arrays.
- \`List<String> input(char c)\` This indicates that the user typed the character \`c\`.
  - Returns an empty array \`[]\` if \`c == '#'\`.
  - Otherwise, returns the top \`3\` historical hot sentences that have the same prefix as the part of the sentence already typed. If less than 3 exist, return as many as you can.`,
  hints: [
    'Trie for prefix search - O(p) where p = prefix length',
    'Store frequency at each end-of-word node',
    'DFS to collect all completions from current prefix',
    'Sort by frequency (desc), then lexicographically',
    'Optimization: Precompute top K at each Trie node',
  ],
  approach: `## Intuition

Autocomplete needs **prefix search** - perfect for **Trie**!

Each input character extends the prefix. We need to:
1. Navigate Trie to prefix
2. Find all sentences with that prefix
3. Return top 3 by frequency

---

## Approach: Trie + Frequency Tracking

**Trie Structure:**
\`\`\`
class TrieNode:
    children: dict[char -> TrieNode]
    is_end: bool
    frequency: int  # How many times this sentence was completed
    sentence: str   # Full sentence (stored at end node)
\`\`\`

### Operations:

**input(c):**1. If c == '#':
   - End of sentence, save it with freq+1
   - Reset current input
   - Return []
2. Else:
   - Append c to current input
   - Navigate Trie to prefix
   - DFS to collect all completions
   - Sort by (frequency desc, lexicographical)
   - Return top 3

**Example:**
\`\`\`
Sentences: ["i love you", "island", "ironman"], times: [5, 3, 2]

input('i'):
  Navigate to 'i', collect all:
  - "i love you" (freq 5)
  - "island" (freq 3)
  - "ironman" (freq 2)
  Return ["i love you", "island", "ironman"]

input(' '):
  Navigate to 'i ', collect:
  - "i love you" (freq 5)
  Return ["i love you"]

input('a'):
  Navigate to 'i a', no matches
  Return []
\`\`\`

---

## Optimization: Precompute Top K

Instead of DFS + sort on every input, precompute top K at each node during insertion.

**Trade-off**: Slower insert, faster query.

---

## Time Complexity:
- Without optimization: O(p + m log m) where p=prefix, m=matching sentences
- With optimization: O(p) - just navigate and return cached

## Space Complexity: O(N * L) where N=sentences, L=avg length`,
  testCases: [
    {
      input: [
        [
          'AutocompleteSystem',
          ['i love you', 'island', 'iroman', 'i love leetcode'],
          [5, 3, 2, 2],
        ],
        ['input', 'i'],
        ['input', ' '],
        ['input', 'a'],
        ['input', '#'],
      ],
      expected: [
        null,
        ['i love you', 'island', 'i love leetcode'],
        ['i love you', 'i love leetcode'],
        [],
        [],
      ],
    },
  ],
  solution: `class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.frequency = 0
        self.sentence = ""

class AutocompleteSystem:
    def __init__(self, sentences: list[str], times: list[int]):
        self.root = TrieNode()
        self.current_input = ""
        
        # Build trie with initial data
        for sentence, freq in zip(sentences, times):
            self._add_to_trie(sentence, freq)
    
    def _add_to_trie(self, sentence: str, frequency: int) -> None:
        """Add sentence to trie with given frequency"""
        node = self.root
        for char in sentence:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_end = True
        node.sentence = sentence
        node.frequency += frequency
    
    def _search_with_prefix(self, prefix: str) -> list[str]:
        """Find all sentences with given prefix"""
        node = self.root
        
        # Navigate to prefix
        for char in prefix:
            if char not in node.children:
                return []  # No matches
            node = node.children[char]
        
        # Collect all completions from this node
        results = []
        self._dfs_collect(node, results)
        
        # Sort by frequency (desc), then lexicographically (asc)
        results.sort(key=lambda x: (-x[1], x[0]))
        
        # Return top 3 sentences
        return [sentence for sentence, freq in results[:3]]
    
    def _dfs_collect(self, node: TrieNode, results: list) -> None:
        """DFS to collect all sentences from this node"""
        if node.is_end:
            results.append((node.sentence, node.frequency))
        
        for child in node.children.values():
            self._dfs_collect(child, results)
    
    def input(self, c: str) -> list[str]:
        """Process input character"""
        if c == '#':
            # End of sentence - save it
            self._add_to_trie(self.current_input, 1)
            self.current_input = ""
            return []
        else:
            # Extend current input
            self.current_input += c
            return self._search_with_prefix(self.current_input)

# Example usage:
# autocomplete = AutocompleteSystem(["i love you", "island", "ironman"], [5, 3, 2])
# autocomplete.input('i')  # ["i love you", "island", "ironman"]
# autocomplete.input(' ')  # ["i love you"]
# autocomplete.input('a')  # []
# autocomplete.input('#')  # []`,
  timeComplexity:
    'O(p + m log m) per input, where p=prefix length, m=matching sentences',
  spaceComplexity:
    'O(N * L) where N=total sentences, L=average sentence length',
  patterns: ['Trie', 'DFS', 'Design', 'Sorting'],
  companies: ['Google', 'Amazon', 'Microsoft', 'Facebook'],
};
