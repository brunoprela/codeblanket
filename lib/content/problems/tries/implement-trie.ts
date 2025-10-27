/**
 * Implement Trie (Prefix Tree)
 * Problem ID: implement-trie
 * Order: 1
 */

import { Problem } from '../../../types';

export const implement_trieProblem: Problem = {
  id: 'implement-trie',
  title: 'Implement Trie (Prefix Tree)',
  difficulty: 'Easy',
  description: `Implement a **trie** (prefix tree) with \`insert\`, \`search\`, and \`starts_with\` methods.

**Trie** is a tree-like data structure used to store strings. Each node represents a character, and paths from root to nodes represent prefixes.


**Methods to Implement:**1. \`Trie()\` - Initializes the trie object
2. \`insert(word)\` - Inserts string \`word\` into the trie
3. \`search(word)\` - Returns \`true\` if \`word\` is in the trie (as complete word)
4. \`starts_with(prefix)\` - Returns \`true\` if there is a word with prefix \`prefix\`

**Key Insight:**
Each node needs:
- Dictionary/array of children
- Boolean flag marking end of word`,
  examples: [
    {
      input:
        'trie.insert("apple"); trie.search("apple"); trie.search("app"); trie.starts_with("app"); trie.insert("app"); trie.search("app")',
      output: 'None, True, False, True, None, True',
      explanation:
        'After inserting "apple", searching for "apple" returns true, but "app" returns false (not complete word). However, "app" is a valid prefix. After inserting "app", searching for it returns true.',
    },
  ],
  constraints: [
    '1 <= word.length, prefix.length <= 2000',
    'word and prefix consist only of lowercase English letters',
    'At most 30000 calls in total to insert, search, and startsWith',
  ],
  hints: [
    'Create a TrieNode class with children (dict or array) and is_end flag',
    'Root node is empty, represents start of all words',
    'Insert: Traverse/create path for each character, mark last node as end',
    'Search: Traverse path, return true only if node exists AND is_end is true',
    'StartsWith: Traverse path, return true if all characters found',
  ],
  starterCode: `class TrieNode:
    """Node in the Trie structure."""
    def __init__(self):
        # Initialize node properties
        pass

class Trie:
    """Prefix tree for efficient string operations."""
    
    def __init__(self):
        """Initialize the trie."""
        pass
    
    def insert(self, word: str) -> None:
        """Insert a word into the trie."""
        pass
    
    def search(self, word: str) -> bool:
        """Check if exact word exists in trie."""
        pass
    
    def starts_with(self, prefix: str) -> bool:
        """Check if any word starts with given prefix."""
        pass
`,
  testCases: [
    {
      input: [
        ['insert', 'search', 'search'],
        ['apple', 'apple', 'app'],
      ],
      expected: [null, true, false],
    },
    {
      input: [
        ['insert', 'starts_with', 'insert', 'search'],
        ['apple', 'app', 'app', 'app'],
      ],
      expected: [null, true, null, true],
    },
  ],
  solution: `class TrieNode:
    """Node in the Trie structure."""
    
    def __init__(self):
        self.children = {}  # char -> TrieNode
        self.is_end_of_word = False


class Trie:
    """
    Prefix tree implementation.
    Time: O(m) for all operations where m = word length
    Space: O(ALPHABET_SIZE * m * n) worst case
    """
    
    def __init__(self):
        """Initialize with empty root node."""
        self.root = TrieNode()
    
    def insert(self, word: str) -> None:
        """
        Insert a word into the trie.
        Time: O(m), Space: O(m)
        """
        node = self.root
        
        # Traverse/create path for each character
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        # Mark end of word
        node.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        """
        Check if exact word exists.
        Time: O(m), Space: O(1)
        """
        node = self._search_prefix(word)
        return node is not None and node.is_end_of_word
    
    def starts_with(self, prefix: str) -> bool:
        """
        Check if any word starts with prefix.
        Time: O(m), Space: O(1)
        """
        return self._search_prefix(prefix) is not None
    
    def _search_prefix(self, prefix: str) -> TrieNode:
        """
        Helper to traverse to end of prefix.
        Returns node at end, or None if not found.
        """
        node = self.root
        
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        
        return node


# Alternative: Array-based (for lowercase only)
class TrieNodeArray:
    def __init__(self):
        self.children = [None] * 26  # a-z
        self.is_end = False


class TrieArray:
    """
    Faster for fixed alphabet, uses more space.
    """
    
    def __init__(self):
        self.root = TrieNodeArray()
    
    def _char_to_index(self, char):
        return ord(char) - ord('a')
    
    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            idx = self._char_to_index(char)
            if node.children[idx] is None:
                node.children[idx] = TrieNodeArray()
            node = node.children[idx]
        node.is_end = True
    
    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            idx = self._char_to_index(char)
            if node.children[idx] is None:
                return False
            node = node.children[idx]
        return node.is_end
    
    def starts_with(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            idx = self._char_to_index(char)
            if node.children[idx] is None:
                return False
            node = node.children[idx]
        return True`,
  timeComplexity: 'O(m) where m is the word/prefix length',
  spaceComplexity: 'O(n * m) where n is number of words',

  leetcodeUrl: 'https://leetcode.com/problems/implement-trie-prefix-tree/',
  youtubeUrl: 'https://www.youtube.com/watch?v=oobqoCJlHA0',
  order: 1,
  topic: 'Tries',
};
