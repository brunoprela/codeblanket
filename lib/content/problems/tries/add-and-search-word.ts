/**
 * Design Add and Search Words Data Structure
 * Problem ID: add-and-search-word
 * Order: 2
 */

import { Problem } from '../../../types';

export const add_and_search_wordProblem: Problem = {
  id: 'add-and-search-word',
  title: 'Design Add and Search Words Data Structure',
  difficulty: 'Medium',
  description: `Design a data structure that supports adding new words and finding if a string matches any previously added string.

Implement the \`WordDictionary\` class:
- \`add_word(word)\` - Adds \`word\` to the data structure
- \`search(word)\` - Returns \`true\` if there is any string that matches \`word\`, or \`false\` otherwise

\`word\` may contain dots \`'.'\` where **dots can be matched with any letter**.


**Approach:**
Use a **Trie** with **DFS** for wildcard matching. When encountering \`'.'\`, try all possible children.

**Key Insight:**
Normal trie search is O(m), but wildcard search can be O(26^k) where k is number of dots.`,
  examples: [
    {
      input:
        'add_word("bad"); add_word("dad"); add_word("mad"); search("pad"); search("bad"); search(".ad"); search("b..")',
      output: 'None, None, None, False, True, True, True',
      explanation:
        'search(".ad") matches "bad", "dad", "mad". search("b..") matches "bad".',
    },
  ],
  constraints: [
    '1 <= word.length <= 25',
    'word in add_word consists of lowercase English letters',
    'word in search consists of "." or lowercase English letters',
    'There will be at most 2 dots in word for search queries',
    'At most 10^4 calls to add_word and search',
  ],
  hints: [
    'Use a Trie to store all words',
    'For search without dots, use normal trie traversal',
    'For dots, use DFS/backtracking to try all possible children',
    'If word[i] is ".", recursively search all children at current node',
    'Base case: when i == len(word), check if node.is_end',
  ],
  starterCode: `class WordDictionary:
    """Data structure supporting add and wildcard search."""
    
    def __init__(self):
        """Initialize the dictionary."""
        pass
    
    def add_word(self, word: str) -> None:
        """Add a word to the dictionary."""
        pass
    
    def search(self, word: str) -> bool:
        """
        Search for word. '.' matches any letter.
        """
        pass
`,
  testCases: [
    {
      input: [
        ['add_word', 'add_word', 'add_word', 'search', 'search', 'search'],
        ['bad', 'dad', 'mad', 'pad', 'bad', '.ad'],
      ],
      expected: [null, null, null, false, true, true],
    },
    {
      input: [
        ['add_word', 'search', 'search'],
        ['a', '.', 'a'],
      ],
      expected: [null, true, true],
    },
  ],
  solution: `class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False


class WordDictionary:
    """
    Trie with wildcard search using DFS.
    Add: O(m), Search: O(m) best, O(26^k * m) worst (k dots)
    Space: O(n * m)
    """
    
    def __init__(self):
        self.root = TrieNode()
    
    def add_word(self, word: str) -> None:
        """
        Add word to trie.
        Time: O(m), Space: O(m)
        """
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word: str) -> bool:
        """
        Search with wildcard support.
        Time: O(m) best case, O(26^k * m) worst case
        """
        def dfs(node, i):
            # Base case: reached end of word
            if i == len(word):
                return node.is_end
            
            char = word[i]
            
            # Wildcard: try all children
            if char == '.':
                for child in node.children.values():
                    if dfs(child, i + 1):
                        return True
                return False
            
            # Regular character: exact match
            if char not in node.children:
                return False
            return dfs(node.children[char], i + 1)
        
        return dfs(self.root, 0)


# Alternative: Iterative BFS approach
class WordDictionaryBFS:
    """
    BFS approach for wildcard search.
    """
    
    def __init__(self):
        self.root = TrieNode()
    
    def add_word(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word: str) -> bool:
        from collections import deque
        
        queue = deque([(self.root, 0)])
        
        while queue:
            node, i = queue.popleft()
            
            if i == len(word):
                if node.is_end:
                    return True
                continue
            
            char = word[i]
            
            if char == '.':
                # Add all children
                for child in node.children.values():
                    queue.append((child, i + 1))
            else:
                # Add specific child
                if char in node.children:
                    queue.append((node.children[char], i + 1))
        
        return False


# Alternative: Optimized with length grouping
class WordDictionaryOptimized:
    """
    Group words by length for faster wildcard search.
    """
    
    def __init__(self):
        self.tries = {}  # length -> Trie
    
    def add_word(self, word: str) -> None:
        length = len(word)
        if length not in self.tries:
            self.tries[length] = WordDictionary()
        self.tries[length].add_word(word)
    
    def search(self, word: str) -> bool:
        length = len(word)
        if length not in self.tries:
            return False
        return self.tries[length].search(word)`,
  timeComplexity: 'O(m) for add, O(m) to O(26^k * m) for search (k = dots)',
  spaceComplexity: 'O(n * m) for n words',

  leetcodeUrl:
    'https://leetcode.com/problems/design-add-and-search-words-data-structure/',
  youtubeUrl: 'https://www.youtube.com/watch?v=BTf05gs_8iU',
  order: 2,
  topic: 'Tries',
};
