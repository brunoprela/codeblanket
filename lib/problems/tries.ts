import { Problem } from '../types';

export const triesProblems: Problem[] = [
  {
    id: 'implement-trie',
    title: 'Implement Trie (Prefix Tree)',
    difficulty: 'Easy',
    description: `Implement a **trie** (prefix tree) with \`insert\`, \`search\`, and \`starts_with\` methods.

**Trie** is a tree-like data structure used to store strings. Each node represents a character, and paths from root to nodes represent prefixes.

**LeetCode:** [208. Implement Trie](https://leetcode.com/problems/implement-trie-prefix-tree/)
**YouTube:** [NeetCode - Implement Trie](https://www.youtube.com/watch?v=oobqoCJlHA0)

**Methods to Implement:**

1. \`Trie()\` - Initializes the trie object
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
    order: 1,
    topic: 'Tries',
    leetcodeUrl: 'https://leetcode.com/problems/implement-trie-prefix-tree/',
    youtubeUrl: 'https://www.youtube.com/watch?v=oobqoCJlHA0',
  },
  {
    id: 'add-and-search-word',
    title: 'Design Add and Search Words Data Structure',
    difficulty: 'Medium',
    description: `Design a data structure that supports adding new words and finding if a string matches any previously added string.

Implement the \`WordDictionary\` class:
- \`add_word(word)\` - Adds \`word\` to the data structure
- \`search(word)\` - Returns \`true\` if there is any string that matches \`word\`, or \`false\` otherwise

\`word\` may contain dots \`'.'\` where **dots can be matched with any letter**.

**LeetCode:** [211. Design Add and Search Words Data Structure](https://leetcode.com/problems/design-add-and-search-words-data-structure/)
**YouTube:** [NeetCode - Add and Search Word](https://www.youtube.com/watch?v=BTf05gs_8iU)

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
    order: 2,
    topic: 'Tries',
    leetcodeUrl:
      'https://leetcode.com/problems/design-add-and-search-words-data-structure/',
    youtubeUrl: 'https://www.youtube.com/watch?v=BTf05gs_8iU',
  },
  {
    id: 'word-search-ii',
    title: 'Word Search II',
    difficulty: 'Hard',
    description: `Given an \`m x n\` board of characters and a list of strings \`words\`, return **all words on the board**.

Each word must be constructed from letters of sequentially **adjacent cells**, where adjacent cells are horizontally or vertically neighboring. The same letter cell may **not be used more than once** in a word.

**LeetCode:** [212. Word Search II](https://leetcode.com/problems/word-search-ii/)
**YouTube:** [NeetCode - Word Search II](https://www.youtube.com/watch?v=asbcE9mZz_U)

**Approach:**
Use **Trie + DFS backtracking**:
1. Build trie from all words
2. DFS from each cell
3. At each step, check if prefix exists in trie
4. If word found, add to result
5. Backtrack and mark visited cells

**Key Optimization:**
Remove found words from trie to avoid duplicates and speed up search.

**Time Complexity:**
O(m * n * 4^L) where L is max word length, but trie pruning makes it much faster in practice.`,
    examples: [
      {
        input:
          'board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]',
        output: '["eat","oath"]',
        explanation:
          '"eat" and "oath" can be formed from adjacent cells. "pea" and "rain" cannot.',
      },
      {
        input: 'board = [["a","b"],["c","d"]], words = ["abcb"]',
        output: '[]',
        explanation: 'Cannot use same cell twice.',
      },
    ],
    constraints: [
      'm == board.length',
      'n == board[i].length',
      '1 <= m, n <= 12',
      'board[i][j] is a lowercase English letter',
      '1 <= words.length <= 30000',
      '1 <= words[i].length <= 10',
      'words[i] consists of lowercase English letters',
      'All words[i] are unique',
    ],
    hints: [
      'Build a Trie from all words first',
      'DFS from each cell in the board',
      'At each cell, check if current path exists in trie',
      'Use trie to prune early - stop if prefix not in trie',
      'Mark cells as visited during DFS, unmark on backtrack',
      'Optimization: Remove word from trie once found to avoid duplicates',
    ],
    starterCode: `from typing import List

def find_words(board: List[List[str]], words: List[str]) -> List[str]:
    """
    Find all words from list that exist in board.
    
    Args:
        board: 2D grid of characters
        words: List of words to search for
        
    Returns:
        List of words found in board
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [
            ['o', 'a', 'a', 'n'],
            ['e', 't', 'a', 'e'],
            ['i', 'h', 'k', 'r'],
            ['i', 'f', 'l', 'v'],
          ],
          ['oath', 'pea', 'eat', 'rain'],
        ],
        expected: ['oath', 'eat'],
      },
      {
        input: [
          [
            ['a', 'b'],
            ['c', 'd'],
          ],
          ['abcb'],
        ],
        expected: [],
      },
    ],
    solution: `from typing import List


class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = None  # Store complete word here


def find_words(board: List[List[str]], words: List[str]) -> List[str]:
    """
    Trie + DFS backtracking.
    Time: O(m*n*4^L) worst, much better with pruning
    Space: O(N) for trie where N = total characters in words
    """
    # Build trie
    root = TrieNode()
    for word in words:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.word = word  # Mark end with word itself
    
    rows, cols = len(board), len(board[0])
    result = []
    
    def dfs(r, c, node):
        # Get character and check trie
        char = board[r][c]
        if char not in node.children:
            return
        
        node = node.children[char]
        
        # Found a word
        if node.word:
            result.append(node.word)
            node.word = None  # Avoid duplicates
        
        # Mark as visited
        board[r][c] = '#'
        
        # Explore all 4 directions
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != '#':
                dfs(nr, nc, node)
        
        # Backtrack
        board[r][c] = char
        
        # Optimization: Remove empty nodes (prune trie)
        if not node.children:
            del node.children
    
    # Start DFS from each cell
    for r in range(rows):
        for c in range(cols):
            dfs(r, c, root)
    
    return result


# Alternative: Without modifying board
def find_words_clean(board: List[List[str]], words: List[str]) -> List[str]:
    """
    Same approach but uses visited set instead of modifying board.
    """
    # Build trie
    root = TrieNode()
    for word in words:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.word = word
    
    rows, cols = len(board), len(board[0])
    result = []
    visited = set()
    
    def dfs(r, c, node):
        if (r, c) in visited:
            return
        
        char = board[r][c]
        if char not in node.children:
            return
        
        node = node.children[char]
        
        if node.word:
            result.append(node.word)
            node.word = None
        
        visited.add((r, c))
        
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                dfs(nr, nc, node)
        
        visited.remove((r, c))
    
    for r in range(rows):
        for c in range(cols):
            dfs(r, c, root)
    
    return result


# Alternative: With early termination
def find_words_optimized(board: List[List[str]], words: List[str]) -> List[str]:
    """
    Additional optimization: count remaining words.
    """
    root = TrieNode()
    for word in words:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.word = word
    
    rows, cols = len(board), len(board[0])
    result = []
    remaining = len(words)
    
    def dfs(r, c, node):
        nonlocal remaining
        
        if remaining == 0:  # Early termination
            return
        
        char = board[r][c]
        if char not in node.children:
            return
        
        node = node.children[char]
        
        if node.word:
            result.append(node.word)
            node.word = None
            remaining -= 1
        
        board[r][c] = '#'
        
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != '#':
                dfs(nr, nc, node)
        
        board[r][c] = char
    
    for r in range(rows):
        for c in range(cols):
            if remaining == 0:
                break
            dfs(r, c, root)
    
    return result`,
    timeComplexity: 'O(m * n * 4^L) where L is max word length',
    spaceComplexity: 'O(N) where N is total characters in all words',
    order: 3,
    topic: 'Tries',
    leetcodeUrl: 'https://leetcode.com/problems/word-search-ii/',
    youtubeUrl: 'https://www.youtube.com/watch?v=asbcE9mZz_U',
  },
];
