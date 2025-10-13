import { Problem } from '../types';

export const triesProblems: Problem[] = [
  {
    id: 'implement-trie',
    title: 'Implement Trie (Prefix Tree)',
    difficulty: 'Easy',
    description: `Implement a **trie** (prefix tree) with \`insert\`, \`search\`, and \`starts_with\` methods.

**Trie** is a tree-like data structure used to store strings. Each node represents a character, and paths from root to nodes represent prefixes.


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

    leetcodeUrl: 'https://leetcode.com/problems/implement-trie-prefix-tree/',
    youtubeUrl: 'https://www.youtube.com/watch?v=oobqoCJlHA0',
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

    leetcodeUrl: 'https://leetcode.com/problems/word-search-ii/',
    youtubeUrl: 'https://www.youtube.com/watch?v=asbcE9mZz_U',
    order: 3,
    topic: 'Tries',
    leetcodeUrl: 'https://leetcode.com/problems/word-search-ii/',
    youtubeUrl: 'https://www.youtube.com/watch?v=asbcE9mZz_U',
  },

  // EASY - Longest Common Prefix
  {
    id: 'longest-common-prefix-trie',
    title: 'Longest Common Prefix',
    difficulty: 'Easy',
    topic: 'Tries',
    description: `Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string \`""\`.`,
    examples: [
      {
        input: 'strs = ["flower","flow","flight"]',
        output: '"fl"',
      },
      {
        input: 'strs = ["dog","racecar","car"]',
        output: '""',
        explanation: 'There is no common prefix.',
      },
    ],
    constraints: [
      '1 <= strs.length <= 200',
      '0 <= strs[i].length <= 200',
      'strs[i] consists of only lowercase English letters',
    ],
    hints: [
      'Build trie from all words',
      'Traverse from root until branching or end',
    ],
    starterCode: `from typing import List

def longest_common_prefix(strs: List[str]) -> str:
    """
    Find longest common prefix in array of strings.
    
    Args:
        strs: Array of strings
        
    Returns:
        Longest common prefix
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [['flower', 'flow', 'flight']],
        expected: 'fl',
      },
      {
        input: [['dog', 'racecar', 'car']],
        expected: '',
      },
    ],
    timeComplexity: 'O(S) where S is sum of all characters',
    spaceComplexity: 'O(S)',
    leetcodeUrl: 'https://leetcode.com/problems/longest-common-prefix/',
    youtubeUrl: 'https://www.youtube.com/watch?v=0sWShKIJoo4',
  },

  // EASY - Implement Magic Dictionary
  {
    id: 'magic-dictionary',
    title: 'Implement Magic Dictionary',
    difficulty: 'Easy',
    topic: 'Tries',
    description: `Design a data structure that is initialized with a list of **different** words. Provided a string, you should determine if you can change exactly one character in this string to match any word in the data structure.

Implement the \`MagicDictionary\` class:
- \`MagicDictionary()\` Initializes the object.
- \`void buildDict(String[] dictionary)\` Sets the data structure with an array of distinct strings \`dictionary\`.
- \`bool search(String searchWord)\` Returns \`true\` if you can change **exactly one character** in \`searchWord\` to match any string in the data structure, otherwise returns \`false\`.`,
    examples: [
      {
        input:
          '["MagicDictionary", "buildDict", "search", "search", "search", "search"], [[], [["hello", "leetcode"]], ["hello"], ["hhllo"], ["hell"], ["leetcoded"]]',
        output: '[null, null, false, true, false, false]',
      },
    ],
    constraints: [
      '1 <= dictionary.length <= 100',
      '1 <= dictionary[i].length <= 100',
      'dictionary[i] consists of only lower-case English letters',
      'All the strings in dictionary are distinct',
      '1 <= searchWord.length <= 100',
      'searchWord consists of only lower-case English letters',
      'buildDict will be called only once before search',
      'At most 100 calls will be made to search',
    ],
    hints: [
      'For each position, try changing character',
      'Check if resulting word exists in trie',
    ],
    starterCode: `from typing import List

class MagicDictionary:
    def __init__(self):
        """Initialize dictionary."""
        # Write your code here
        pass
    
    def build_dict(self, dictionary: List[str]) -> None:
        """Build dictionary from word list."""
        # Write your code here
        pass
    
    def search(self, search_word: str) -> bool:
        """Search for word with one character change."""
        # Write your code here
        pass
`,
    testCases: [
      {
        input: [[['hello', 'leetcode']], 'hello', 'hhllo', 'hell', 'leetcoded'],
        expected: [null, false, true, false, false],
      },
    ],
    timeComplexity: 'O(m * 26) for search where m is word length',
    spaceComplexity: 'O(n * m) for n words',
    leetcodeUrl: 'https://leetcode.com/problems/implement-magic-dictionary/',
    youtubeUrl: 'https://www.youtube.com/watch?v=9HHuFBx2T_k',
  },

  // EASY - Replace Words
  {
    id: 'replace-words',
    title: 'Replace Words',
    difficulty: 'Easy',
    topic: 'Tries',
    description: `In English, we have a concept called **root**, which can be followed by some other word to form another longer word - let us call this word **derivative**. For example, when the **root** \`"help"\` is followed by the word \`"ful"\`, we can form a derivative \`"helpful"\`.

Given a \`dictionary\` consisting of many **roots** and a \`sentence\` consisting of words separated by spaces, replace all the derivatives in the sentence with the **root** forming it. If a derivative can be replaced by more than one **root**, replace it with the **root** that has **the shortest length**.

Return the \`sentence\` after the replacement.`,
    examples: [
      {
        input:
          'dictionary = ["cat","bat","rat"], sentence = "the cattle was rattled by the battery"',
        output: '"the cat was rat by the bat"',
      },
      {
        input:
          'dictionary = ["a","b","c"], sentence = "aadsfasf absbs bbab cadsfafs"',
        output: '"a a b c"',
      },
    ],
    constraints: [
      '1 <= dictionary.length <= 1000',
      '1 <= dictionary[i].length <= 100',
      'dictionary[i] consists of only lower-case letters',
      '1 <= sentence.length <= 10^6',
      'sentence consists of only lower-case letters and spaces',
      'The number of words in sentence is in the range [1, 1000]',
      'The length of each word in sentence is in the range [1, 1000]',
      'Every two consecutive words in sentence will be separated by exactly one space',
      'sentence does not have leading or trailing spaces',
    ],
    hints: [
      'Build trie from all roots',
      'For each word, find shortest root prefix',
    ],
    starterCode: `from typing import List

def replace_words(dictionary: List[str], sentence: str) -> str:
    """
    Replace words with shortest root.
    
    Args:
        dictionary: List of roots
        sentence: Input sentence
        
    Returns:
        Sentence with words replaced
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [['cat', 'bat', 'rat'], 'the cattle was rattled by the battery'],
        expected: 'the cat was rat by the bat',
      },
      {
        input: [['a', 'b', 'c'], 'aadsfasf absbs bbab cadsfafs'],
        expected: 'a a b c',
      },
    ],
    timeComplexity:
      'O(N + M) where N is total chars in dictionary, M in sentence',
    spaceComplexity: 'O(N)',
    leetcodeUrl: 'https://leetcode.com/problems/replace-words/',
    youtubeUrl: 'https://www.youtube.com/watch?v=pJJ17U_Qiw8',
  },

  // MEDIUM - Search Suggestions System
  {
    id: 'search-suggestions-system',
    title: 'Search Suggestions System',
    difficulty: 'Medium',
    topic: 'Tries',
    description: `You are given an array of strings \`products\` and a string \`searchWord\`.

Design a system that suggests at most three product names from \`products\` after each character of \`searchWord\` is typed. Suggested products should have common prefix with \`searchWord\`. If there are more than three products with a common prefix return the three lexicographically minimum products.

Return a list of lists of the suggested products after each character of \`searchWord\` is typed.`,
    examples: [
      {
        input:
          'products = ["mobile","mouse","moneypot","monitor","mousepad"], searchWord = "mouse"',
        output:
          '[["mobile","moneypot","monitor"],["mobile","moneypot","monitor"],["mouse","mousepad"],["mouse","mousepad"],["mouse","mousepad"]]',
      },
      {
        input: 'products = ["havana"], searchWord = "havana"',
        output:
          '[["havana"],["havana"],["havana"],["havana"],["havana"],["havana"]]',
      },
    ],
    constraints: [
      '1 <= products.length <= 1000',
      '1 <= products[i].length <= 3000',
      '1 <= sum(products[i].length) <= 2 * 10^4',
      'All the strings of products are unique',
      'products[i] consists of lowercase English letters',
      '1 <= searchWord.length <= 1000',
      'searchWord consists of lowercase English letters',
    ],
    hints: [
      'Build trie from products',
      'For each prefix, DFS to find up to 3 words',
    ],
    starterCode: `from typing import List

def suggested_products(products: List[str], search_word: str) -> List[List[str]]:
    """
    Find product suggestions for each prefix.
    
    Args:
        products: List of product names
        search_word: Search query
        
    Returns:
        List of suggestions for each prefix
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          ['mobile', 'mouse', 'moneypot', 'monitor', 'mousepad'],
          'mouse',
        ],
        expected: [
          ['mobile', 'moneypot', 'monitor'],
          ['mobile', 'moneypot', 'monitor'],
          ['mouse', 'mousepad'],
          ['mouse', 'mousepad'],
          ['mouse', 'mousepad'],
        ],
      },
    ],
    timeComplexity:
      'O(N * M + S * M) where N = products, M = avg length, S = searchWord length',
    spaceComplexity: 'O(N * M)',
    leetcodeUrl: 'https://leetcode.com/problems/search-suggestions-system/',
    youtubeUrl: 'https://www.youtube.com/watch?v=D4T2N0yAr20',
  },

  // MEDIUM - Maximum XOR of Two Numbers
  {
    id: 'maximum-xor-two-numbers',
    title: 'Maximum XOR of Two Numbers in an Array',
    difficulty: 'Medium',
    topic: 'Tries',
    description: `Given an integer array \`nums\`, return the maximum result of \`nums[i] XOR nums[j]\`, where \`0 <= i <= j < n\`.`,
    examples: [
      {
        input: 'nums = [3,10,5,25,2,8]',
        output: '28',
        explanation: 'The maximum result is 5 XOR 25 = 28.',
      },
      {
        input: 'nums = [14,70,53,83,49,91,36,80,92,51,66,70]',
        output: '127',
      },
    ],
    constraints: ['1 <= nums.length <= 2 * 10^5', '0 <= nums[i] <= 2^31 - 1'],
    hints: [
      'Build binary trie of all numbers',
      'For each number, try to find opposite bits',
      'Maximize XOR by choosing opposite bits',
    ],
    starterCode: `from typing import List

def find_maximum_xor(nums: List[int]) -> int:
    """
    Find maximum XOR of two numbers.
    
    Args:
        nums: Input array
        
    Returns:
        Maximum XOR value
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[3, 10, 5, 25, 2, 8]],
        expected: 28,
      },
      {
        input: [[14, 70, 53, 83, 49, 91, 36, 80, 92, 51, 66, 70]],
        expected: 127,
      },
    ],
    timeComplexity: 'O(n * 32)',
    spaceComplexity: 'O(n * 32)',
    leetcodeUrl:
      'https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/',
    youtubeUrl: 'https://www.youtube.com/watch?v=jCuNJRm_Pw0',
  },

  // MEDIUM - Concatenated Words
  {
    id: 'concatenated-words',
    title: 'Concatenated Words',
    difficulty: 'Medium',
    topic: 'Tries',
    description: `Given an array of strings \`words\` (**without duplicates**), return all the **concatenated words** in the given list of \`words\`.

A **concatenated word** is defined as a string that is comprised entirely of at least two shorter words (not necessarily distinct) in the given array.`,
    examples: [
      {
        input:
          'words = ["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"]',
        output: '["catsdogcats","dogcatsdog","ratcatdogcat"]',
      },
      {
        input: 'words = ["cat","dog","catdog"]',
        output: '["catdog"]',
      },
    ],
    constraints: [
      '1 <= words.length <= 10^4',
      '1 <= words[i].length <= 30',
      'words[i] consists of only lowercase English letters',
      'All the strings of words are unique',
      '1 <= sum(words[i].length) <= 10^5',
    ],
    hints: [
      'Build trie from all words',
      'For each word, check if it can be formed by concatenating',
      'Use DP to check all possible splits',
    ],
    starterCode: `from typing import List

def find_all_concatenated_words_in_a_dict(words: List[str]) -> List[str]:
    """
    Find all concatenated words.
    
    Args:
        words: List of words
        
    Returns:
        List of concatenated words
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [
            'cat',
            'cats',
            'catsdogcats',
            'dog',
            'dogcatsdog',
            'hippopotamuses',
            'rat',
            'ratcatdogcat',
          ],
        ],
        expected: ['catsdogcats', 'dogcatsdog', 'ratcatdogcat'],
      },
      {
        input: [['cat', 'dog', 'catdog']],
        expected: ['catdog'],
      },
    ],
    timeComplexity: 'O(n * m^2) where n = words, m = avg length',
    spaceComplexity: 'O(n * m)',
    leetcodeUrl: 'https://leetcode.com/problems/concatenated-words/',
    youtubeUrl: 'https://www.youtube.com/watch?v=vSc10FaEWmk',
  },
];
