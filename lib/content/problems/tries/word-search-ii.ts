/**
 * Word Search II
 * Problem ID: word-search-ii
 * Order: 3
 */

import { Problem } from '../../../types';

export const word_search_iiProblem: Problem = {
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
};
