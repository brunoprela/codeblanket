/**
 * Word Search
 * Problem ID: word-search
 * Order: 9
 */

import { Problem } from '../../../types';

export const word_searchProblem: Problem = {
  id: 'word-search',
  title: 'Word Search',
  difficulty: 'Medium',
  topic: 'Backtracking',
  description: `Given an \`m x n\` grid of characters \`board\` and a string \`word\`, return \`true\` if \`word\` exists in the grid.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.`,
  examples: [
    {
      input:
        'board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"',
      output: 'true',
    },
    {
      input:
        'board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"',
      output: 'true',
    },
    {
      input:
        'board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"',
      output: 'false',
    },
  ],
  constraints: [
    'm == board.length',
    'n = board[i].length',
    '1 <= m, n <= 6',
    '1 <= word.length <= 15',
    'board and word consists of only lowercase and uppercase English letters',
  ],
  hints: [
    'Try starting from each cell',
    'Use DFS backtracking',
    'Mark visited cells temporarily',
    'Restore cells after backtracking',
  ],
  starterCode: `from typing import List

def exist(board: List[List[str]], word: str) -> bool:
    """
    Check if word exists in board.
    
    Args:
        board: 2D grid of characters
        word: Word to search for
        
    Returns:
        True if word exists
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [
          ['A', 'B', 'C', 'E'],
          ['S', 'F', 'C', 'S'],
          ['A', 'D', 'E', 'E'],
        ],
        'ABCCED',
      ],
      expected: true,
    },
    {
      input: [
        [
          ['A', 'B', 'C', 'E'],
          ['S', 'F', 'C', 'S'],
          ['A', 'D', 'E', 'E'],
        ],
        'SEE',
      ],
      expected: true,
    },
    {
      input: [
        [
          ['A', 'B', 'C', 'E'],
          ['S', 'F', 'C', 'S'],
          ['A', 'D', 'E', 'E'],
        ],
        'ABCB',
      ],
      expected: false,
    },
  ],
  timeComplexity: 'O(m * n * 4^L) where L is word length',
  spaceComplexity: 'O(L)',
  leetcodeUrl: 'https://leetcode.com/problems/word-search/',
  youtubeUrl: 'https://www.youtube.com/watch?v=pfiQ_PS1g8E',
};
