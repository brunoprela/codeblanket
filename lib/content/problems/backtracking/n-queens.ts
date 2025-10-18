/**
 * N-Queens
 * Problem ID: n-queens
 * Order: 3
 */

import { Problem } from '../../../types';

export const n_queensProblem: Problem = {
  id: 'n-queens',
  title: 'N-Queens',
  difficulty: 'Hard',
  description: `The **n-queens** puzzle is the problem of placing \`n\` queens on an \`n x n\` chessboard such that no two queens attack each other.

Given an integer \`n\`, return **all distinct solutions** to the n-queens puzzle. You may return the answer in **any order**.

Each solution contains a distinct board configuration where \`'Q'\` and \`'.'\` represent a queen and an empty space, respectively.


**Approach:**
Use backtracking with constraint checking. Place queens row by row. For each row, try each column and check if it is valid (no conflicts with previously placed queens in same column or diagonals).

**Constraints:**
- No two queens in same row (guaranteed by placing one per row)
- No two queens in same column (track with set)
- No two queens on same diagonal (track with sets: r-c and r+c)

**Example 4-Queens Solution:**
\`\`\`
. Q . .
. . . Q
Q . . .
. . Q .
\`\`\``,
  examples: [
    {
      input: 'n = 4',
      output: '[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]',
      explanation: 'There are two distinct solutions to the 4-queens puzzle.',
    },
    {
      input: 'n = 1',
      output: '[["Q"]]',
    },
  ],
  constraints: ['1 <= n <= 9'],
  hints: [
    'Place queens row by row using backtracking',
    'Track used columns with a set',
    'Track diagonals: positive diagonal uses (row - col), negative diagonal uses (row + col)',
    'Check constraints before placing queen (early pruning)',
    'Backtrack by removing queen and unmarking constraints',
    'Time: O(N!) with heavy pruning from constraints',
  ],
  starterCode: `from typing import List

def solve_n_queens(n: int) -> List[List[str]]:
    """
    Solve the N-Queens puzzle.
    
    Args:
        n: Size of the board (n x n)
        
    Returns:
        List of all distinct solutions
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [4],
      expected: [
        ['.Q..', '...Q', 'Q...', '..Q.'],
        ['..Q.', 'Q...', '...Q', '.Q..'],
      ],
    },
    {
      input: [1],
      expected: [['Q']],
    },
    {
      input: [2],
      expected: [],
    },
    {
      input: [3],
      expected: [],
    },
  ],
  solution: `from typing import List


def solve_n_queens(n: int) -> List[List[str]]:
    """
    Backtracking with constraint sets.
    Time: O(N!), Space: O(N^2) for board
    """
    result = []
    board = [['.'] * n for _ in range(n)]
    
    # Track used columns and diagonals
    cols = set()
    pos_diag = set()  # r - c
    neg_diag = set()  # r + c
    
    def backtrack(row):
        # Base case: placed all queens
        if row == n:
            result.append([''.join(r) for r in board])
            return
        
        # Try each column in current row
        for col in range(n):
            # Check if position is valid
            if (col in cols or 
                (row - col) in pos_diag or 
                (row + col) in neg_diag):
                continue
            
            # Place queen
            board[row][col] = 'Q'
            cols.add(col)
            pos_diag.add(row - col)
            neg_diag.add(row + col)
            
            # Recurse to next row
            backtrack(row + 1)
            
            # Remove queen (backtrack)
            board[row][col] = '.'
            cols.remove(col)
            pos_diag.remove(row - col)
            neg_diag.remove(row + col)
    
    backtrack(0)
    return result


# Alternative: Without constraint sets (check each time)
def solve_n_queens_check(n: int) -> List[List[str]]:
    """
    Backtracking with explicit validity checking.
    Less efficient but easier to understand.
    """
    result = []
    board = [['.'] * n for _ in range(n)]
    
    def is_valid(row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 'Q':
                return False
        
        # Check positive diagonal (top-left)
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j -= 1
        
        # Check negative diagonal (top-right)
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j += 1
        
        return True
    
    def backtrack(row):
        if row == n:
            result.append([''.join(r) for r in board])
            return
        
        for col in range(n):
            if is_valid(row, col):
                board[row][col] = 'Q'
                backtrack(row + 1)
                board[row][col] = '.'
    
    backtrack(0)
    return result


# Bonus: Count solutions only (more efficient)
def total_n_queens(n: int) -> int:
    """
    Count number of solutions without generating boards.
    """
    cols = set()
    pos_diag = set()
    neg_diag = set()
    
    def backtrack(row):
        if row == n:
            return 1
        
        count = 0
        for col in range(n):
            if (col in cols or 
                (row - col) in pos_diag or 
                (row + col) in neg_diag):
                continue
            
            cols.add(col)
            pos_diag.add(row - col)
            neg_diag.add(row + col)
            
            count += backtrack(row + 1)
            
            cols.remove(col)
            pos_diag.remove(row - col)
            neg_diag.remove(row + col)
        
        return count
    
    return backtrack(0)`,
  timeComplexity: 'O(N!) with constraint pruning',
  spaceComplexity: 'O(N^2) for board + O(N) for sets',
  order: 3,
  topic: 'Backtracking',
  leetcodeUrl: 'https://leetcode.com/problems/n-queens/',
  youtubeUrl: 'https://www.youtube.com/watch?v=Ph95IHmRp5M',
};
