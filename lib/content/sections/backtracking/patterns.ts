/**
 * Backtracking Patterns Section
 */

export const patternsSection = {
  id: 'patterns',
  title: 'Backtracking Patterns',
  content: `**Pattern 1: Subsets / Combinations**

Generate all subsets (power set) of a set.

**Example: Subsets of [1,2,3]**
\`\`\`
Result: [[], [1], [2], [1,2], [3], [1,3], [2,3], [1,2,3]]

Decision at each element: include or exclude

                        []
                /                   \\
            [1]                      []
          /      \\                /     \\
      [1,2]      [1]            [2]      []
      /  \\       /  \\           /  \\     /  \\
  [1,2,3][1,2] [1,3][1]     [2,3][2]  [3] []
\`\`\`

**Code:**
\`\`\`python
def subsets (nums):
    result = []
    
    def backtrack (start, path):
        result.append (path[:])  # Add current subset
        
        for i in range (start, len (nums)):
            path.append (nums[i])       # Include nums[i]
            backtrack (i + 1, path)     # Explore
            path.pop()                 # Backtrack
    
    backtrack(0, [])
    return result
\`\`\`

**Key Points:**
- start parameter prevents duplicates ([1,2] same as [2,1])
- Append at each level (not just leaves)

---

**Pattern 2: Permutations**

Generate all arrangements of elements.

**Example: Permutations of [1,2,3]**
\`\`\`
Result: [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]

All elements used, different orders
\`\`\`

**Code:**
\`\`\`python
def permute (nums):
    result = []
    
    def backtrack (path, remaining):
        if not remaining:
            result.append (path[:])
            return
        
        for i in range (len (remaining)):
            # Choose remaining[i]
            backtrack(
                path + [remaining[i]], 
                remaining[:i] + remaining[i+1:]
            )
    
    backtrack([], nums)
    return result
\`\`\`

**Alternative (Using visited set):**
\`\`\`python
def permute_visited (nums):
    result = []
    
    def backtrack (path):
        if len (path) == len (nums):
            result.append (path[:])
            return
        
        for num in nums:
            if num not in path:  # Or use visited set
                path.append (num)
                backtrack (path)
                path.pop()
    
    backtrack([])
    return result
\`\`\`

---

**Pattern 3: Constraint Satisfaction (N-Queens)**

Place N queens on N×N board so no two queens attack each other.

**Visualization (4-Queens):**
\`\`\`
Valid Solution:
. Q . .
. . . Q
Q . . .
. . Q .

Each row must have exactly 1 queen
No two queens share column, diagonal
\`\`\`

**Code:**
\`\`\`python
def solve_n_queens (n):
    result = []
    board = [['.'] * n for _ in range (n)]
    
    def is_valid (row, col):
        # Check column
        for i in range (row):
            if board[i][col] == 'Q':
                return False
        
        # Check diagonals
        for i, j in zip (range (row-1, -1, -1), range (col-1, -1, -1)):
            if board[i][j] == 'Q':
                return False
        for i, j in zip (range (row-1, -1, -1), range (col+1, n)):
            if board[i][j] == 'Q':
                return False
        
        return True
    
    def backtrack (row):
        if row == n:
            result.append(['.join (r) for r in board])
            return
        
        for col in range (n):
            if is_valid (row, col):
                board[row][col] = 'Q'  # Place queen
                backtrack (row + 1)     # Next row
                board[row][col] = '.'  # Remove queen
    
    backtrack(0)
    return result
\`\`\`

---

**Pattern 4: Word Search / Path Finding**

Find if word exists in grid (can move up/down/left/right).

**Approach:**
- Try each cell as starting point
- DFS + backtracking to explore paths
- Mark visited cells, unmark on backtrack

\`\`\`python
def exist (board, word):
    rows, cols = len (board), len (board[0])
    
    def backtrack (r, c, index):
        # Found complete word
        if index == len (word):
            return True
        
        # Out of bounds or wrong character
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            board[r][c] != word[index]):
            return False
        
        # Mark visited
        temp = board[r][c]
        board[r][c] = '#'
        
        # Explore all 4 directions
        found = (backtrack (r+1, c, index+1) or
                 backtrack (r-1, c, index+1) or
                 backtrack (r, c+1, index+1) or
                 backtrack (r, c-1, index+1))
        
        # Unmark (backtrack)
        board[r][c] = temp
        
        return found
    
    # Try each cell as starting point
    for r in range (rows):
        for c in range (cols):
            if backtrack (r, c, 0):
                return True
    return False
\`\`\``,
};
