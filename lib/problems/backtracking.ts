import { Problem } from '../types';

export const backtrackingProblems: Problem[] = [
  {
    id: 'subsets',
    title: 'Subsets',
    difficulty: 'Easy',
    description: `Given an integer array \`nums\` of **unique** elements, return **all possible subsets** (the power set).

The solution set **must not** contain duplicate subsets. Return the solution in **any order**.


**Approach:**
Use backtracking to build subsets incrementally. At each element, we have two choices: include it or exclude it. Use a \`start\` index to avoid generating duplicate subsets.

**Example Decision Tree for [1,2,3]:**
\`\`\`
                    []
          /                   \\
        [1]                    []
       /   \\                 /    \\
    [1,2]  [1]            [2]      []
    /  \\   /  \\          /  \\     /  \\
[1,2,3][1,2][1,3][1] [2,3][2]  [3] []
\`\`\``,
    examples: [
      {
        input: 'nums = [1,2,3]',
        output: '[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]',
        explanation: 'All possible subsets.',
      },
      {
        input: 'nums = [0]',
        output: '[[],[0]]',
      },
    ],
    constraints: [
      '1 <= nums.length <= 10',
      '-10 <= nums[i] <= 10',
      'All the numbers of nums are unique',
    ],
    hints: [
      'Use backtracking to explore all possibilities',
      'At each position, you can either include or exclude the element',
      'Use a start index to avoid generating duplicates ([1,2] vs [2,1])',
      'Add the current subset to results at every recursive call',
      'Time complexity: O(2^N * N) where N is array length',
    ],
    starterCode: `from typing import List

def subsets(nums: List[int]) -> List[List[int]]:
    """
    Generate all subsets (power set).
    
    Args:
        nums: Array of unique integers
        
    Returns:
        List of all possible subsets
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 2, 3]],
        expected: [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]],
      },
      {
        input: [[0]],
        expected: [[], [0]],
      },
      {
        input: [[1, 2]],
        expected: [[], [1], [2], [1, 2]],
      },
    ],
    solution: `from typing import List


def subsets(nums: List[int]) -> List[List[int]]:
    """
    Backtracking approach.
    Time: O(2^N * N), Space: O(N) recursion depth
    """
    result = []
    
    def backtrack(start, path):
        # Add current subset (at every level)
        result.append(path[:])
        
        # Try including each element from start onward
        for i in range(start, len(nums)):
            path.append(nums[i])      # Include nums[i]
            backtrack(i + 1, path)    # Explore with nums[i]
            path.pop()                # Backtrack (exclude nums[i])
    
    backtrack(0, [])
    return result


# Alternative: Iterative approach
def subsets_iterative(nums: List[int]) -> List[List[int]]:
    """
    Iterative approach: build subsets by adding each element.
    Time: O(2^N * N), Space: O(1) if we don't count output
    """
    result = [[]]
    
    for num in nums:
        # Add num to all existing subsets
        result += [subset + [num] for subset in result]
    
    return result


# Alternative: Bit manipulation
def subsets_bitmask(nums: List[int]) -> List[List[int]]:
    """
    Bit manipulation approach: each subset corresponds to a bitmask.
    Time: O(2^N * N), Space: O(1) if we don't count output
    """
    n = len(nums)
    result = []
    
    # Iterate through all possible bitmasks (0 to 2^n - 1)
    for mask in range(1 << n):  # 2^n subsets
        subset = []
        for i in range(n):
            # Check if i-th bit is set
            if mask & (1 << i):
                subset.append(nums[i])
        result.append(subset)
    
    return result`,
    timeComplexity: 'O(2^N * N)',
    spaceComplexity: 'O(N) for recursion depth',
    order: 1,
    topic: 'Backtracking',
    leetcodeUrl: 'https://leetcode.com/problems/subsets/',
    youtubeUrl: 'https://www.youtube.com/watch?v=REOH22Xwdkk',
  },
  {
    id: 'permutations',
    title: 'Permutations',
    difficulty: 'Medium',
    description: `Given an array \`nums\` of **distinct** integers, return **all possible permutations**. You can return the answer in **any order**.


**Approach:**
Use backtracking. At each step, try adding each unused number to the current permutation. When we've used all numbers, we have a complete permutation.

**Key Difference from Subsets:**
- Subsets: partial selections (can stop at any size)
- Permutations: use ALL elements, order matters

**Example for [1,2,3]:**
\`\`\`
Start with []
Try 1: [1]
  Try 2: [1,2]
    Try 3: [1,2,3] ← complete permutation
  Try 3: [1,3]
    Try 2: [1,3,2] ← complete permutation
Try 2: [2]
  Try 1: [2,1]
    Try 3: [2,1,3]
  ...
\`\`\``,
    examples: [
      {
        input: 'nums = [1,2,3]',
        output: '[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]',
      },
      {
        input: 'nums = [0,1]',
        output: '[[0,1],[1,0]]',
      },
      {
        input: 'nums = [1]',
        output: '[[1]]',
      },
    ],
    constraints: [
      '1 <= nums.length <= 6',
      '-10 <= nums[i] <= 10',
      'All the integers of nums are unique',
    ],
    hints: [
      'Use backtracking to build permutations',
      'Track which elements have been used (visited set or check if in current path)',
      'When current path length equals input length, add to results',
      'Try each unused element at each position',
      'Time: O(N! * N) - N! permutations, O(N) to copy each',
    ],
    starterCode: `from typing import List

def permute(nums: List[int]) -> List[List[int]]:
    """
    Generate all permutations.
    
    Args:
        nums: Array of distinct integers
        
    Returns:
        List of all permutations
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 2, 3]],
        expected: [
          [1, 2, 3],
          [1, 3, 2],
          [2, 1, 3],
          [2, 3, 1],
          [3, 1, 2],
          [3, 2, 1],
        ],
      },
      {
        input: [[0, 1]],
        expected: [
          [0, 1],
          [1, 0],
        ],
      },
      {
        input: [[1]],
        expected: [[1]],
      },
    ],
    solution: `from typing import List


def permute(nums: List[int]) -> List[List[int]]:
    """
    Backtracking with 'in path' check.
    Time: O(N! * N), Space: O(N) recursion
    """
    result = []
    
    def backtrack(path):
        # Base case: used all numbers
        if len(path) == len(nums):
            result.append(path[:])
            return
        
        # Try each number
        for num in nums:
            if num in path:  # Already used
                continue
            
            path.append(num)      # Choose
            backtrack(path)       # Explore
            path.pop()            # Unchoose (backtrack)
    
    backtrack([])
    return result


# Alternative: Using visited set (more efficient)
def permute_visited(nums: List[int]) -> List[List[int]]:
    """
    Backtracking with visited set.
    More efficient than 'in path' check.
    """
    result = []
    visited = set()
    
    def backtrack(path):
        if len(path) == len(nums):
            result.append(path[:])
            return
        
        for num in nums:
            if num not in visited:
                visited.add(num)
                path.append(num)
                
                backtrack(path)
                
                path.pop()
                visited.remove(num)
    
    backtrack([])
    return result


# Alternative: Swap-based approach (in-place)
def permute_swap(nums: List[int]) -> List[List[int]]:
    """
    Backtracking with swapping.
    Generates permutations in-place.
    """
    result = []
    
    def backtrack(start):
        if start == len(nums):
            result.append(nums[:])
            return
        
        for i in range(start, len(nums)):
            # Swap
            nums[start], nums[i] = nums[i], nums[start]
            
            # Recurse
            backtrack(start + 1)
            
            # Swap back (backtrack)
            nums[start], nums[i] = nums[i], nums[start]
    
    backtrack(0)
    return result


# Alternative: Using remaining list
def permute_remaining(nums: List[int]) -> List[List[int]]:
    """
    Backtracking by passing remaining elements.
    """
    result = []
    
    def backtrack(path, remaining):
        if not remaining:
            result.append(path)
            return
        
        for i in range(len(remaining)):
            backtrack(
                path + [remaining[i]],
                remaining[:i] + remaining[i+1:]
            )
    
    backtrack([], nums)
    return result`,
    timeComplexity: 'O(N! * N)',
    spaceComplexity: 'O(N) for recursion depth',
    order: 2,
    topic: 'Backtracking',
    leetcodeUrl: 'https://leetcode.com/problems/permutations/',
    youtubeUrl: 'https://www.youtube.com/watch?v=s7AvT7cGdSo',
  },
  {
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
  },
];
