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

  // EASY - Letter Case Permutation
  {
    id: 'letter-case-permutation',
    title: 'Letter Case Permutation',
    difficulty: 'Easy',
    topic: 'Backtracking',
    description: `Given a string \`s\`, you can transform every letter individually to be lowercase or uppercase to create another string.

Return a list of all possible strings we could create. Return the output in **any order**.`,
    examples: [
      {
        input: 's = "a1b2"',
        output: '["a1b2","a1B2","A1b2","A1B2"]',
      },
      {
        input: 's = "3z4"',
        output: '["3z4","3Z4"]',
      },
    ],
    constraints: [
      '1 <= s.length <= 12',
      's consists of lowercase English letters, uppercase English letters, and digits',
    ],
    hints: [
      'For each letter, branch to try both cases',
      'For digits, just continue with same char',
    ],
    starterCode: `from typing import List

def letter_case_permutation(s: str) -> List[str]:
    """
    Generate all letter case permutations.
    
    Args:
        s: Input string
        
    Returns:
        List of all permutations
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['a1b2'],
        expected: ['a1b2', 'a1B2', 'A1b2', 'A1B2'],
      },
      {
        input: ['3z4'],
        expected: ['3z4', '3Z4'],
      },
      {
        input: ['C'],
        expected: ['c', 'C'],
      },
    ],
    timeComplexity: 'O(2^n * n)',
    spaceComplexity: 'O(2^n * n)',
    leetcodeUrl: 'https://leetcode.com/problems/letter-case-permutation/',
    youtubeUrl: 'https://www.youtube.com/watch?v=ZCqRY1JbVAo',
  },

  // EASY - Binary Watch
  {
    id: 'binary-watch',
    title: 'Binary Watch',
    difficulty: 'Easy',
    topic: 'Backtracking',
    description: `A binary watch has 4 LEDs on the top to represent the hours (0-11), and 6 LEDs on the bottom to represent the minutes (0-59). Each LED represents a zero or one, with the least significant bit on the right.

Given an integer \`turnedOn\` which represents the number of LEDs that are currently on (ignoring the PM), return all possible times the watch could represent. You may return the answer in **any order**.

The hour must not contain a leading zero. The minute must consist of two digits and may contain a leading zero.`,
    examples: [
      {
        input: 'turnedOn = 1',
        output:
          '["0:01","0:02","0:04","0:08","0:16","0:32","1:00","2:00","4:00","8:00"]',
      },
      {
        input: 'turnedOn = 9',
        output: '[]',
      },
    ],
    constraints: ['0 <= turnedOn <= 10'],
    hints: [
      'Generate all possible hour and minute combinations',
      'Count bits in each combination',
      'Filter by total bits equal to turnedOn',
    ],
    starterCode: `from typing import List

def read_binary_watch(turnedOn: int) -> List[str]:
    """
    Find all possible times with given LED count.
    
    Args:
        turnedOn: Number of LEDs on
        
    Returns:
        List of possible times
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [1],
        expected: [
          '0:01',
          '0:02',
          '0:04',
          '0:08',
          '0:16',
          '0:32',
          '1:00',
          '2:00',
          '4:00',
          '8:00',
        ],
      },
      {
        input: [9],
        expected: [],
      },
      {
        input: [0],
        expected: ['0:00'],
      },
    ],
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/binary-watch/',
    youtubeUrl: 'https://www.youtube.com/watch?v=CwDj8xGG2lU',
  },

  // EASY - Combination Sum III
  {
    id: 'combination-sum-iii',
    title: 'Combination Sum III',
    difficulty: 'Easy',
    topic: 'Backtracking',
    description: `Find all valid combinations of \`k\` numbers that sum up to \`n\` such that the following conditions are true:

- Only numbers \`1\` through \`9\` are used.
- Each number is used **at most once**.

Return a list of all possible valid combinations. The list must not contain the same combination twice, and the combinations may be returned in any order.`,
    examples: [
      {
        input: 'k = 3, n = 7',
        output: '[[1,2,4]]',
        explanation: '1 + 2 + 4 = 7. There are no other valid combinations.',
      },
      {
        input: 'k = 3, n = 9',
        output: '[[1,2,6],[1,3,5],[2,3,4]]',
      },
    ],
    constraints: ['2 <= k <= 9', '1 <= n <= 60'],
    hints: [
      'Use backtracking with numbers 1-9',
      'Track remaining sum and count',
      'Prune when sum exceeds target',
    ],
    starterCode: `from typing import List

def combination_sum3(k: int, n: int) -> List[List[int]]:
    """
    Find combinations of k numbers summing to n.
    
    Args:
        k: Number of elements in combination
        n: Target sum
        
    Returns:
        List of valid combinations
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [3, 7],
        expected: [[1, 2, 4]],
      },
      {
        input: [3, 9],
        expected: [
          [1, 2, 6],
          [1, 3, 5],
          [2, 3, 4],
        ],
      },
      {
        input: [4, 1],
        expected: [],
      },
    ],
    timeComplexity: 'O(9! / (k! * (9-k)!))',
    spaceComplexity: 'O(k)',
    leetcodeUrl: 'https://leetcode.com/problems/combination-sum-iii/',
    youtubeUrl: 'https://www.youtube.com/watch?v=xVGCxTmXRBI',
  },

  // MEDIUM - Combination Sum
  {
    id: 'combination-sum',
    title: 'Combination Sum',
    difficulty: 'Medium',
    topic: 'Backtracking',
    description: `Given an array of **distinct** integers \`candidates\` and a target integer \`target\`, return a list of all **unique combinations** of \`candidates\` where the chosen numbers sum to \`target\`. You may return the combinations in **any order**.

The **same** number may be chosen from \`candidates\` an **unlimited number of times**. Two combinations are unique if the frequency of at least one of the chosen numbers is different.

The test cases are generated such that the number of unique combinations that sum up to \`target\` is less than \`150\` combinations for the given input.`,
    examples: [
      {
        input: 'candidates = [2,3,6,7], target = 7',
        output: '[[2,2,3],[7]]',
      },
      {
        input: 'candidates = [2,3,5], target = 8',
        output: '[[2,2,2,2],[2,3,3],[3,5]]',
      },
    ],
    constraints: [
      '1 <= candidates.length <= 30',
      '2 <= candidates[i] <= 40',
      'All elements of candidates are distinct',
      '1 <= target <= 40',
    ],
    hints: [
      'Use backtracking',
      'Numbers can be reused',
      'Start from current index to avoid duplicates',
    ],
    starterCode: `from typing import List

def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    """
    Find combinations that sum to target (with repetition).
    
    Args:
        candidates: Available numbers
        target: Target sum
        
    Returns:
        List of valid combinations
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[2, 3, 6, 7], 7],
        expected: [[2, 2, 3], [7]],
      },
      {
        input: [[2, 3, 5], 8],
        expected: [
          [2, 2, 2, 2],
          [2, 3, 3],
          [3, 5],
        ],
      },
      {
        input: [[2], 1],
        expected: [],
      },
    ],
    timeComplexity: 'O(2^target)',
    spaceComplexity: 'O(target)',
    leetcodeUrl: 'https://leetcode.com/problems/combination-sum/',
    youtubeUrl: 'https://www.youtube.com/watch?v=GBKI9VSKdGg',
  },

  // MEDIUM - Generate Parentheses
  {
    id: 'generate-parentheses',
    title: 'Generate Parentheses',
    difficulty: 'Medium',
    topic: 'Backtracking',
    description: `Given \`n\` pairs of parentheses, write a function to generate all combinations of well-formed parentheses.`,
    examples: [
      {
        input: 'n = 3',
        output: '["((()))","(()())","(())()","()(())","()()()"]',
      },
      {
        input: 'n = 1',
        output: '["()"]',
      },
    ],
    constraints: ['1 <= n <= 8'],
    hints: [
      'Add opening parenthesis if count < n',
      'Add closing parenthesis if close < open',
      'Base case: when both counts equal n',
    ],
    starterCode: `from typing import List

def generate_parenthesis(n: int) -> List[str]:
    """
    Generate all valid parentheses combinations.
    
    Args:
        n: Number of pairs
        
    Returns:
        List of valid combinations
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [3],
        expected: ['((()))', '(()())', '(())()', '()(())', '()()()'],
      },
      {
        input: [1],
        expected: ['()'],
      },
      {
        input: [2],
        expected: ['(())', '()()'],
      },
    ],
    timeComplexity: 'O(4^n / sqrt(n))',
    spaceComplexity: 'O(n)',
    leetcodeUrl: 'https://leetcode.com/problems/generate-parentheses/',
    youtubeUrl: 'https://www.youtube.com/watch?v=s9fokUqJ76A',
  },

  // MEDIUM - Word Search
  {
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
  },
];
