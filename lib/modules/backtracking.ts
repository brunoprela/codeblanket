import { Module } from '@/lib/types';

export const backtrackingModule: Module = {
  id: 'backtracking',
  title: 'Backtracking',
  description:
    'Master backtracking for exploring all possible solutions through exhaustive search with pruning.',
  icon: '🔙',
  timeComplexity: 'O(2^N) to O(N!) depending on problem',
  spaceComplexity: 'O(N) for recursion depth',
  sections: [
    {
      id: 'introduction',
      title: 'Introduction to Backtracking',
      content: `**Backtracking** is an algorithmic technique for solving problems **recursively** by building candidates for solutions incrementally and **abandoning** (backtracking from) candidates as soon as it determines they cannot lead to a valid solution.

**Core Concept:**
1. **Make a choice** (add to current solution)
2. **Explore** consequences recursively
3. **Undo the choice** (backtrack) if it doesn't work
4. **Try next choice**

**Think of it as:**
- Exploring a maze: try a path, if it's a dead end, backtrack and try another
- Building a solution tree: explore branches, prune invalid ones

**When to Use Backtracking:**
- Generate **all** possible solutions/combinations
- Problems with **constraints** to satisfy
- **Exhaustive search** required
- Keywords: "all", "generate", "combinations", "permutations", "subsets"

**Backtracking vs. Brute Force:**
- **Brute Force**: Try every possibility, check validity after
- **Backtracking**: Stop exploring invalid paths early (**pruning**)

**Visual Example: Generate Subsets of [1,2,3]**
\`\`\`
Decision Tree:
                    []
          /          |          \\
        [1]         [2]         [3]
       /   \\         |
    [1,2]  [1,3]   [2,3]
      |
   [1,2,3]

Backtracking explores all branches, building solutions incrementally.
\`\`\`

**Template:**
\`\`\`python
def backtrack(path, choices):
    # Base case: valid solution found
    if is_valid_solution(path):
        result.append(path.copy())  # Save solution
        return
    
    # Try each possible choice
    for choice in get_choices(choices):
        # Make choice
        path.append(choice)
        
        # Explore with this choice
        backtrack(path, updated_choices)
        
        # Undo choice (backtrack)
        path.pop()
\`\`\`

**Common Problem Types:**
1. **Combination/Subset** problems
2. **Permutation** problems
3. **Constraint satisfaction** (N-Queens, Sudoku)
4. **Path finding** (mazes, word search)
5. **Partition** problems`,
    },
    {
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
def subsets(nums):
    result = []
    
    def backtrack(start, path):
        result.append(path[:])  # Add current subset
        
        for i in range(start, len(nums)):
            path.append(nums[i])       # Include nums[i]
            backtrack(i + 1, path)     # Explore
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
def permute(nums):
    result = []
    
    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return
        
        for i in range(len(remaining)):
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
def permute_visited(nums):
    result = []
    
    def backtrack(path):
        if len(path) == len(nums):
            result.append(path[:])
            return
        
        for num in nums:
            if num not in path:  # Or use visited set
                path.append(num)
                backtrack(path)
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
def solve_n_queens(n):
    result = []
    board = [['.'] * n for _ in range(n)]
    
    def is_valid(row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 'Q':
                return False
        
        # Check diagonals
        for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
            if board[i][j] == 'Q':
                return False
        for i, j in zip(range(row-1, -1, -1), range(col+1, n)):
            if board[i][j] == 'Q':
                return False
        
        return True
    
    def backtrack(row):
        if row == n:
            result.append([''.join(r) for r in board])
            return
        
        for col in range(n):
            if is_valid(row, col):
                board[row][col] = 'Q'  # Place queen
                backtrack(row + 1)     # Next row
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
def exist(board, word):
    rows, cols = len(board), len(board[0])
    
    def backtrack(r, c, index):
        # Found complete word
        if index == len(word):
            return True
        
        # Out of bounds or wrong character
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            board[r][c] != word[index]):
            return False
        
        # Mark visited
        temp = board[r][c]
        board[r][c] = '#'
        
        # Explore all 4 directions
        found = (backtrack(r+1, c, index+1) or
                 backtrack(r-1, c, index+1) or
                 backtrack(r, c+1, index+1) or
                 backtrack(r, c-1, index+1))
        
        # Unmark (backtrack)
        board[r][c] = temp
        
        return found
    
    # Try each cell as starting point
    for r in range(rows):
        for c in range(cols):
            if backtrack(r, c, 0):
                return True
    return False
\`\`\``,
    },
    {
      id: 'complexity',
      title: 'Complexity Analysis',
      content: `**Backtracking Complexities:**

**Subsets:**
- **Time**: O(2^N * N) 
  - 2^N subsets
  - O(N) to copy each subset
- **Space**: O(N) recursion depth

**Permutations:**
- **Time**: O(N! * N)
  - N! permutations
  - O(N) to copy each
- **Space**: O(N) recursion depth

**Combinations:**
- **Time**: O(C(N,K) * K) = O(N! / (K! * (N-K)!) * K)
- **Space**: O(K) recursion depth

**N-Queens:**
- **Time**: O(N!)
  - N choices for row 1
  - At most N-1 for row 2
  - Exponential backtracking
- **Space**: O(N^2) for board + O(N) recursion

**Word Search:**
- **Time**: O(M * N * 4^L) where L = word length
  - M*N starting positions
  - 4^L paths (4 directions, L length)
- **Space**: O(L) recursion depth

**Sudoku:**
- **Time**: O(9^(N*N)) where N = 9
  - 9 choices per empty cell
  - Highly constrained, pruning helps
- **Space**: O(N*N) for board

**General Backtracking:**
- **Time**: O(b^d) where b = branching factor, d = depth
- **Space**: O(d) for recursion stack

**Optimization Techniques:**

**1. Early Pruning**
Check constraints before recursing, not after:
\`\`\`python
# Bad
if len(path) == n and is_valid(path):  # Check at end
    result.append(path)

# Good  
if not is_valid(path):  # Check early
    return
if len(path) == n:
    result.append(path)
\`\`\`

**2. Constraint Propagation**
Maintain state to avoid recomputation:
\`\`\`python
# Use sets to track used columns/diagonals in N-Queens
cols = set()
diag1 = set()  # r - c
diag2 = set()  # r + c
\`\`\`

**3. Memoization**
Cache results of subproblems (when applicable):
\`\`\`python
memo = {}
def backtrack(state):
    if state in memo:
        return memo[state]
    # ... compute
    memo[state] = result
    return result
\`\`\``,
    },
    {
      id: 'templates',
      title: 'Code Templates',
      content: `**Template 1: Basic Backtracking**
\`\`\`python
def backtracking_template(input):
    result = []
    
    def backtrack(path, choices):
        # Base case
        if is_solution(path):
            result.append(path.copy())
            return
        
        # Try each choice
        for choice in choices:
            # Make choice
            path.append(choice)
            
            # Recurse
            backtrack(path, updated_choices)
            
            # Unmake choice (backtrack)
            path.pop()
    
    backtrack([], input)
    return result
\`\`\`

**Template 2: Subsets (Include/Exclude)**
\`\`\`python
def subsets_template(nums):
    result = []
    
    def backtrack(start, path):
        result.append(path[:])  # Add at every level
        
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)  # Start from i+1
            path.pop()
    
    backtrack(0, [])
    return result
\`\`\`

**Template 3: Combinations (Fixed Size)**
\`\`\`python
def combinations_template(nums, k):
    result = []
    
    def backtrack(start, path):
        if len(path) == k:  # Fixed size
            result.append(path[:])
            return
        
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result
\`\`\`

**Template 4: Permutations**
\`\`\`python
def permutations_template(nums):
    result = []
    
    def backtrack(path):
        if len(path) == len(nums):
            result.append(path[:])
            return
        
        for num in nums:
            if num in path:  # Already used
                continue
            path.append(num)
            backtrack(path)
            path.pop()
    
    backtrack([])
    return result
\`\`\`

**Template 5: Grid DFS + Backtracking**
\`\`\`python
def grid_backtrack_template(grid):
    rows, cols = len(grid), len(grid[0])
    
    def backtrack(r, c, state):
        # Base cases
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return False
        if grid[r][c] == visited_marker:
            return False
        
        # Mark visited
        temp = grid[r][c]
        grid[r][c] = visited_marker
        
        # Explore 4 directions
        found = (backtrack(r+1, c, state) or
                 backtrack(r-1, c, state) or
                 backtrack(r, c+1, state) or
                 backtrack(r, c-1, state))
        
        # Unmark (backtrack)
        grid[r][c] = temp
        
        return found
    
    # Try each cell as starting point
    for r in range(rows):
        for c in range(cols):
            if backtrack(r, c, initial_state):
                return True
    return False
\`\`\`

**Template 6: With Constraints (N-Queens)**
\`\`\`python
def constraint_backtrack_template(n):
    result = []
    board = [['.'] * n for _ in range(n)]
    cols = set()
    diag1 = set()  # r - c
    diag2 = set()  # r + c
    
    def backtrack(row):
        if row == n:
            result.append([''.join(r) for r in board])
            return
        
        for col in range(n):
            # Check constraints
            if (col in cols or 
                (row - col) in diag1 or 
                (row + col) in diag2):
                continue
            
            # Make choice
            board[row][col] = 'Q'
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            
            # Recurse
            backtrack(row + 1)
            
            # Undo choice
            board[row][col] = '.'
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)
    
    backtrack(0)
    return result
\`\`\``,
    },
    {
      id: 'interview',
      title: 'Interview Strategy',
      content: `**Recognition Signals:**

**Use Backtracking when you see:**
- "Generate **all** possible..."
- "Find **all** combinations/permutations/subsets"
- "Count number of ways to..."
- **Constraint satisfaction** (N-Queens, Sudoku)
- "Can you form..." with constraints
- Words: "all", "every", "combinations", "arrangements"

---

**Problem-Solving Steps:**

**Step 1: Identify Problem Type**
- **Subsets?** → Include/exclude decision at each element
- **Permutations?** → Different orders of same elements
- **Combinations?** → Choose K from N, order doesn't matter
- **Constraint satisfaction?** → Place elements satisfying rules
- **Path finding?** → Navigate grid/graph with constraints

**Step 2: Define State**
What information do we need to track?
- **Current path/solution**
- **Remaining choices**
- **Constraints met** (e.g., used columns in N-Queens)

**Step 3: Identify Base Case**
When have we found a complete solution?
- Length reaches target?
- All elements placed?
- Reached destination?

**Step 4: Define Choices**
What options do we have at each step?
- Which elements to add?
- Which cells to explore?
- Which values to try?

**Step 5: Implement Backtracking**
- Make choice
- Recurse
- Undo choice

---

**Interview Communication:**

**Example: Generate Subsets**

1. **Clarify:**
   - "Should the empty set be included?" (Yes)
   - "Are there duplicates in input?" (Affects algorithm)
   - "Does order matter?" (Usually no for subsets)

2. **Explain approach:**
   - "I'll use backtracking to build subsets incrementally."
   - "At each element, I have two choices: include or exclude."
   - "I'll use a start index to avoid duplicates."

3. **Walk through example:**
   \`\`\`
   nums = [1,2,3]
   
   Start with []
   Try including 1: [1]
     Try including 2: [1,2]
       Try including 3: [1,2,3] ← add to result
       Backtrack, try excluding 3
     Backtrack, try excluding 2 but including 3: [1,3]
   Backtrack to [], try excluding 1 but including 2: [2]
   ...
   \`\`\`

4. **Complexity:**
   - "Time: O(2^N * N) - 2^N subsets, O(N) to copy each."
   - "Space: O(N) for recursion depth."

5. **Optimize:**
   - "Could use bit manipulation for space optimization."
   - "Could use iterative approach with queue."

---

**Common Mistakes:**

**1. Forgetting to Copy**
❌ result.append(path)  # Reference, will change!
✅ result.append(path.copy()) or result.append(path[:])

**2. Wrong Base Case**
❌ Check constraints after recursion
✅ Check constraints before recursion (pruning)

**3. Not Backtracking**
❌ Modify state but don't undo
✅ Always undo changes after exploring

**4. Duplicate Solutions**
❌ Not using start index in combinations
✅ Use start parameter to avoid revisiting elements

---

**Practice Plan:**

1. **Basics (Day 1-2):**
   - Subsets
   - Permutations
   - Combinations

2. **Intermediate (Day 3-4):**
   - Combination Sum
   - Letter Combinations of Phone Number
   - Palindrome Partitioning

3. **Advanced (Day 5-7):**
   - N-Queens
   - Word Search
   - Sudoku Solver

4. **Resources:**
   - LeetCode Backtracking tag (100+ problems)
   - Draw decision trees
   - Practice recognizing patterns`,
    },
  ],
  keyTakeaways: [
    'Backtracking explores all possibilities by making choices, exploring, and undoing (backtracking)',
    'Three steps: (1) make choice, (2) recurse with choice, (3) undo choice (backtrack)',
    'Subsets: include/exclude decision at each index, use start parameter to avoid duplicates',
    'Permutations: try all unused elements at each position, track with visited set or remaining list',
    'Combinations: like subsets but with fixed size K, stop when path reaches size K',
    'Constraint satisfaction: check validity before recursing (early pruning)',
    'Always copy path when adding to results (path[:] or path.copy())',
    'Time complexity typically O(2^N) for subsets, O(N!) for permutations, O(b^d) generally',
  ],
  relatedProblems: ['subsets', 'permutations', 'n-queens'],
};
