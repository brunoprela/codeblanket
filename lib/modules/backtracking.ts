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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain what backtracking is and how it differs from brute force. What makes it more efficient?',
          sampleAnswer:
            'Backtracking is a systematic way to explore all possible solutions by making choices incrementally and abandoning paths that cannot lead to valid solutions. It differs from brute force in that it prunes the search space - when we detect a path cannot succeed, we backtrack immediately rather than exploring all possibilities. For example, in N-Queens, if placing a queen creates a conflict, we backtrack without trying to place remaining queens on that board. Brute force would try all placements. The efficiency comes from pruning: we avoid exploring exponentially many invalid paths. Backtracking is essentially DFS with pruning - explore, check constraints, backtrack if invalid, continue if valid.',
          keyPoints: [
            'Incremental choice-making with pruning',
            'Abandon invalid paths early',
            'vs Brute force: explores all possibilities',
            'Prunes search space by checking constraints',
            'DFS with constraint checking',
          ],
        },
        {
          id: 'q2',
          question:
            'Describe the three steps of backtracking pattern. Why is the "undo" step crucial?',
          sampleAnswer:
            'The three steps are: choose (make a choice), explore (recurse with that choice), unchoose (undo the choice). The undo step is crucial because it restores the state for exploring other branches. Without undo, previous choices pollute the state for sibling branches. For example, in permutations, after exploring with element A in position 1, we must remove A before trying B in position 1. The undo ensures each branch starts from the same parent state. This is what enables systematic exploration of the entire solution space - each path is independent. The pattern: modify state, recurse, restore state.',
          keyPoints: [
            'Three steps: choose, explore, unchoose',
            'Undo restores state for other branches',
            'Without undo: state pollution',
            'Enables independent exploration of branches',
            'Pattern: modify, recurse, restore',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk me through when you would use backtracking vs other approaches like greedy or dynamic programming.',
          sampleAnswer:
            'Use backtracking when you need to find all solutions or explore all possibilities with constraints - problems where greedy does not work and DP does not apply. Greedy makes locally optimal choices and cannot backtrack - use when local optimum leads to global optimum. DP solves overlapping subproblems by memoization - use when problem has optimal substructure. Backtracking is for: generating all combinations/permutations, constraint satisfaction like N-Queens, finding all paths. For example, subset sum: backtracking finds all subsets that sum to target. DP finds if any subset exists. Backtracking explores decision trees with pruning when you need exhaustive search.',
          keyPoints: [
            'Backtracking: all solutions, constraints, exploration',
            'Greedy: local optimum, no backtracking',
            'DP: overlapping subproblems, memoization',
            'Backtracking for: all solutions, constraint satisfaction',
            'Use when exhaustive search needed with pruning',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the core concept of backtracking?',
          options: [
            'Greedy selection',
            'Make choice, explore, undo choice, try next',
            'Dynamic programming',
            'Sorting',
          ],
          correctAnswer: 1,
          explanation:
            'Backtracking: 1) Make a choice, 2) Explore consequences recursively, 3) Undo the choice (backtrack) if invalid, 4) Try next choice. This explores all possibilities with pruning.',
        },
        {
          id: 'mc2',
          question: 'How does backtracking differ from brute force?',
          options: [
            'They are the same',
            'Backtracking prunes invalid paths early instead of checking validity after',
            'Backtracking is always faster',
            'Brute force is better',
          ],
          correctAnswer: 1,
          explanation:
            'Backtracking stops exploring invalid paths early (pruning), while brute force generates all possibilities first then checks validity. Backtracking is more efficient.',
        },
        {
          id: 'mc3',
          question: 'What keywords signal a backtracking problem?',
          options: [
            'Maximum, minimum',
            'All, generate, combinations, permutations, subsets',
            'Shortest path',
            'Sort, search',
          ],
          correctAnswer: 1,
          explanation:
            'Keywords "all", "generate", "combinations", "permutations", "subsets" indicate you need to explore all possibilities, which is perfect for backtracking.',
        },
        {
          id: 'mc4',
          question: 'Why must you copy the path when adding to results?',
          options: [
            'For speed',
            'Path is mutated during backtracking - copy preserves current state',
            'Random requirement',
            'Uses less memory',
          ],
          correctAnswer: 1,
          explanation:
            'Path is modified during backtracking (choices added/removed). Without copying, all results would reference the same mutated array. path.copy() preserves the current solution state.',
        },
        {
          id: 'mc5',
          question: 'What is the typical space complexity of backtracking?',
          options: [
            'O(1)',
            'O(N) for recursion call stack depth',
            'O(N²)',
            'O(2^N)',
          ],
          correctAnswer: 1,
          explanation:
            'Backtracking space complexity is typically O(N) for the recursion call stack, where N is the depth of recursion (solution length). Output space is not counted.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Compare the subset and permutation patterns. What is the key difference in how they explore choices?',
          sampleAnswer:
            'Subsets explore include/exclude decisions for each element - binary choice at each step. Permutations explore which element to place at each position - n choices initially, n-1 next, etc. Key difference: subsets maintain element order (no rearrangement), permutations try all arrangements. In subsets, once we skip an element, we never go back to it. In permutations, we try each element at each position using a "used" array or swapping. Subsets generate 2^n results (each element in or out). Permutations generate n! results (all arrangements). The recursion tree shape differs: subsets are binary tree, permutations are n-ary tree with shrinking branches.',
          keyPoints: [
            'Subsets: include/exclude binary choice',
            'Permutations: which element at this position',
            'Subsets: maintain order, 2^n results',
            'Permutations: all arrangements, n! results',
            'Tree shape: binary vs n-ary',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the constraint satisfaction pattern with N-Queens. How do you check if a placement is valid?',
          sampleAnswer:
            'N-Queens places N queens on NxN board so none attack each other. We place one queen per row, trying each column. Before placing, check if valid: no queen in same column, no queen on same diagonal. Track used columns with set. For diagonals, use math: row-col identifies one diagonal direction (45 degrees), row+col identifies the other (135 degrees). These are unique per diagonal. So maintain sets for columns, diag1 (row-col), diag2 (row+col). Place queen if position not in any set, add to sets, recurse to next row, then remove from sets (backtrack). This constraint checking prunes invalid placements early. Without it, we would explore all N^N placements.',
          keyPoints: [
            'One queen per row, try each column',
            'Check: column, two diagonals',
            'Track: column set, diag1 (row-col), diag2 (row+col)',
            'Place, add to sets, recurse, remove from sets',
            'Pruning: N^N → much fewer valid placements',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe the path-finding pattern for word search. Why do we need to mark visited cells?',
          sampleAnswer:
            'Word search finds if word exists in grid starting from each cell, moving adjacent (up/down/left/right). From each cell, try all 4 directions recursively if char matches next in word. We mark visited cells to prevent cycles - without marking, we could use same cell multiple times, which violates the problem. Mark current cell as visited before recursing, unmark after returning (backtrack). This ensures each path uses each cell at most once. The visited marking is temporary per path - when we backtrack and try a different direction, previous cells become available again. This is classic backtracking state management: modify (mark), recurse, restore (unmark).',
          keyPoints: [
            'Try starting from each cell',
            'Recurse in 4 directions if char matches',
            'Mark visited to prevent cycles',
            'Unmark after recursion (backtrack)',
            'Temporary marking per path',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the pattern for generating subsets/combinations?',
          options: [
            'Iterate through all',
            'For each element: include it or skip it (two choices per element)',
            'Sort first',
            'Use hash map',
          ],
          correctAnswer: 1,
          explanation:
            'Subsets pattern: for each element, make two recursive calls - one including the element, one excluding it. This generates all 2^N subsets.',
        },
        {
          id: 'mc2',
          question:
            'How do you prevent duplicate permutations with repeated elements?',
          options: [
            'Sort only',
            'Use a set to track which elements used at each recursion level',
            'Random selection',
            'Cannot prevent',
          ],
          correctAnswer: 1,
          explanation:
            'For duplicates: sort array, use set to track used elements at each level. Skip if current element equals previous AND previous not used (prevents duplicate permutations).',
        },
        {
          id: 'mc3',
          question: 'What makes N-Queens a constraint satisfaction problem?',
          options: [
            'It is difficult',
            'Must satisfy constraints: no two queens attack each other (row, column, diagonal)',
            'Uses recursion',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'N-Queens: place N queens on N×N board such that no two queens attack each other. Must satisfy row, column, and diagonal constraints - classic constraint satisfaction.',
        },
        {
          id: 'mc4',
          question: 'What is pruning in backtracking?',
          options: [
            'Removing code',
            'Stop exploring paths early when they cannot lead to valid solutions',
            'Sorting',
            'Random selection',
          ],
          correctAnswer: 1,
          explanation:
            'Pruning stops exploring invalid paths early. E.g., Sudoku: if placing digit violates rules, backtrack immediately instead of continuing. Drastically reduces search space.',
        },
        {
          id: 'mc5',
          question:
            'What is the difference between combinations and permutations?',
          options: [
            'They are the same',
            "Combinations: order doesn't matter [1,2]==[2,1], Permutations: order matters [1,2]!=[2,1]",
            'Permutations are faster',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Combinations: {1,2,3} choosing 2 gives 3 results ([1,2],[1,3],[2,3]) - order irrelevant. Permutations: 6 results ([1,2],[1,3],[2,1],[2,3],[3,1],[3,2]) - order matters.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain why permutations are O(n! × n) while subsets are O(2^n × n). What causes the factorial vs exponential difference?',
          sampleAnswer:
            'Permutations generate all arrangements of n elements: n choices for first position, n-1 for second, n-2 for third, etc. This gives n × (n-1) × (n-2) × ... × 1 = n! permutations. Each takes O(n) to copy, giving O(n! × n). Subsets make binary include/exclude decision for each element: 2 choices per element for n elements gives 2^n subsets. Each takes O(n) to copy, giving O(2^n × n). The difference: permutations have shrinking choices at each level (n, n-1, n-2...), subsets have fixed 2 choices per level. For n=5: permutations = 120, subsets = 32. Factorial grows much faster than exponential for larger n.',
          keyPoints: [
            'Permutations: n choices, then n-1, n-2... = n!',
            'Subsets: 2 choices per element = 2^n',
            'Both: O(n) to copy each solution',
            'Factorial vs exponential: shrinking vs fixed choices',
            'n=5: 120 perms vs 32 subsets',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through N-Queens complexity. Why is it O(n!) rather than O(n^n)?',
          sampleAnswer:
            'N-Queens places one queen per row, trying each column. First row has n choices, but second row has fewer valid choices due to constraints (column and diagonal conflicts). On average, later rows have exponentially fewer valid choices. Without any pruning, it would be O(n^n) - n choices per row for n rows. With constraint checking, we prune invalid placements early, reducing to approximately O(n!) - similar to permutations but even less due to diagonal constraints. The exact complexity is hard to express but empirically closer to n! than n^n. For n=8, n^n would be 16 million, n! is 40 thousand, and actual is even less due to pruning. Constraint checking massively reduces search space.',
          keyPoints: [
            'Without pruning: O(n^n) - n choices per row',
            'With constraints: invalid placements pruned',
            'Reduces to approximately O(n!)',
            'Actually less than n! due to diagonal constraints',
            'n=8: n^n=16M, n!=40K, actual even less',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe early pruning optimization. How much does it improve backtracking performance?',
          sampleAnswer:
            'Early pruning checks constraints before recursing rather than after building complete solution. Bad approach: build entire solution, then check if valid. Good approach: check validity at each step, backtrack immediately if invalid. For example, in N-Queens, check column and diagonal conflicts before placing queen. If invalid, do not recurse to next row. This prevents exploring entire subtrees that cannot succeed. The improvement is exponential - instead of exploring b^d nodes (b = branching, d = depth), we might explore b^(d/2) or less. For N-Queens, without pruning, we explore n^n placements. With pruning, much less. Early pruning is the core of what makes backtracking practical.',
          keyPoints: [
            'Check constraints before recursing, not after',
            'Prevents exploring invalid subtrees',
            'Bad: build complete solution, then check',
            'Good: check at each step, backtrack early',
            'Improvement: exponential reduction in search space',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the time complexity of generating all subsets?',
          options: [
            'O(N)',
            'O(2^N * N) - 2^N subsets, O(N) to copy each',
            'O(N²)',
            'O(N!)',
          ],
          correctAnswer: 1,
          explanation:
            'Subsets: 2^N possible subsets (each element in or out). Each subset takes O(N) to copy into result. Total: O(2^N * N).',
        },
        {
          id: 'mc2',
          question:
            'What is the time complexity of generating all permutations?',
          options: [
            'O(2^N)',
            'O(N! * N) - N! permutations, O(N) to copy each',
            'O(N²)',
            'O(N log N)',
          ],
          correctAnswer: 1,
          explanation:
            'Permutations: N! possible orderings (N choices for first, N-1 for second, etc.). Each takes O(N) to copy. Total: O(N! * N).',
        },
        {
          id: 'mc3',
          question:
            'What optimization technique reduces backtracking search space most?',
          options: [
            'Sorting',
            'Early pruning - check constraints before recursing, not after',
            'Using hash maps',
            'Random selection',
          ],
          correctAnswer: 1,
          explanation:
            'Early pruning checks constraints immediately and backtracks if invalid, preventing exploration of entire invalid subtrees. This exponentially reduces search space.',
        },
        {
          id: 'mc4',
          question: 'What is constraint propagation in backtracking?',
          options: [
            'Sorting constraints',
            'Maintaining state (sets/flags) to avoid recomputing validity checks',
            'Random selection',
            'Removing constraints',
          ],
          correctAnswer: 1,
          explanation:
            'Constraint propagation: maintain state like sets for used columns/diagonals in N-Queens. Avoids O(N) validation each time - check set in O(1) instead.',
        },
        {
          id: 'mc5',
          question:
            'What is the space complexity of backtracking (excluding output)?',
          options: [
            'O(2^N)',
            'O(d) where d is recursion depth/solution length',
            'O(N²)',
            'O(N!)',
          ],
          correctAnswer: 1,
          explanation:
            'Backtracking space (excluding output) is O(d) for recursion stack where d is depth. Path state also O(d). Output space not counted in space complexity.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Walk me through the basic backtracking template. What does each part accomplish?',
          sampleAnswer:
            'The basic template has: result list, helper function, and base case. Helper takes current state (path, used elements, etc.). Base case: if complete solution (path length equals target), add copy of path to result and return. Otherwise, iterate through choices, for each valid choice: add to path (choose), recurse (explore), remove from path (unchoose). The iteration represents branching at each node. Choose-explore-unchoose is the core pattern - ensures state is independent for each branch. Return result after helper explores all paths. This template adapts to subsets, combinations, permutations by changing what choices are and what complete means.',
          keyPoints: [
            'Result list, helper function, base case',
            'Base: complete solution → add to result',
            'Loop through choices',
            'Choose, recurse, unchoose pattern',
            'Adapts to different problem types',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the subset template with start index. Why is start index important for avoiding duplicates?',
          sampleAnswer:
            'Subset template recursively decides include or exclude for each element, using start index to track progress. At each recursion, we have two branches: include nums[start] then recurse with start+1, or skip nums[start] and recurse with start+1. Start index ensures we only consider elements at or after current position, preventing duplicates. Without start index, we would reconsider earlier elements and generate [1,2] and [2,1] as separate subsets - but they are the same subset. Start index maintains order: once we pass an element, we never go back. This gives us exactly 2^n unique subsets. Alternative: pass remaining elements as parameter.',
          keyPoints: [
            'Start index tracks progress through array',
            'Two branches: include or exclude current',
            'Recurse with start+1 in both branches',
            'Prevents reconsidering earlier elements',
            'Avoids duplicates by maintaining order',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe the permutation template with used array. Why swap and track used array?',
          sampleAnswer:
            'Permutation template generates all arrangements. Two approaches: used array or swapping. Used array: maintain boolean array tracking which elements are used. At each position, try each unused element: mark used, add to path, recurse, remove from path, mark unused. This explores all arrangements by trying each element at each position. Swapping approach: swap current position with each position from current to end, recurse, swap back. This implicitly tracks used elements via array partitioning. Used array is clearer but needs O(n) extra space. Swapping is in-place but harder to understand. Both generate n! permutations by trying all orderings.',
          keyPoints: [
            'Two approaches: used array or swapping',
            'Used array: track which elements used',
            'Try each unused at each position',
            'Swapping: swap, recurse, swap back',
            'Both: O(n!) permutations',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the basic backtracking template structure?',
          options: [
            'Loop only',
            'Base case, try choices, recurse, undo choice',
            'Sorting',
            'Hash map',
          ],
          correctAnswer: 1,
          explanation:
            'Template: 1) Base case (solution found), 2) Loop through choices, 3) Make choice and recurse, 4) Undo choice (backtrack). This explores all paths systematically.',
        },
        {
          id: 'mc2',
          question:
            'In subsets template, what are the two choices for each element?',
          options: [
            'Use it twice or skip',
            'Include it or exclude it',
            'Sort or not sort',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Subsets: for each element, two recursive calls - include it (add to path) or exclude it (skip). This binary decision tree generates all 2^N subsets.',
        },
        {
          id: 'mc3',
          question: 'How do you track used elements in permutations template?',
          options: [
            'Array',
            'Set or boolean array to mark used elements',
            'Stack',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Permutations: use set or boolean array to track which elements are already in current path. Check before adding, prevents duplicates in same permutation.',
        },
        {
          id: 'mc4',
          question: 'In word search template, why mark cells as visited?',
          options: [
            'For speed',
            'Prevents using same cell twice in same path (cycles)',
            'Random requirement',
            'Memory optimization',
          ],
          correctAnswer: 1,
          explanation:
            'Word search: mark visited cells to prevent reusing them in same path. After recursion, unmark (backtrack) so cell can be used in different paths. Prevents infinite loops.',
        },
        {
          id: 'mc5',
          question: 'What is common to all backtracking templates?',
          options: [
            'Sorting',
            'Recursive structure with make choice → explore → undo choice pattern',
            'Hash maps',
            'Binary search',
          ],
          correctAnswer: 1,
          explanation:
            'All backtracking templates follow: make choice (add to path), explore (recurse), undo choice (remove from path). This systematic exploration with backtracking is the universal pattern.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'How do you recognize that a problem needs backtracking? What keywords or patterns signal this?',
          sampleAnswer:
            'Several signals indicate backtracking. First, "all possible" or "find all" - generating all solutions. Second, "combinations", "permutations", "subsets" - explicit generation problems. Third, constraint satisfaction: "N-Queens", "Sudoku", "valid placements". Fourth, "can you find a path" or "does there exist" with complex constraints. Fifth, when greedy does not work and you need exhaustive search with pruning. The key question: do I need to explore all possibilities with ability to abandon invalid paths? If yes, backtracking. Examples: "generate all combinations of size k", "find all solutions to puzzle", "count ways to partition".',
          keyPoints: [
            'Keywords: all possible, find all',
            'Explicit: combinations, permutations, subsets',
            'Constraint satisfaction: N-Queens, Sudoku',
            'Path finding with constraints',
            'Need exhaustive search with pruning',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through your approach to a backtracking problem in an interview, from identification to explaining complexity.',
          sampleAnswer:
            'First, I identify the pattern: "generate all subsets, so this is backtracking subset pattern". I clarify: duplicates in input? Empty subset allowed? Then I explain approach: "I will use recursive backtracking with start index. At each step, two choices: include current element or skip it. Base case: when start index reaches end, add current path to result". I discuss complexity: "2^n subsets, each takes O(n) to copy, so O(2^n × n) time. O(n) space for recursion depth". I draw decision tree for small example: [1,2] branches into include 1 or not, then include 2 or not, giving [], [2], [1], [1,2]. While coding, I explain choose-explore-unchoose pattern. Finally, I mention optimizations like early pruning if problem has constraints.',
          keyPoints: [
            'Identify pattern and explain why backtracking',
            'Clarify: duplicates, edge cases',
            'Explain approach with base case and choices',
            'State complexity with reasoning',
            'Draw decision tree for example',
            'Code with choose-explore-unchoose commentary',
          ],
        },
        {
          id: 'q3',
          question:
            'What are the most common mistakes in backtracking problems and how do you avoid them?',
          sampleAnswer:
            'First: forgetting to copy path when adding to result. Paths are references; without copy, all results point to same list that gets modified. Use path[:] or path.copy(). Second: forgetting to backtrack (unchoose). State must be restored for sibling branches. Third: wrong base case - not checking if solution complete. Fourth: not checking constraints before recursing, leading to wasted exploration. Fifth: modifying input accidentally. Sixth: off-by-one with indices, especially start parameter. My strategy: always use choose-explore-unchoose pattern explicitly, test with small examples, verify state restoration, add constraint checks before recursing. Drawing the recursion tree helps catch logic errors.',
          keyPoints: [
            'Copy path when adding to result',
            'Always backtrack (unchoose)',
            'Check correct base case',
            'Check constraints before recursing',
            'Do not modify input accidentally',
            'Test small examples, draw recursion tree',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What keywords signal a backtracking problem?',
          options: [
            'Shortest, fastest',
            'All, generate, combinations, permutations, subsets',
            'Minimum, maximum',
            'Sort, search',
          ],
          correctAnswer: 1,
          explanation:
            'Keywords "all", "generate", "combinations", "permutations", "subsets" indicate exhaustive search needed - perfect for backtracking. "Minimum/maximum" often suggest greedy or DP.',
        },
        {
          id: 'mc2',
          question:
            'What should you clarify first in a backtracking interview?',
          options: [
            'Complexity only',
            'Duplicates allowed? Order matters? Any constraints? Need all or just one?',
            'Language',
            'Nothing',
          ],
          correctAnswer: 1,
          explanation:
            'Clarify: duplicates (affects pruning), order (combinations vs permutations), constraints (early pruning), all vs one solution (affects when to return). These determine implementation.',
        },
        {
          id: 'mc3',
          question: 'What is the most common mistake in backtracking?',
          options: [
            'Using recursion',
            'Forgetting to copy path before adding to result (reference issue)',
            'Good naming',
            'Complexity analysis',
          ],
          correctAnswer: 1,
          explanation:
            'Most common: result.append(path) without copy. Path is mutated, so all results reference same array. Must use path.copy() or path[:] to preserve current state.',
        },
        {
          id: 'mc4',
          question: 'How should you communicate your backtracking solution?',
          options: [
            'Just code',
            'Clarify, explain decision tree/choices, walk through example, analyze complexity',
            'Write fast',
            'Skip explanation',
          ],
          correctAnswer: 1,
          explanation:
            'Structure: 1) Clarify problem, 2) Explain decision tree (what choices at each level), 3) Walk through small example, 4) Code with comments, 5) Complexity (time and space).',
        },
        {
          id: 'mc5',
          question: 'What is a good practice progression for backtracking?',
          options: [
            'Random order',
            'Week 1: Subsets/Combinations, Week 2: Permutations, Week 3: Constraint problems (N-Queens, Sudoku)',
            'Start with hardest',
            'Skip practice',
          ],
          correctAnswer: 1,
          explanation:
            'Progress: Week 1 basics (subsets, combinations) → Week 2 intermediate (permutations, phone numbers) → Week 3 advanced (N-Queens, Sudoku, word search). Build from simple to complex.',
        },
      ],
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
