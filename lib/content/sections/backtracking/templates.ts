/**
 * Code Templates Section
 */

export const templatesSection = {
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
            result.append(['.join(r) for r in board])
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
};
