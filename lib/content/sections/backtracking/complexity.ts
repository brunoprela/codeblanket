/**
 * Complexity Analysis Section
 */

export const complexitySection = {
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
};
