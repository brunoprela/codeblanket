/**
 * Introduction to Backtracking Section
 */

export const introductionSection = {
  id: 'introduction',
  title: 'Introduction to Backtracking',
  content: `**Backtracking** is an algorithmic technique for solving problems **recursively** by building candidates for solutions incrementally and **abandoning** (backtracking from) candidates as soon as it determines they cannot lead to a valid solution.

**Core Concept:**1. **Make a choice** (add to current solution)
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
def backtrack (path, choices):
    # Base case: valid solution found
    if is_valid_solution (path):
        result.append (path.copy())  # Save solution
        return
    
    # Try each possible choice
    for choice in get_choices (choices):
        # Make choice
        path.append (choice)
        
        # Explore with this choice
        backtrack (path, updated_choices)
        
        # Undo choice (backtrack)
        path.pop()
\`\`\`

**Common Problem Types:**1. **Combination/Subset** problems
2. **Permutation** problems
3. **Constraint satisfaction** (N-Queens, Sudoku)
4. **Path finding** (mazes, word search)
5. **Partition** problems`,
};
