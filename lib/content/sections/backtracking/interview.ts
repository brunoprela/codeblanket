/**
 * Interview Strategy Section
 */

export const interviewSection = {
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

**Example: Generate Subsets**1. **Clarify:**
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
❌ result.append (path)  # Reference, will change!
✅ result.append (path.copy()) or result.append (path[:])

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

**Practice Plan:**1. **Basics (Day 1-2):**
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
};
