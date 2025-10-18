/**
 * Interview Strategy Section
 */

export const interviewSection = {
  id: 'interview',
  title: 'Interview Strategy',
  content: `**Recognition Signals:**

**Use Tree algorithms when you see:**
- "Binary tree", "Binary search tree"
- "Root", "leaf", "parent", "child"
- "Traversal" (preorder, inorder, postorder, level-order)
- "Ancestor", "descendant", "subtree"
- "Depth", "height", "level"
- Hierarchical relationships
- "Path from root to leaf"

---

**Problem-Solving Steps:**

**Step 1: Clarify Tree Type**
- Binary tree or N-ary tree?
- Binary search tree (BST)?
- Balanced or unbalanced?
- Can there be duplicates?

**Step 2: Choose Traversal**
- **Need sorted order?** → Inorder (for BST)
- **Need to process parents first?** → Preorder (top-down)
- **Need to process children first?** → Postorder (bottom-up)
- **Need level information?** → BFS/Level-order

**Step 3: Recursive vs Iterative**
- **Recursive**: Cleaner, natural for trees
- **Iterative**: Better for very deep trees (avoid stack overflow)
- Most tree problems favor recursive solutions

**Step 4: Identify Pattern**
- Single node processing? → Simple DFS
- Comparing subtrees? → Recursive with left/right results
- Path problems? → DFS with path tracking
- Level problems? → BFS
- BST property? → Binary search approach

**Step 5: Handle Edge Cases**
- Empty tree (root is None)
- Single node
- Skewed tree (all left or all right)
- Complete/perfect tree

---

**Interview Communication:**

**Example: Maximum Depth**

1. **Clarify:**
   - "Is this a binary tree or N-ary?"
   - "Can the tree be empty?"

2. **Explain approach:**
   - "I'll use recursive DFS."
   - "The depth of a node is 1 + max depth of its subtrees."
   - "Base case: null node has depth 0."

3. **Walk through example:**
   \`\`\`
       3
      / \\
     9  20
        / \\
       15  7
   
   max_depth(3):
     max_depth(9) = 1 (leaf)
     max_depth(20):
       max_depth(15) = 1
       max_depth(7) = 1
       return 1 + max(1,1) = 2
     return 1 + max(1,2) = 3
   \`\`\`

4. **Complexity:**
   - "Time: O(N) - visit each node once."
   - "Space: O(H) - recursion stack, where H is height."

5. **Mention alternatives:**
   - "Could also use BFS, counting levels."
   - "Would use O(W) space for queue."

---

**Common Follow-ups:**

**Q: Can you do it iteratively?**
- Show BFS or iterative DFS with explicit stack

**Q: What if the tree is very deep?**
- Discuss stack overflow risk
- Suggest iterative approach

**Q: Can you optimize space?**
- DFS is already O(H)
- For balanced tree, H = O(log N) - optimal

**Q: How would this work for BST?**
- Leverage BST property for pruning/optimization

---

**Practice Plan:**

1. **Basics (Day 1-2):**
   - All traversals (pre/in/post/level)
   - Max Depth, Min Depth
   - Same Tree, Symmetric Tree

2. **BST (Day 3-4):**
   - Validate BST
   - Lowest Common Ancestor (BST)
   - Insert, Delete in BST

3. **Path Problems (Day 5):**
   - Path Sum
   - Binary Tree Maximum Path Sum
   - All Paths from Root to Leaves

4. **Advanced (Day 6-7):**
   - Serialize/Deserialize
   - Lowest Common Ancestor (Binary Tree)
   - Diameter of Tree
   - Balanced Tree Check

5. **Resources:**
   - LeetCode Tree tag (200+ problems)
   - Draw trees for every problem
   - Practice both recursive and iterative`,
};
