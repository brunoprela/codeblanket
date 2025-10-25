/**
 * Lowest Common Ancestor (LCA) Section
 */

export const lowestcommonancestorSection = {
  id: 'lowest-common-ancestor',
  title: 'Lowest Common Ancestor (LCA)',
  content: `**Lowest Common Ancestor (LCA)** is the deepest node that is an ancestor of both given nodes in a tree.

**Definition:**
- The LCA of nodes **p** and **q** is the lowest (deepest) node in the tree that has both **p** and **q** as descendants
- A node can be a descendant of itself

**Example:**
\`\`\`
        3
       / \\
      5   1
     / \\ / \\
    6  2 0  8
      / \\
     7   4
\`\`\`

- LCA(5, 1) = 3 (root is only common ancestor)
- LCA(5, 4) = 5 (node can be ancestor of itself)
- LCA(6, 4) = 5 (5 is the lowest node containing both)
- LCA(7, 4) = 2 (2 is parent of both)

---

## Why LCA Matters

**Real-World Applications:**
- **File systems:** Find common parent directory
- **Version control:** Find common ancestor commit
- **Network routing:** Find common network node
- **Biology:** Find common evolutionary ancestor

**Interview Importance:**
- Top 10 most asked tree question at FAANG
- Tests understanding of recursion and tree traversal
- Multiple approaches with different trade-offs
- Foundation for more complex tree problems

---

## Approach 1: Binary Search Tree (BST)

**For BST only:** Use the ordering property.

**Algorithm:**
1. If both nodes are **less than** root → LCA is in **left subtree**
2. If both nodes are **greater than** root → LCA is in **right subtree**  
3. Otherwise → **root is the LCA** (split point)

**Implementation:**
\`\`\`python
def lca_bst (root, p, q):
    """LCA for Binary Search Tree - O(H) time, O(1) space."""
    while root:
        # Both in left subtree
        if p.val < root.val and q.val < root.val:
            root = root.left
        # Both in right subtree
        elif p.val > root.val and q.val > root.val:
            root = root.right
        else:
            # Split point found
            return root
    return None
\`\`\`

**Time:** O(H) where H is height  
**Space:** O(1) - iterative, no recursion

**Key Insight:** BST property tells us which direction to go without exploring both subtrees.

---

## Approach 2: Binary Tree (General Case)

**For any binary tree:** Cannot use ordering, must explore both subtrees.

**Recursive Algorithm:**
1. **Base case:** If root is None or equals p or q, return root
2. **Recurse:** Search for p and q in left and right subtrees
3. **Combine:**
   - If **both** left and right return non-null → root is LCA (split point)
   - If **only left** returns non-null → LCA is in left subtree
   - If **only right** returns non-null → LCA is in right subtree

**Implementation:**
\`\`\`python
def lca_binary_tree (root, p, q):
    """
    LCA for general Binary Tree - O(N) time, O(H) space.
    
    Key insight: The first node where p and q diverge is the LCA.
    """
    # Base case: empty tree or found one of the target nodes
    if not root or root == p or root == q:
        return root
    
    # Recursively search left and right subtrees
    left = lca_binary_tree (root.left, p, q)
    right = lca_binary_tree (root.right, p, q)
    
    # Case 1: Found both nodes in different subtrees
    # → Current root is the split point (LCA)
    if left and right:
        return root
    
    # Case 2: Both nodes in one subtree
    # → Return whichever subtree found something
    return left if left else right
\`\`\`

**Time:** O(N) - might visit all nodes  
**Space:** O(H) - recursion depth

**Why This Works:**

The algorithm works bottom-up:
1. Each recursive call returns the LCA for its subtree
2. When both subtrees return non-null, we found the split point
3. When only one returns non-null, both nodes are in that subtree

**Example Walkthrough:**

For LCA(7, 4) in the tree above:
\`\`\`
lca(3, 7, 4)
  lca(5, 7, 4)
    lca(6, 7, 4) → null (neither found)
    lca(2, 7, 4)
      lca(7, 7, 4) → 7 (found!)
      lca(4, 7, 4) → 4 (found!)
      left=7, right=4 → return 2 (split point!)
    left=null, right=2 → return 2
  left=2, right=null → return 2
left=2, right=null → return 2
\`\`\`

---

## Approach 3: With Parent Pointers

**If nodes have parent pointers:** Treat it like finding intersection of two linked lists.

**Algorithm:**
1. Get paths from both nodes to root
2. Find first common node in paths

**Implementation:**
\`\`\`python
def lca_with_parent (p, q):
    """LCA when nodes have parent pointers - O(H) time, O(H) space."""
    # Store all ancestors of p
    ancestors = set()
    while p:
        ancestors.add (p)
        p = p.parent
    
    # Find first ancestor of q that's also ancestor of p
    while q:
        if q in ancestors:
            return q
        q = q.parent
    
    return None
\`\`\`

**Optimized (Two Pointers):**
\`\`\`python
def lca_with_parent_optimized (p, q):
    """Space-optimized version - O(H) time, O(1) space."""
    # Get depths
    def get_depth (node):
        depth = 0
        while node:
            depth += 1
            node = node.parent
        return depth
    
    depth_p = get_depth (p)
    depth_q = get_depth (q)
    
    # Move deeper node up to same level
    while depth_p > depth_q:
        p = p.parent
        depth_p -= 1
    
    while depth_q > depth_p:
        q = q.parent
        depth_q -= 1
    
    # Move both up until they meet
    while p != q:
        p = p.parent
        q = q.parent
    
    return p
\`\`\`

---

## Complexity Comparison

| Approach | Time | Space | When to Use |
|----------|------|-------|-------------|
| **BST** | O(H) | O(1) | Binary Search Tree only |
| **Binary Tree** | O(N) | O(H) | Any binary tree (most common) |
| **Parent Pointers** | O(H) | O(1) or O(H) | When parent pointers available |

**Note:** H = log N for balanced trees, H = N for skewed trees

---

## Common Variations

**1. LCA of Multiple Nodes**
Find LCA of k nodes instead of just 2.

**2. LCA with Distance**
Return both LCA and distance to both nodes.

**3. LCA in DAG (Directed Acyclic Graph)**
More complex, may have multiple LCAs.

**4. Range LCA Query**
Preprocess tree for O(1) LCA queries (using binary lifting).

---

## Interview Tips

**Problem Recognition:**
- Keywords: "lowest common ancestor", "LCA", "common parent"
- "Find node that is ancestor of both"
- Sometimes disguised: "find merge point", "common node"

**Clarifying Questions:**
1. Is it a BST or general binary tree?
2. Can nodes be null?
3. Are both nodes guaranteed to exist in tree?
4. Can a node be its own ancestor?

**Common Mistakes:**
- Confusing BST approach with general tree approach
- Forgetting base case (node can be ancestor of itself)
- Not handling case where one node is ancestor of other
- Forgetting to check if nodes exist in tree

**Code Structure:**
\`\`\`python
def lowest_common_ancestor (root, p, q):
    # 1. Base case
    if not root or root == p or root == q:
        return root
    
    # 2. Recursive case
    left = lowest_common_ancestor (root.left, p, q)
    right = lowest_common_ancestor (root.right, p, q)
    
    # 3. Combine results
    if left and right:
        return root  # Split point
    return left if left else right
\`\`\`

**Optimization:**
- BST: Use iterative approach (O(1) space)
- General tree: Recursive is cleanest
- Parent pointers: Two-pointer technique

---

## Related Problems

- **Distance Between Nodes:** Find LCA, then calculate distance
- **Kth Ancestor:** Binary lifting technique
- **LCA in N-ary Tree:** Similar logic, check all children
- **Maximum Path Sum:** Uses LCA concept`,
};
