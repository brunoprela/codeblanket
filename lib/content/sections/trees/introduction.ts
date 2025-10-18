/**
 * Introduction to Trees Section
 */

export const introductionSection = {
  id: 'introduction',
  title: 'Introduction to Trees',
  content: `A **tree** is a hierarchical data structure consisting of nodes connected by edges, with one node designated as the **root**. Trees are acyclic (no cycles) and connected (path exists between any two nodes).

**Key Terminology:**

- **Root**: Top node with no parent
- **Parent**: Node with children
- **Child**: Node with a parent
- **Leaf**: Node with no children
- **Siblings**: Nodes with the same parent
- **Ancestor**: Node on the path from root to current node
- **Descendant**: Node in the subtree of current node
- **Height**: Length of longest path from node to a leaf
- **Depth/Level**: Length of path from root to node
- **Subtree**: Tree rooted at any node

**Binary Tree:**
Each node has at most 2 children (left and right).

\`\`\`
        1
       / \\
      2   3
     / \\
    4   5
\`\`\`

**Types of Binary Trees:**

**1. Full Binary Tree**
Every node has 0 or 2 children.
\`\`\`
      1
     / \\
    2   3
   / \\
  4   5
\`\`\`

**2. Complete Binary Tree**
All levels filled except possibly the last, which is filled left to right.
\`\`\`
      1
     / \\
    2   3
   / \\
  4   5
\`\`\`

**3. Perfect Binary Tree**
All internal nodes have 2 children, all leaves at same level.
\`\`\`
      1
     / \\
    2   3
   / \\ / \\
  4  5 6  7
\`\`\`

**4. Binary Search Tree (BST)**
Left subtree < node < right subtree (for all nodes).
\`\`\`
      5
     / \\
    3   7
   / \\ / \\
  1  4 6  9
\`\`\`

**5. Balanced Tree**
Height difference between left and right subtrees ≤ 1 for all nodes.

**Python Implementation:**
\`\`\`python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Create a tree:
#     1
#    / \\
#   2   3
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
\`\`\`

**When to Use Trees:**
- Hierarchical relationships (file systems, org charts)
- Fast search, insert, delete (BST: O(log N) average)
- Priority queues (heaps)
- Expression parsing
- Decision-making (decision trees)`,
};
