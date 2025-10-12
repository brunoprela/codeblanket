import { Module } from '@/lib/types';

export const treesModule: Module = {
  id: 'trees',
  title: 'Trees',
  description:
    'Master tree structures, traversals, and recursive problem-solving for hierarchical data.',
  icon: '🌳',
  timeComplexity: 'O(N) for traversals, O(log N) to O(N) for BST operations',
  spaceComplexity: 'O(H) where H is height',
  sections: [
    {
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
    },
    {
      id: 'traversals',
      title: 'Tree Traversals',
      content: `Tree traversals visit each node exactly once in a specific order. There are two main categories: **Depth-First Search (DFS)** and **Breadth-First Search (BFS)**.

**Depth-First Search (DFS) Traversals:**

Given tree:
\`\`\`
      1
     / \\
    2   3
   / \\
  4   5
\`\`\`

**1. Preorder (Root → Left → Right)**
Visit order: **1**, 2, 4, 5, 3
- Process root first
- Useful for: Copying tree, prefix expressions
\`\`\`python
def preorder(root):
    if not root:
        return
    print(root.val)        # Process root
    preorder(root.left)    # Left subtree
    preorder(root.right)   # Right subtree
\`\`\`

**2. Inorder (Left → Root → Right)**
Visit order: 4, 2, 5, **1**, 3
- Process root between subtrees
- **Gives sorted order for BST!**
- Useful for: BST validation, sorted output
\`\`\`python
def inorder(root):
    if not root:
        return
    inorder(root.left)     # Left subtree
    print(root.val)        # Process root
    inorder(root.right)    # Right subtree
\`\`\`

**3. Postorder (Left → Right → Root)**
Visit order: 4, 5, 2, 3, **1**
- Process root last
- Useful for: Deleting tree, postfix expressions
\`\`\`python
def postorder(root):
    if not root:
        return
    postorder(root.left)   # Left subtree
    postorder(root.right)  # Right subtree
    print(root.val)        # Process root
\`\`\`

---

**Breadth-First Search (BFS) / Level Order:**

Visit nodes level by level, left to right.
Visit order: 1, 2, 3, 4, 5

\`\`\`python
from collections import deque

def level_order(root):
    if not root:
        return
    
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        print(node.val)
        
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
\`\`\`

---

**Iterative DFS (Using Stack):**

\`\`\`python
def preorder_iterative(root):
    if not root:
        return
    
    stack = [root]
    
    while stack:
        node = stack.pop()
        print(node.val)
        
        # Push right first (so left is processed first)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
\`\`\`

**Inorder Iterative (More Complex):**
\`\`\`python
def inorder_iterative(root):
    stack = []
    curr = root
    
    while stack or curr:
        # Go to leftmost node
        while curr:
            stack.append(curr)
            curr = curr.left
        
        # Process node
        curr = stack.pop()
        print(curr.val)
        
        # Move to right subtree
        curr = curr.right
\`\`\`

---

**Complexity:**
- **Time**: O(N) - visit each node once
- **Space**:
  - Recursive: O(H) for call stack
  - Iterative: O(H) for stack/queue
  - Where H = height (log N for balanced, N for skewed)`,
      codeExample: `from typing import Optional, List
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def preorder_traversal(root: Optional[TreeNode]) -> List[int]:
    """Preorder: Root -> Left -> Right"""
    result = []
    
    def traverse(node):
        if not node:
            return
        result.append(node.val)      # Root
        traverse(node.left)           # Left
        traverse(node.right)          # Right
    
    traverse(root)
    return result


def inorder_traversal(root: Optional[TreeNode]) -> List[int]:
    """Inorder: Left -> Root -> Right (sorted for BST)"""
    result = []
    
    def traverse(node):
        if not node:
            return
        traverse(node.left)           # Left
        result.append(node.val)       # Root
        traverse(node.right)          # Right
    
    traverse(root)
    return result


def postorder_traversal(root: Optional[TreeNode]) -> List[int]:
    """Postorder: Left -> Right -> Root"""
    result = []
    
    def traverse(node):
        if not node:
            return
        traverse(node.left)           # Left
        traverse(node.right)          # Right
        result.append(node.val)       # Root
    
    traverse(root)
    return result


def level_order_traversal(root: Optional[TreeNode]) -> List[List[int]]:
    """BFS: Level by level"""
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result`,
    },
    {
      id: 'patterns',
      title: 'Common Tree Patterns',
      content: `**Pattern 1: Recursive DFS**

Most tree problems can be solved recursively:
1. Define base case (null node)
2. Recursively solve for left and right
3. Combine results

**Example: Maximum Depth**
\`\`\`python
def max_depth(root):
    if not root:
        return 0
    
    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)
    
    return 1 + max(left_depth, right_depth)
\`\`\`

---

**Pattern 2: Binary Search Tree Operations**

Leverage BST property: left < root < right

**Search:**
\`\`\`python
def search_bst(root, target):
    if not root or root.val == target:
        return root
    
    if target < root.val:
        return search_bst(root.left, target)
    else:
        return search_bst(root.right, target)
\`\`\`

**Insert:**
\`\`\`python
def insert_bst(root, val):
    if not root:
        return TreeNode(val)
    
    if val < root.val:
        root.left = insert_bst(root.left, val)
    else:
        root.right = insert_bst(root.right, val)
    
    return root
\`\`\`

---

**Pattern 3: Path Problems**

Track path from root to target.

**Example: Path Sum**
\`\`\`python
def has_path_sum(root, target_sum):
    if not root:
        return False
    
    # Leaf node check
    if not root.left and not root.right:
        return root.val == target_sum
    
    # Recursively check subtrees
    remaining = target_sum - root.val
    return (has_path_sum(root.left, remaining) or
            has_path_sum(root.right, remaining))
\`\`\`

---

**Pattern 4: Level-by-Level Processing**

Use BFS for level-specific operations.

**Example: Right Side View**
\`\`\`python
def right_side_view(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        
        for i in range(level_size):
            node = queue.popleft()
            
            # Last node in level
            if i == level_size - 1:
                result.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return result
\`\`\`

---

**Pattern 5: Subtree Problems**

Check properties for entire subtrees.

**Example: Same Tree**
\`\`\`python
def is_same_tree(p, q):
    # Both null
    if not p and not q:
        return True
    
    # One null
    if not p or not q:
        return False
    
    # Values different
    if p.val != q.val:
        return False
    
    # Check subtrees
    return (is_same_tree(p.left, q.left) and
            is_same_tree(p.right, q.right))
\`\`\`

---

**Pattern 6: Bottom-Up vs Top-Down**

**Top-Down (Preorder):**
- Pass information down from parent to children
- Example: Depth tracking

\`\`\`python
def top_down(root, depth=0):
    if not root:
        return
    
    # Use depth information
    process(root, depth)
    
    # Pass down to children
    top_down(root.left, depth + 1)
    top_down(root.right, depth + 1)
\`\`\`

**Bottom-Up (Postorder):**
- Gather information from children to parent
- Example: Height calculation

\`\`\`python
def bottom_up(root):
    if not root:
        return 0
    
    # Get info from children
    left_result = bottom_up(root.left)
    right_result = bottom_up(root.right)
    
    # Compute and return result
    return 1 + max(left_result, right_result)
\`\`\``,
    },
    {
      id: 'complexity',
      title: 'Complexity Analysis',
      content: `**Tree Operation Complexities:**

**Binary Search Tree (Balanced):**
| Operation | Average | Worst (Skewed) |
|-----------|---------|----------------|
| Search | O(log N) | O(N) |
| Insert | O(log N) | O(N) |
| Delete | O(log N) | O(N) |
| Min/Max | O(log N) | O(N) |

**Traversal Complexities:**
| Traversal | Time | Space (Recursive) | Space (Iterative) |
|-----------|------|-------------------|-------------------|
| Preorder | O(N) | O(H) | O(H) |
| Inorder | O(N) | O(H) | O(H) |
| Postorder | O(N) | O(H) | O(H) |
| Level Order | O(N) | O(W) | O(W) |

Where:
- N = number of nodes
- H = height (log N for balanced, N for skewed)
- W = maximum width (N/2 for complete trees)

**Common Problem Complexities:**

**Height/Depth:**
- Time: O(N) - visit all nodes
- Space: O(H) - recursion depth

**Diameter:**
- Time: O(N) - visit all nodes
- Space: O(H) - recursion stack

**Lowest Common Ancestor:**
- Time: O(N) - might visit all nodes
- Space: O(H) - recursion depth

**Path Sum:**
- Time: O(N) - might check all paths
- Space: O(H) - recursion + path storage

**Serialize/Deserialize:**
- Time: O(N) - process each node once
- Space: O(N) - store all nodes

**Balanced Tree Check:**
- Time: O(N) - visit each node once
- Space: O(H) - recursion depth

**Key Insights:**

**Space Complexity:**
- **Recursive DFS**: O(H) for call stack
- **Iterative DFS**: O(H) for explicit stack
- **BFS**: O(W) for queue (worst case N/2 for complete tree)

**Height Matters:**
- Balanced tree: H = O(log N)
- Skewed tree: H = O(N)
- Operations are H-dependent, so balance is crucial

**Complete Binary Tree:**
- Height: O(log N)
- Width: O(N/2) at deepest level
- Efficient for heaps and priority queues`,
    },
    {
      id: 'templates',
      title: 'Code Templates',
      content: `**Template 1: Basic Recursive DFS**
\`\`\`python
def dfs_template(root):
    """Generic recursive DFS pattern."""
    # Base case
    if not root:
        return base_value
    
    # Recursive case
    left_result = dfs_template(root.left)
    right_result = dfs_template(root.right)
    
    # Combine results
    return combine(root.val, left_result, right_result)
\`\`\`

**Template 2: BFS Level Order**
\`\`\`python
from collections import deque

def bfs_template(root):
    """Generic BFS level order pattern."""
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result
\`\`\`

**Template 3: BST Search/Insert**
\`\`\`python
def bst_search(root, target):
    """Binary search tree search."""
    if not root or root.val == target:
        return root
    
    if target < root.val:
        return bst_search(root.left, target)
    else:
        return bst_search(root.right, target)


def bst_insert(root, val):
    """Binary search tree insert."""
    if not root:
        return TreeNode(val)
    
    if val < root.val:
        root.left = bst_insert(root.left, val)
    else:
        root.right = bst_insert(root.right, val)
    
    return root
\`\`\`

**Template 4: Path Tracking**
\`\`\`python
def path_template(root, target):
    """Track path to target."""
    def dfs(node, current_path):
        if not node:
            return False
        
        current_path.append(node.val)
        
        # Check if target found
        if node.val == target:
            return True
        
        # Search in subtrees
        if dfs(node.left, current_path) or dfs(node.right, current_path):
            return True
        
        # Backtrack
        current_path.pop()
        return False
    
    path = []
    dfs(root, path)
    return path
\`\`\`

**Template 5: Bottom-Up Aggregation**
\`\`\`python
def bottom_up_template(root):
    """Aggregate information from bottom to top."""
    def helper(node):
        if not node:
            return (0, some_default)  # Return tuple of info
        
        # Get info from children
        left_count, left_data = helper(node.left)
        right_count, right_data = helper(node.right)
        
        # Compute current info
        current_count = 1 + left_count + right_count
        current_data = combine(node.val, left_data, right_data)
        
        # Update global result if needed
        update_result(current_data)
        
        return (current_count, current_data)
    
    helper(root)
    return result
\`\`\`

**Template 6: Iterative Inorder**
\`\`\`python
def inorder_iterative(root):
    """Iterative inorder traversal."""
    result = []
    stack = []
    curr = root
    
    while stack or curr:
        # Go to leftmost
        while curr:
            stack.append(curr)
            curr = curr.left
        
        # Process node
        curr = stack.pop()
        result.append(curr.val)
        
        # Move to right
        curr = curr.right
    
    return result
\`\`\``,
    },
    {
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
    },
  ],
  keyTakeaways: [
    'Trees are hierarchical: each node has 0+ children, with one root and no cycles',
    'Binary trees: each node has ≤ 2 children (left, right)',
    'BST property: left < root < right for all nodes enables O(log N) operations',
    'DFS traversals: preorder (root first), inorder (root middle, sorted for BST), postorder (root last)',
    'BFS/level-order: process nodes level by level using a queue',
    'Recursive solutions: define base case (null), recursively solve subtrees, combine results',
    'Time: O(N) for traversals; Space: O(H) for recursion where H = log N (balanced) to N (skewed)',
    'Use DFS for path problems, BST for search/insert, BFS for level-specific operations',
  ],
  relatedProblems: [
    'invert-binary-tree',
    'validate-bst',
    'binary-tree-max-path-sum',
  ],
};
