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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain why trees are hierarchical and what makes them different from linear structures like arrays or linked lists.',
          sampleAnswer:
            'Trees are hierarchical because each node can have multiple children, creating parent-child relationships in levels. A node at level k has children at level k+1, forming a tree-like branching structure. This is fundamentally different from linear structures like arrays or linked lists where elements have at most one predecessor and one successor, forming a single chain. Trees enable representing hierarchical relationships like file systems (directories contain files and subdirectories), organization charts (managers have employees), or DOM structure (elements contain child elements). The hierarchy enables divide-and-conquer algorithms - we can recursively solve for subtrees independently.',
          keyPoints: [
            'Hierarchical: nodes have multiple children',
            'Parent-child relationships in levels',
            'vs Linear: single chain, one predecessor/successor',
            'Represents: file systems, org charts, DOM',
            'Enables divide-and-conquer on subtrees',
          ],
        },
        {
          id: 'q2',
          question:
            'Describe what makes a binary tree special compared to general trees. When would you use binary trees over general trees?',
          sampleAnswer:
            'Binary trees limit each node to at most two children - left and right. This constraint gives structure and enables specific algorithms. Binary Search Trees use this for efficient O(log n) search by maintaining left < parent < right ordering. General trees allow unlimited children per node, more flexible but harder to balance and search efficiently. Binary trees are preferred when you need fast search (BST), when data naturally has binary decisions (decision trees), or when implementing heaps and priority queues. The two-child limit makes many operations simpler to implement and analyze. For data with many children per node like file systems, general trees are better.',
          keyPoints: [
            'Binary: at most two children per node',
            'Enables BST with O(log n) search',
            'General trees: unlimited children',
            'Binary: fast search, binary decisions, heaps',
            'General: when naturally many children',
          ],
        },
        {
          id: 'q3',
          question:
            'What is a Binary Search Tree and what property makes it useful? Walk me through why search is O(log n).',
          sampleAnswer:
            'A Binary Search Tree maintains the property: for every node, all left subtree values are less than node value, and all right subtree values are greater. This ordering enables binary search. To search for a value, I compare with root: if target is less, search left; if greater, search right; if equal, found. Each comparison eliminates half the remaining tree. In a balanced BST with n nodes, height is log n. Since we make one comparison per level, search is O(log n). For example, searching in a balanced tree of 1000 nodes takes at most 10 comparisons (2^10 = 1024). This is why BSTs are powerful - same logarithmic efficiency as binary search on arrays, but with fast insertion and deletion.',
          keyPoints: [
            'Property: left < node < right for all nodes',
            'Enables binary search on tree',
            'Each comparison eliminates half',
            'Balanced tree height: log n',
            'O(log n) search, insert, delete',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What makes trees hierarchical structures?',
          options: [
            'They are stored in memory sequentially',
            'Each node can have multiple children, creating parent-child relationships in levels',
            'They are always sorted',
            'They use arrays internally',
          ],
          correctAnswer: 1,
          explanation:
            'Trees are hierarchical because nodes can have multiple children, creating parent-child relationships across levels. This branching structure differs from linear structures like arrays or linked lists.',
        },
        {
          id: 'mc2',
          question:
            'What is the maximum number of children a node can have in a binary tree?',
          options: ['Unlimited', '2', '3', '1'],
          correctAnswer: 1,
          explanation:
            'Binary trees limit each node to at most 2 children: left and right. This constraint enables specific algorithms like BST with efficient O(log N) search.',
        },
        {
          id: 'mc3',
          question: 'What property defines a Binary Search Tree?',
          options: [
            'All nodes have exactly 2 children',
            'Left subtree < node < right subtree for all nodes',
            'All leaves are at the same level',
            'Height is always balanced',
          ],
          correctAnswer: 1,
          explanation:
            'BST property: for every node, all values in left subtree are less than node value, and all values in right subtree are greater. This ordering enables O(log N) search in balanced trees.',
        },
        {
          id: 'mc4',
          question: 'What is a leaf node?',
          options: [
            'The root node',
            'A node with no children',
            'A node with one child',
            'A node with two children',
          ],
          correctAnswer: 1,
          explanation:
            'A leaf node is a node with no children - it is at the bottom of the tree. Leaves are terminal nodes in tree traversals.',
        },
        {
          id: 'mc5',
          question:
            'In a balanced BST with 1000 nodes, approximately how many comparisons are needed to search for a value?',
          options: ['1000', '10', '100', '500'],
          correctAnswer: 1,
          explanation:
            'A balanced tree with 1000 nodes has height log₂(1000) ≈ 10. Search makes one comparison per level, so at most 10 comparisons are needed (since 2^10 = 1024).',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Compare inorder, preorder, and postorder traversals. When would you use each one and why?',
          sampleAnswer:
            'Inorder (left, node, right) gives sorted order for BSTs - crucial for retrieving data in order. Use it when you need elements in ascending order. Preorder (node, left, right) processes parent before children, useful for copying trees or creating prefix expressions. Use it when parent context is needed before processing children. Postorder (left, right, node) processes children before parent, useful for deleting trees or evaluating postfix expressions. Use it when you need child results before processing parent, like calculating subtree sizes. The key is understanding what information you need when: inorder for sorted data, preorder for top-down processing, postorder for bottom-up processing.',
          keyPoints: [
            'Inorder: sorted order for BSTs',
            'Preorder: parent before children (copy tree)',
            'Postorder: children before parent (delete tree)',
            'Inorder: ascending order',
            'Preorder: top-down, Postorder: bottom-up',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the difference between DFS and BFS. When would you choose one over the other?',
          sampleAnswer:
            'DFS explores as deep as possible before backtracking, using stack (or recursion). BFS explores level by level, using queue. DFS is better when solutions are likely deep in tree, when you need to explore all paths (like counting paths), or when space is limited (O(h) vs O(w) where h is height, w is width). BFS is better for finding shortest path (level-order guarantees minimum depth), when solutions are likely near root, or for level-specific operations. For example, finding minimum depth: BFS stops at first leaf, efficient. Finding maximum depth: DFS natural recursion. The choice depends on tree shape and what you are searching for.',
          keyPoints: [
            'DFS: deep first, stack/recursion, O(h) space',
            'BFS: level by level, queue, O(w) space',
            'DFS: solutions deep, all paths, save space',
            'BFS: shortest path, solutions near root, level operations',
            'Choice depends on tree shape and goal',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk me through how iterative DFS works with an explicit stack. Why does it match recursive behavior?',
          sampleAnswer:
            'Iterative DFS uses explicit stack to simulate recursion. Push root onto stack. While stack not empty: pop node, process it, push right child then left child (right first so left is processed first). This matches recursion because recursion uses call stack implicitly - when we call on left subtree, the right subtree call is waiting on the call stack. Our explicit stack does the same: by pushing right first, it waits at bottom of stack while we process left. The stack stores pending work just like recursion stores pending function calls. Order of pushing (right then left) ensures we process nodes in same order as recursive preorder. Iterative gives us control over stack and avoids stack overflow for deep trees.',
          keyPoints: [
            'Explicit stack simulates call stack',
            'Push right then left (left processed first)',
            'Stack stores pending work',
            'Matches recursive call order',
            'Avoids stack overflow, more control',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What order does inorder traversal visit nodes in a BST?',
          options: [
            'Random order',
            'Ascending sorted order',
            'Descending order',
            'Level order',
          ],
          correctAnswer: 1,
          explanation:
            'Inorder traversal (left, node, right) visits nodes in ascending sorted order for a Binary Search Tree. This is because it processes left (smaller values) before node before right (larger values).',
        },
        {
          id: 'mc2',
          question:
            'Which traversal processes the parent node before its children?',
          options: ['Inorder', 'Preorder', 'Postorder', 'Level order'],
          correctAnswer: 1,
          explanation:
            'Preorder traversal (node, left, right) processes the parent before its children, making it useful for copying trees or creating prefix expressions where parent context is needed first.',
        },
        {
          id: 'mc3',
          question: 'What data structure does BFS use for tree traversal?',
          options: ['Stack', 'Queue', 'Array', 'Hash map'],
          correctAnswer: 1,
          explanation:
            'BFS uses a queue to explore nodes level by level. Nodes are added to the queue and processed in FIFO order, ensuring each level is completed before moving to the next.',
        },
        {
          id: 'mc4',
          question: 'What is the space complexity of recursive DFS?',
          options: ['O(1)', 'O(H) where H is tree height', 'O(N)', 'O(log N)'],
          correctAnswer: 1,
          explanation:
            'Recursive DFS uses O(H) space for the call stack where H is tree height. In the worst case (skewed tree), H = N. In a balanced tree, H = log N.',
        },
        {
          id: 'mc5',
          question: 'When is BFS preferred over DFS?',
          options: [
            'When solutions are deep in the tree',
            'When finding shortest path or solutions near the root',
            'When space is limited',
            'Always',
          ],
          correctAnswer: 1,
          explanation:
            'BFS is preferred for finding shortest paths (minimum depth to leaf) or when solutions are likely near the root. BFS guarantees finding nodes level by level in order of distance from root.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain the recursive DFS pattern for tree problems. What are the three steps and why is this pattern so powerful?',
          sampleAnswer:
            'The recursive DFS pattern has three steps: base case (null check, return default), recurse on children (get results from left and right subtrees), compute current result (use children results and current node). This is powerful because it naturally divides the problem - solve for subtrees recursively, then combine their results. For example, tree height: base case returns 0 for null, recurse gets left and right heights, compute returns 1 + max(left, right). The pattern works for most tree problems: max depth, balanced check, path sum, etc. It leverages trees recursive structure - a tree is root plus two subtrees. Once you master this pattern, many tree problems become straightforward.',
          keyPoints: [
            'Three steps: base case, recurse, compute',
            'Base: null check, return default',
            'Recurse: get children results',
            'Compute: combine results with current node',
            'Works for most tree problems',
          ],
        },
        {
          id: 'q2',
          question:
            'Describe the subtree pattern. How do you validate that a tree is a valid BST using this approach?',
          sampleAnswer:
            'The subtree pattern processes each node with context from ancestors. For BST validation, each node must be within a valid range: left subtree must be less than node, right subtree must be greater. I pass down min and max bounds: for left child, max becomes current node value (everything must be less). For right child, min becomes current node value (everything must be greater). At each node, I check if value is in valid range (min < value < max), then recurse with updated bounds. This ensures BST property for entire tree. Without bounds, just checking node > left and node < right is insufficient - you miss violations deeper in tree. The pattern: pass down ancestor context, check current node against context.',
          keyPoints: [
            'Pass context from ancestors',
            'BST: each node has valid range',
            'Left: update max bound',
            'Right: update min bound',
            'Ensures global BST property',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk me through the path pattern for finding all root-to-leaf paths. How do you track and return paths?',
          sampleAnswer:
            'Path pattern maintains current path as we traverse. Use list to track nodes from root to current. At each node, add it to path, check if leaf (no children), if leaf add copy of path to result. Then recurse on children with updated path. After recursion, remove current node from path (backtrack). The backtracking is crucial - when returning from a subtree, we remove that subtree root so path is correct for the other subtree. For example, after exploring left subtree, we backtrack then explore right subtree with correct path. This DFS with backtracking pattern works for all path problems: path sum, max path, specific target path. Key is maintaining and backtracking the path.',
          keyPoints: [
            'Maintain current path as list',
            'Add node, check if leaf',
            'Recurse with updated path',
            'Backtrack: remove node after recursion',
            'Works for all path problems',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What are the three essential steps in the recursive DFS pattern?',
          options: [
            'Initialize, loop, return',
            'Base case, recurse on children, compute current result',
            'Sort, search, merge',
            'Read, write, update',
          ],
          correctAnswer: 1,
          explanation:
            'The recursive DFS pattern consists of: 1) Base case (null check, return default), 2) Recurse on children (get results from subtrees), 3) Compute current result (combine children results with current node).',
        },
        {
          id: 'mc2',
          question: 'Why is passing bounds necessary when validating a BST?',
          options: [
            'To make it faster',
            'To ensure nodes satisfy BST property relative to all ancestors, not just parent',
            'To save memory',
            'It is not necessary',
          ],
          correctAnswer: 1,
          explanation:
            'Without bounds, checking only node > left and node < right misses violations deeper in the tree. Bounds ensure each node satisfies BST property relative to all ancestors, maintaining the global BST property.',
        },
        {
          id: 'mc3',
          question:
            'What is backtracking in the path pattern and why is it crucial?',
          options: [
            'Going backwards in the tree',
            'Removing nodes from the path after recursion to maintain correct state',
            'Deleting nodes',
            'Reversing the tree',
          ],
          correctAnswer: 1,
          explanation:
            'Backtracking removes the current node from the path after exploring its subtree. This ensures the path is correct when exploring the other subtree. Without backtracking, the path would incorrectly contain nodes from both subtrees.',
        },
        {
          id: 'mc4',
          question:
            'What is the difference between top-down and bottom-up tree traversal?',
          options: [
            'Top-down goes left, bottom-up goes right',
            'Top-down passes info from parent to children, bottom-up gathers info from children to parent',
            'They are the same',
            'Top-down is faster',
          ],
          correctAnswer: 1,
          explanation:
            'Top-down (preorder-style) passes information down from parent to children (e.g., tracking depth). Bottom-up (postorder-style) gathers information from children to compute parent result (e.g., calculating height).',
        },
        {
          id: 'mc5',
          question:
            'When comparing two trees for structure and values, what is the base case?',
          options: [
            'Both are leaves',
            'Both are null (return true) or one is null (return false)',
            'Values are equal',
            'Trees are balanced',
          ],
          correctAnswer: 1,
          explanation:
            'Base case: if both trees are null, they are equal (return true). If only one is null, they differ in structure (return false). Then check if values match and recurse on subtrees.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Compare time complexity of BST operations in balanced vs unbalanced trees. Why does balance matter so much?',
          sampleAnswer:
            'In a balanced BST, operations are O(log n) because height is log n - each comparison eliminates half the remaining nodes. In an unbalanced (skewed) tree, operations degrade to O(n) because height can be n - the tree becomes like a linked list. For example, inserting 1, 2, 3, 4, 5 in order creates a right-skewed tree where searching for 5 requires checking all 5 nodes. Balance matters because it maintains the logarithmic efficiency that makes BSTs useful. This is why self-balancing trees like AVL and Red-Black exist - they guarantee O(log n) by rebalancing after insertions and deletions. Without balance, BSTs lose their advantage over arrays.',
          keyPoints: [
            'Balanced: O(log n), height is log n',
            'Unbalanced: O(n), height can be n',
            'Skewed tree like linked list',
            'Balance maintains logarithmic efficiency',
            'Self-balancing trees guarantee O(log n)',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain why DFS uses O(h) space while BFS uses O(w) space. When is each more space-efficient?',
          sampleAnswer:
            'DFS uses O(h) space because recursive call stack or explicit stack stores nodes along one path from root to current node - at most h nodes where h is height. BFS uses O(w) space because queue stores all nodes at current level - at most w nodes where w is width. DFS is more space-efficient for balanced trees (h = log n, w = n/2) or deep narrow trees. BFS is more space-efficient for shallow wide trees. For a complete binary tree, DFS uses O(log n) space, BFS uses O(n) space - DFS wins. For a skewed tree, DFS uses O(n), BFS uses O(1) - BFS wins. The choice depends on tree shape.',
          keyPoints: [
            'DFS: O(h) for path stack',
            'BFS: O(w) for level queue',
            'DFS better: balanced or deep narrow trees',
            'BFS better: shallow wide trees',
            'Complete tree: DFS wins, Skewed: BFS wins',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk me through why visiting all nodes in any traversal is O(n) time. What work is done at each node?',
          sampleAnswer:
            'Any complete traversal visits each of n nodes exactly once, giving O(n) time. At each node, we do constant work: process the node value, push/pop from stack or queue, make recursive calls. Even though we make two recursive calls (left and right), each call processes a disjoint subtree - no node is visited multiple times. The total work across all nodes is O(n) constant operations. For example, in preorder: visit node (O(1)), recurse left (processes left subtree once), recurse right (processes right subtree once). No overlap. The tree structure ensures each node is reached exactly once through parent links. This is why all basic traversals (inorder, preorder, postorder, level-order) are O(n).',
          keyPoints: [
            'Visit each of n nodes exactly once',
            'Constant work per node',
            'Two recursive calls process disjoint subtrees',
            'No node visited multiple times',
            'All traversals: O(n) time',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the time complexity of search in a balanced BST?',
          options: ['O(N)', 'O(log N)', 'O(1)', 'O(N²)'],
          correctAnswer: 1,
          explanation:
            'In a balanced BST with height log N, each comparison eliminates half the remaining nodes, giving O(log N) time complexity for search, insert, and delete operations.',
        },
        {
          id: 'mc2',
          question:
            'What happens to BST operation complexity in a skewed (unbalanced) tree?',
          options: [
            'Stays O(log N)',
            'Degrades to O(N)',
            'Becomes O(1)',
            'Becomes O(N²)',
          ],
          correctAnswer: 1,
          explanation:
            'In a skewed tree, height can be N (like a linked list). Operations must traverse from root to leaf, taking O(N) time. This is why balance is crucial for BST efficiency.',
        },
        {
          id: 'mc3',
          question: 'What is the space complexity of recursive DFS?',
          options: [
            'O(1)',
            'O(H) where H is tree height',
            'O(N)',
            'O(W) where W is width',
          ],
          correctAnswer: 1,
          explanation:
            'Recursive DFS uses O(H) space for the call stack where H is height. The stack stores nodes along one path from root to current node. In balanced trees H = log N, in skewed trees H = N.',
        },
        {
          id: 'mc4',
          question:
            'What is the space complexity of BFS (level-order traversal)?',
          options: [
            'O(H) where H is height',
            'O(W) where W is maximum width',
            'O(1)',
            'O(log N)',
          ],
          correctAnswer: 1,
          explanation:
            'BFS uses a queue that stores all nodes at the current level. Maximum queue size is the width W of the tree. In a complete binary tree, this can be N/2 nodes at the bottom level.',
        },
        {
          id: 'mc5',
          question: 'Why are all basic tree traversals O(N) time complexity?',
          options: [
            'They use nested loops',
            'They visit each of N nodes exactly once, doing constant work per node',
            'They are slow',
            'They require sorting',
          ],
          correctAnswer: 1,
          explanation:
            'All traversals (inorder, preorder, postorder, level-order) visit each node exactly once and perform constant work at each node. Total work is N × O(1) = O(N).',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Walk me through the basic recursive DFS template. What makes this pattern applicable to so many tree problems?',
          sampleAnswer:
            'The basic recursive DFS template has three parts: null check (base case), recursive calls on children (divide), combine results (conquer). This works for many problems because most tree problems have recursive substructure - answer for tree depends on answers for subtrees. For max depth: base returns 0, recursive gets left and right depths, combine with 1 + max. For sum of nodes: base returns 0, recursive gets left and right sums, combine with node.val + left + right. The template is universal because it matches how trees are defined recursively. Once you recognize a problem fits this pattern, implementation is straightforward - just fill in the three parts.',
          keyPoints: [
            'Three parts: base, recurse, combine',
            'Matches recursive tree structure',
            'Works because: answer depends on subtree answers',
            'Just fill in three parts',
            'Universal pattern for tree problems',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the BFS template with queue. Why do we need to track level size separately for level-order problems?',
          sampleAnswer:
            'BFS template: initialize queue with root, while queue not empty, process nodes. For level-order, we need level size because queue mixes nodes from current level with children added during processing. Without tracking size, we cannot tell when one level ends and next begins. The pattern: before processing level, save queue size (number of nodes at this level), process exactly that many nodes, adding their children. This ensures we process one complete level before starting next. For example, with queue [1,2,3], size is 3, we process exactly 3 nodes, adding their children to queue for next level. Level size separates levels within the queue.',
          keyPoints: [
            'Queue processes level by level',
            'Level size separates levels',
            'Save size before processing',
            'Process exactly size nodes',
            'Children added for next level',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe the top-down vs bottom-up recursive patterns. When would you use each?',
          sampleAnswer:
            'Top-down passes information from ancestors down to descendants as parameters. Use when current node needs context from parents - like passing bounds for BST validation or accumulated path for path problems. Bottom-up returns information from children up to parents as return values. Use when parent needs results computed by children - like tree height, node counts, balanced checks. Top-down: parameters carry down, process before recursing. Bottom-up: return values bubble up, process after recursing. Many problems can use either, but one feels more natural. For example, path sum is natural top-down (pass remaining sum down), tree height is natural bottom-up (return heights up).',
          keyPoints: [
            'Top-down: pass context down as parameters',
            'Bottom-up: return results up as return values',
            'Top-down: parent context needed',
            'Bottom-up: children results needed',
            'Many problems work with either',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What are the three parts of the basic recursive DFS template?',
          options: [
            'Start, middle, end',
            'Null check (base case), recursive calls on children, combine results',
            'Initialize, process, return',
            'Read, compute, write',
          ],
          correctAnswer: 1,
          explanation:
            'The DFS template has three parts: 1) Null check for base case, 2) Recursive calls on children to divide the problem, 3) Combine results from children with current node to conquer.',
        },
        {
          id: 'mc2',
          question:
            'In the BFS template, why do we track level size separately?',
          options: [
            'To make it faster',
            'To separate levels within the queue since it mixes current level with children',
            'To save memory',
            'It is not necessary',
          ],
          correctAnswer: 1,
          explanation:
            'Level size lets us process exactly one level at a time. Without it, we cannot tell when one level ends since the queue mixes current level nodes with their children added during processing.',
        },
        {
          id: 'mc3',
          question: 'When should you use a top-down recursive pattern?',
          options: [
            'Always',
            'When current node needs context from ancestors (pass down as parameters)',
            'When it is faster',
            'Never',
          ],
          correctAnswer: 1,
          explanation:
            'Use top-down when nodes need context from ancestors, like passing bounds for BST validation or accumulated sum for path problems. Information flows down as parameters.',
        },
        {
          id: 'mc4',
          question: 'When should you use a bottom-up recursive pattern?',
          options: [
            'When it uses less memory',
            'When parent needs results computed by children (return up)',
            'Always for trees',
            'Only for balanced trees',
          ],
          correctAnswer: 1,
          explanation:
            'Use bottom-up when parent needs results from children, like computing tree height or checking if balanced. Children compute and return values that parent uses.',
        },
        {
          id: 'mc5',
          question:
            'What is the key advantage of iterative DFS/BFS templates over recursive?',
          options: [
            'They are always faster',
            'Explicit stack/queue control and avoid stack overflow for deep trees',
            'They use less code',
            'They are easier to understand',
          ],
          correctAnswer: 1,
          explanation:
            'Iterative templates give explicit control over the stack/queue and avoid stack overflow issues with very deep recursion. They are especially useful for very deep or unbalanced trees.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'How do you recognize that a problem requires tree algorithms? What keywords signal this?',
          sampleAnswer:
            'Several signals indicate tree problems. First, explicit mention: "binary tree", "BST", "tree node". Second, hierarchical relationship keywords: "parent-child", "ancestor-descendant", "root-to-leaf". Third, traversal language: "inorder", "preorder", "level-order". Fourth, BST-specific: "sorted order", "search in log time". Fifth, structural properties: "balanced", "symmetric", "path sum", "depth", "diameter". The key question: does the problem involve hierarchical data or recursive substructure? Even if not explicitly tree, thinking in tree terms can help - like decision trees for game states. If you see "node", "children", or "recursive structure", consider tree algorithms.',
          keyPoints: [
            'Explicit: binary tree, BST, tree node',
            'Hierarchical: parent-child, ancestor-descendant',
            'Traversal: inorder, preorder, level-order',
            'BST: sorted, search log time',
            'Structural: balanced, symmetric, paths, depth',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through your approach to a tree problem in an interview, from problem statement to explaining complexity.',
          sampleAnswer:
            'First, I clarify: binary tree or BST? Are there null nodes? What should I return? Then I identify the pattern: is it traversal (DFS/BFS), recursive divide-and-conquer, path problem, or BST property? I explain my approach: "I will use recursive DFS, base case for null returns 0, recurse on children, combine results with 1 + max". I draw a small tree example and trace through the recursion step by step, showing how we build up from leaves. While coding, I handle null checks carefully and explain the three recursive steps. After coding, I state complexity: "O(n) time to visit all nodes, O(h) space for recursion where h is height". I mention both approaches if applicable: "Could also do iteratively with stack to avoid recursion overhead".',
          keyPoints: [
            'Clarify: binary/BST, nulls, return value',
            'Identify pattern: traversal, recursive, path, BST',
            'Draw example, trace recursion',
            'Code with null checks and explanation',
            'State time and space complexity',
            'Mention alternative approaches',
          ],
        },
        {
          id: 'q3',
          question:
            'What are the most common mistakes in tree problems and how do you avoid them?',
          sampleAnswer:
            'First: forgetting null check, causing null pointer errors. I always start recursive functions with "if not root: return default". Second: confusing preorder/inorder/postorder - mixing up when to process node vs recurse. I remember: pre = before children, in = between children, post = after children. Third: using wrong base case - returning wrong default value. I think through: what should empty tree return? Fourth: forgetting to pass updated parameters in recursion - like not updating remaining sum in path problems. Fifth: modifying tree structure incorrectly - losing references. I draw pointers before coding. Strategy: always draw example, trace carefully, test null and single node cases.',
          keyPoints: [
            'Null checks: start with if not root',
            'Traversal order: pre/in/post timing',
            'Correct base case return value',
            'Pass updated parameters in recursion',
            'Draw pointers for structure modifications',
            'Test: null, single node cases',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What keywords in a problem description indicate a tree algorithm is needed?',
          options: [
            'Array, sort, search',
            'Binary tree, BST, parent-child, root-to-leaf, inorder',
            'Hash map, frequency, count',
            'String, substring, pattern',
          ],
          correctAnswer: 1,
          explanation:
            'Keywords like "binary tree", "BST", hierarchical relationships ("parent-child", "ancestor"), traversals ("inorder", "preorder"), and structural properties ("balanced", "symmetric", "depth") indicate tree problems.',
        },
        {
          id: 'mc2',
          question:
            'What should you clarify first in a tree interview problem?',
          options: [
            'The test cases',
            "Whether it's a binary tree or BST, and what to return",
            'The programming language',
            'How many nodes',
          ],
          correctAnswer: 1,
          explanation:
            "Always clarify whether it's a binary tree (any values) or BST (ordered values), if there are null nodes, and what the function should return. BST problems can use ordering for optimization.",
        },
        {
          id: 'mc3',
          question: 'What is the most common mistake in tree problems?',
          options: [
            'Using wrong variable names',
            'Forgetting null checks, causing null pointer errors',
            'Using too much memory',
            'Making it too fast',
          ],
          correctAnswer: 1,
          explanation:
            'Forgetting to check if a node is null before accessing its properties causes null pointer errors. Always start recursive functions with "if not root: return default_value".',
        },
        {
          id: 'mc4',
          question:
            'When explaining tree solution complexity, what should you mention?',
          options: [
            'Only time complexity',
            'Time O(N) to visit all nodes, Space O(H) for recursion stack where H is height',
            'Only space complexity',
            'That trees are slow',
          ],
          correctAnswer: 1,
          explanation:
            'Mention time complexity O(N) for visiting all nodes and space complexity O(H) for the recursion stack, where H ranges from log N (balanced) to N (skewed tree).',
        },
        {
          id: 'mc5',
          question:
            'What is a recommended practice progression for tree mastery?',
          options: [
            'Start with the hardest problems',
            'Start with basics (traversals, depth), then BST, then paths, then advanced',
            'Only practice BST problems',
            'Skip practice and memorize solutions',
          ],
          correctAnswer: 1,
          explanation:
            'Progress from basic traversals and depth problems, to BST operations, to path problems, and finally advanced topics like serialization and LCA. This builds intuition incrementally.',
        },
      ],
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
