/**
 * Tree Traversals Section
 */

export const traversalsSection = {
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
};
