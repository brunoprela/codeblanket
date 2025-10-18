/**
 * Common Tree Patterns Section
 */

export const patternsSection = {
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
};
