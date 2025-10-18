/**
 * BFS on Trees (Level-Order Traversal) Section
 */

export const treebfsSection = {
  id: 'tree-bfs',
  title: 'BFS on Trees (Level-Order Traversal)',
  content: `**Level-Order Traversal** visits nodes level by level, left to right.

**Example Tree:**
\`\`\`
      1
     / \\
    2   3
   / \\
  4   5
\`\`\`
**Level-order:** 1, 2, 3, 4, 5

**Basic Template:**
\`\`\`python
from collections import deque

def level_order(root):
    if not root:
        return []
    
    queue = deque([root])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node.val)
        
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    return result
\`\`\`

**Level-by-Level Processing:**
\`\`\`python
def level_order_levels(root):
    if not root:
        return []
    
    queue = deque([root])
    result = []
    
    while queue:
        level = []
        level_size = len(queue)  # Important!
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result
\`\`\`

**Common Variations:**

**Right Side View:**
\`\`\`python
def right_side_view(root):
    if not root:
        return []
    
    queue = deque([root])
    result = []
    
    while queue:
        level_size = len(queue)
        
        for i in range(level_size):
            node = queue.popleft()
            
            # Add rightmost node of each level
            if i == level_size - 1:
                result.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return result
\`\`\`

**Zigzag Level Order:**
\`\`\`python
def zigzag_level_order(root):
    if not root:
        return []
    
    queue = deque([root])
    result = []
    left_to_right = True
    
    while queue:
        level = []
        level_size = len(queue)
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        if not left_to_right:
            level.reverse()
        
        result.append(level)
        left_to_right = not left_to_right
    
    return result
\`\`\``,
};
