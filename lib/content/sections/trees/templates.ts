/**
 * Code Templates Section
 */

export const templatesSection = {
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
};
