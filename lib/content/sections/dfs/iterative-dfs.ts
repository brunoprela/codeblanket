/**
 * Iterative DFS with Stack Section
 */

export const iterativedfsSection = {
  id: 'iterative-dfs',
  title: 'Iterative DFS with Stack',
  content: `While recursion is natural for DFS, you can implement it iteratively using an explicit stack. This is useful to avoid stack overflow for deep graphs.

**Why Use Iterative DFS:**
- Avoid recursion depth limits
- More control over the stack
- Can be more efficient in some cases

**Recursive vs Iterative:**

**Recursive:**
\`\`\`python
def dfs_recursive(node):
    if not node:
        return
    process(node)
    dfs_recursive(node.left)
    dfs_recursive(node.right)
\`\`\`

**Iterative:**
\`\`\`python
def dfs_iterative(root):
    if not root:
        return
    
    stack = [root]
    
    while stack:
        node = stack.pop()
        process(node)
        
        # Push right first (so left is processed first)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
\`\`\`

**Iterative Preorder:**
\`\`\`python
def preorder_iterative(root):
    result = []
    stack = [root] if root else []
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    
    return result
\`\`\`

**Iterative Inorder (trickier):**
\`\`\`python
def inorder_iterative(root):
    result = []
    stack = []
    curr = root
    
    while stack or curr:
        # Go to leftmost node
        while curr:
            stack.append(curr)
            curr = curr.left
        
        # Process node
        curr = stack.pop()
        result.append(curr.val)
        
        # Move to right subtree
        curr = curr.right
    
    return result
\`\`\`

**Iterative Postorder (most complex):**
\`\`\`python
def postorder_iterative(root):
    if not root:
        return []
    
    result = []
    stack = [root]
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        
        # Push left first (reverse of preorder)
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    
    return result[::-1]  # Reverse the result
\`\`\``,
};
