/**
 * Common DFS Patterns Section
 */

export const patternsSection = {
  id: 'patterns',
  title: 'Common DFS Patterns',
  content: `**Pattern 1: Top-Down (Pass information down)**
\`\`\`python
def path_sum (root, target):
    def dfs (node, remaining):
        if not node:
            return False
        if not node.left and not node.right:
            return remaining == node.val
        return (dfs (node.left, remaining - node.val) or
                dfs (node.right, remaining - node.val))
    return dfs (root, target)
\`\`\`

**Pattern 2: Bottom-Up (Return information up)**
\`\`\`python
def max_depth (root):
    if not root:
        return 0
    left = max_depth (root.left)
    right = max_depth (root.right)
    return 1 + max (left, right)
\`\`\`

**Pattern 3: Path Tracking**
\`\`\`python
def all_paths (root):
    paths = []
    def dfs (node, path):
        if not node:
            return
        path.append (node.val)
        if not node.left and not node.right:
            paths.append (path[:])
        dfs (node.left, path)
        dfs (node.right, path)
        path.pop()  # Backtrack
    dfs (root, [])
    return paths
\`\`\`

**Pattern 4: Validating Trees**
\`\`\`python
def is_valid_bst (root):
    def dfs (node, min_val, max_val):
        if not node:
            return True
        if node.val <= min_val or node.val >= max_val:
            return False
        return (dfs (node.left, min_val, node.val) and
                dfs (node.right, node.val, max_val))
    return dfs (root, float('-inf'), float('inf'))
\`\`\`

**Pattern 5: Connected Components (Graph)**
\`\`\`python
def count_components (graph):
    visited = set()
    count = 0
    
    def dfs (node):
        visited.add (node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs (neighbor)
    
    for node in graph:
        if node not in visited:
            dfs (node)
            count += 1
    return count
\`\`\``,
};
