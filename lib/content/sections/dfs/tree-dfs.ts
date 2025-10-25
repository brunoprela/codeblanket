/**
 * DFS on Trees Section
 */

export const treedfsSection = {
  id: 'tree-dfs',
  title: 'DFS on Trees',
  content: `**Tree DFS** is the simplest form of DFS. Trees have no cycles, making recursion straightforward.

**Three Main Orders:**

**1. Preorder (Root → Left → Right)**
- Process root first
- Used for: copying trees, prefix expressions
\`\`\`python
def preorder (root):
    if not root:
        return
    print(root.val)        # Process root
    preorder (root.left)    # Left subtree
    preorder (root.right)   # Right subtree
\`\`\`

**2. Inorder (Left → Root → Right)**
- Process root in the middle
- Used for: BST sorted output, validation
\`\`\`python
def inorder (root):
    if not root:
        return
    inorder (root.left)     # Left subtree
    print(root.val)        # Process root
    inorder (root.right)    # Right subtree
\`\`\`

**3. Postorder (Left → Right → Root)**
- Process root last
- Used for: deleting trees, postfix expressions
\`\`\`python
def postorder (root):
    if not root:
        return
    postorder (root.left)   # Left subtree
    postorder (root.right)  # Right subtree
    print(root.val)        # Process root
\`\`\`

**Common Tree DFS Patterns:**

**Pattern 1: Calculate property for each node**
\`\`\`python
def max_depth (root):
    if not root:
        return 0
    left = max_depth (root.left)
    right = max_depth (root.right)
    return 1 + max (left, right)
\`\`\`

**Pattern 2: Path-based problems**
\`\`\`python
def has_path_sum (root, target):
    if not root:
        return False
    if not root.left and not root.right:
        return root.val == target
    target -= root.val
    return (has_path_sum (root.left, target) or 
            has_path_sum (root.right, target))
\`\`\``,
};
