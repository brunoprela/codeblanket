import { Problem } from '../types';

export const treesProblems: Problem[] = [
  {
    id: 'invert-binary-tree',
    title: 'Invert Binary Tree',
    difficulty: 'Easy',
    description: `Given the \`root\` of a binary tree, invert the tree, and return its root.

**Inverting** a binary tree means swapping the left and right children of every node in the tree.


**Approach:**
Use recursion to swap left and right children at each node. Recursively invert the left and right subtrees, then swap them.

**Note:** This is famously the problem Max Howell (creator of Homebrew) could not solve in his Google interview!`,
    examples: [
      {
        input: 'root = [4,2,7,1,3,6,9]',
        output: '[4,7,2,9,6,3,1]',
        explanation:
          'The tree is inverted, all left-right children are swapped.',
      },
      {
        input: 'root = [2,1,3]',
        output: '[2,3,1]',
      },
      {
        input: 'root = []',
        output: '[]',
      },
    ],
    constraints: [
      'The number of nodes in the tree is in the range [0, 100]',
      '-100 <= Node.val <= 100',
    ],
    hints: [
      'Use recursion to solve this elegantly',
      'Base case: if node is null, return null',
      'Recursively invert left and right subtrees',
      'Swap the left and right children',
      'Can also be solved iteratively using BFS or DFS',
    ],
    starterCode: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def invert_tree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    Invert a binary tree.
    
    Args:
        root: Root of the binary tree
        
    Returns:
        Root of the inverted tree
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[[4, 2, 7, 1, 3, 6, 9]]],
        expected: [4, 7, 2, 9, 6, 3, 1],
      },
      {
        input: [[[2, 1, 3]]],
        expected: [2, 3, 1],
      },
      {
        input: [[[]]],
        expected: [],
      },
      {
        input: [[[1]]],
        expected: [1],
      },
    ],
    solution: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def invert_tree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    Recursive solution.
    Time: O(N), Space: O(H) for recursion stack
    """
    # Base case
    if not root:
        return None
    
    # Recursively invert subtrees
    left = invert_tree(root.left)
    right = invert_tree(root.right)
    
    # Swap children
    root.left = right
    root.right = left
    
    return root


# Alternative: More concise
def invert_tree_concise(root: Optional[TreeNode]) -> Optional[TreeNode]:
    if not root:
        return None
    
    # Swap and recurse in one step
    root.left, root.right = (
        invert_tree_concise(root.right),
        invert_tree_concise(root.left)
    )
    
    return root


# Alternative: Iterative BFS
def invert_tree_iterative(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    Iterative solution using BFS.
    Time: O(N), Space: O(W) where W is max width
    """
    if not root:
        return None
    
    from collections import deque
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        
        # Swap children
        node.left, node.right = node.right, node.left
        
        # Add children to queue
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    return root`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(h) where h is the height of the tree',

    leetcodeUrl: 'https://leetcode.com/problems/invert-binary-tree/',
    youtubeUrl: 'https://www.youtube.com/watch?v=OnSn2XEQ4MY',
    order: 1,
    topic: 'Trees',
    leetcodeUrl: 'https://leetcode.com/problems/invert-binary-tree/',
    youtubeUrl: 'https://www.youtube.com/watch?v=OnSn2XEQ4MY',
  },
  {
    id: 'validate-bst',
    title: 'Validate Binary Search Tree',
    difficulty: 'Medium',
    description: `Given the \`root\` of a binary tree, determine if it is a **valid binary search tree (BST)**.

A valid BST is defined as follows:
- The left subtree of a node contains only nodes with keys **less than** the node's key.
- The right subtree of a node contains only nodes with keys **greater than** the node's key.
- Both the left and right subtrees must also be binary search trees.


**Approach:**
Use recursion with valid range tracking. For each node, ensure its value is within a valid range (min, max). When going left, update max. When going right, update min.

**Common Pitfall:** Don't just compare with immediate children - must validate entire subtrees!`,
    examples: [
      {
        input: 'root = [2,1,3]',
        output: 'true',
        explanation: 'This is a valid BST.',
      },
      {
        input: 'root = [5,1,4,null,null,3,6]',
        output: 'false',
        explanation: "The root's value is 5 but its right child's value is 4.",
      },
      {
        input: 'root = [5,4,6,null,null,3,7]',
        output: 'false',
        explanation:
          'The node with value 3 is in the right subtree of 5, but 3 < 5.',
      },
    ],
    constraints: [
      'The number of nodes in the tree is in the range [1, 10^4]',
      '-2^31 <= Node.val <= 2^31 - 1',
    ],
    hints: [
      'Each node must be within a valid range (min, max)',
      'For left child: max becomes parent value',
      'For right child: min becomes parent value',
      'Use negative/positive infinity for initial bounds',
      'Alternative: Inorder traversal of BST should be strictly increasing',
    ],
    starterCode: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def is_valid_bst(root: Optional[TreeNode]) -> bool:
    """
    Validate if a binary tree is a valid BST.
    
    Args:
        root: Root of the binary tree
        
    Returns:
        True if valid BST, False otherwise
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[[2, 1, 3]]],
        expected: true,
      },
      {
        input: [[[5, 1, 4, null, null, 3, 6]]],
        expected: false,
      },
      {
        input: [[[5, 4, 6, null, null, 3, 7]]],
        expected: false,
      },
      {
        input: [[[1]]],
        expected: true,
      },
    ],
    solution: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def is_valid_bst(root: Optional[TreeNode]) -> bool:
    """
    Range validation approach.
    Time: O(N), Space: O(H) for recursion
    """
    def validate(node, min_val, max_val):
        # Empty node is valid
        if not node:
            return True
        
        # Check if current value is within valid range
        if node.val <= min_val or node.val >= max_val:
            return False
        
        # Validate left subtree: must be < node.val
        # Validate right subtree: must be > node.val
        return (validate(node.left, min_val, node.val) and
                validate(node.right, node.val, max_val))
    
    return validate(root, float('-inf'), float('inf'))


# Alternative: Inorder traversal (should be strictly increasing)
def is_valid_bst_inorder(root: Optional[TreeNode]) -> bool:
    """
    Inorder traversal approach.
    Time: O(N), Space: O(H)
    """
    def inorder(node):
        if not node:
            return True
        
        # Check left subtree
        if not inorder(node.left):
            return False
        
        # Check current node (must be greater than previous)
        if node.val <= self.prev:
            return False
        self.prev = node.val
        
        # Check right subtree
        return inorder(node.right)
    
    self = type('obj', (), {'prev': float('-inf')})()
    return inorder(root)


# Alternative: Iterative inorder
def is_valid_bst_iterative(root: Optional[TreeNode]) -> bool:
    """
    Iterative inorder traversal.
    Time: O(N), Space: O(H)
    """
    stack = []
    prev = float('-inf')
    curr = root
    
    while stack or curr:
        # Go to leftmost
        while curr:
            stack.append(curr)
            curr = curr.left
        
        # Process node
        curr = stack.pop()
        
        # Check if strictly increasing
        if curr.val <= prev:
            return False
        prev = curr.val
        
        # Move to right
        curr = curr.right
    
    return True`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(h) where h is the height of the tree',

    leetcodeUrl: 'https://leetcode.com/problems/validate-binary-search-tree/',
    youtubeUrl: 'https://www.youtube.com/watch?v=s6ATEkipzow',
    order: 2,
    topic: 'Trees',
    leetcodeUrl: 'https://leetcode.com/problems/validate-binary-search-tree/',
    youtubeUrl: 'https://www.youtube.com/watch?v=s6ATEkipzow',
  },
  {
    id: 'binary-tree-max-path-sum',
    title: 'Binary Tree Maximum Path Sum',
    difficulty: 'Hard',
    description: `A **path** in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence **at most once**. Note that the path does not need to pass through the root.

The **path sum** of a path is the sum of the node's values in the path.

Given the \`root\` of a binary tree, return **the maximum path sum** of any **non-empty** path.


**Approach:**
Use post-order DFS. For each node, calculate:
1. Maximum path going through left child
2. Maximum path going through right child
3. Maximum path using current node as highest point (left + node + right)

Track global maximum while returning the maximum single path (for parent to use).`,
    examples: [
      {
        input: 'root = [1,2,3]',
        output: '6',
        explanation:
          'The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 = 6.',
      },
      {
        input: 'root = [-10,9,20,null,null,15,7]',
        output: '42',
        explanation:
          'The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 + 7 = 42.',
      },
    ],
    constraints: [
      'The number of nodes in the tree is in the range [1, 3 * 10^4]',
      '-1000 <= Node.val <= 1000',
    ],
    hints: [
      'Use post-order DFS to process children before parent',
      'For each node, consider: max path through left, right, or both',
      'Track global maximum separately from what you return to parent',
      'Return to parent: max single path (node + max(left, right, 0))',
      'Update global max: consider path through current node (left + node + right)',
      'Use max(0, child) to ignore negative paths',
    ],
    starterCode: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def max_path_sum(root: Optional[TreeNode]) -> int:
    """
    Find the maximum path sum in a binary tree.
    
    Args:
        root: Root of the binary tree
        
    Returns:
        Maximum path sum
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[[1, 2, 3]]],
        expected: 6,
      },
      {
        input: [[[-10, 9, 20, null, null, 15, 7]]],
        expected: 42,
      },
      {
        input: [[[-3]]],
        expected: -3,
      },
      {
        input: [[[2, -1]]],
        expected: 2,
      },
    ],
    solution: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def max_path_sum(root: Optional[TreeNode]) -> int:
    """
    Post-order DFS with global max tracking.
    Time: O(N), Space: O(H) for recursion
    """
    max_sum = [float('-inf')]  # Use list to modify in nested function
    
    def dfs(node):
        if not node:
            return 0
        
        # Get max path sum from children (ignore negative paths)
        left_max = max(0, dfs(node.left))
        right_max = max(0, dfs(node.right))
        
        # Update global max considering path through current node
        current_max = node.val + left_max + right_max
        max_sum[0] = max(max_sum[0], current_max)
        
        # Return max single path for parent to use
        return node.val + max(left_max, right_max)
    
    dfs(root)
    return max_sum[0]


# Alternative: Using class variable
def max_path_sum_class(root: Optional[TreeNode]) -> int:
    """
    Using class variable for cleaner syntax.
    """
    class Solution:
        def __init__(self):
            self.max_sum = float('-inf')
        
        def dfs(self, node):
            if not node:
                return 0
            
            # Recursively get max path from children
            left = max(0, self.dfs(node.left))
            right = max(0, self.dfs(node.right))
            
            # Update global max
            self.max_sum = max(self.max_sum, node.val + left + right)
            
            # Return max single path
            return node.val + max(left, right)
    
    sol = Solution()
    sol.dfs(root)
    return sol.max_sum


# Alternative: Using nonlocal
def max_path_sum_nonlocal(root: Optional[TreeNode]) -> int:
    """
    Using nonlocal keyword.
    """
    max_sum = float('-inf')
    
    def dfs(node):
        nonlocal max_sum
        
        if not node:
            return 0
        
        left = max(0, dfs(node.left))
        right = max(0, dfs(node.right))
        
        # Update max considering current node as highest point
        max_sum = max(max_sum, node.val + left + right)
        
        # Return single path for parent
        return node.val + max(left, right)
    
    dfs(root)
    return max_sum`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(h) where h is the height of the tree',

    leetcodeUrl: 'https://leetcode.com/problems/binary-tree-maximum-path-sum/',
    youtubeUrl: 'https://www.youtube.com/watch?v=Hr5cWUld4vU',
    order: 3,
    topic: 'Trees',
    leetcodeUrl: 'https://leetcode.com/problems/binary-tree-maximum-path-sum/',
    youtubeUrl: 'https://www.youtube.com/watch?v=Hr5cWUld4vU',
  },

  // EASY - Same Tree
  {
    id: 'same-tree',
    title: 'Same Tree',
    difficulty: 'Easy',
    topic: 'Trees',
    description: `Given the roots of two binary trees \`p\` and \`q\`, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.`,
    examples: [
      {
        input: 'p = [1,2,3], q = [1,2,3]',
        output: 'true',
      },
      {
        input: 'p = [1,2], q = [1,null,2]',
        output: 'false',
      },
    ],
    constraints: [
      'The number of nodes in both trees is in the range [0, 100]',
      '-10^4 <= Node.val <= 10^4',
    ],
    hints: [
      'Use recursion to compare nodes',
      'Base cases: both null (true), one null (false)',
    ],
    starterCode: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def is_same_tree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    """
    Check if two binary trees are identical.
    
    Args:
        p: Root of first tree
        q: Root of second tree
        
    Returns:
        True if trees are identical
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [1, 2, 3],
          [1, 2, 3],
        ],
        expected: true,
      },
      {
        input: [
          [1, 2],
          [1, null, 2],
        ],
        expected: false,
      },
      {
        input: [[], []],
        expected: true,
      },
    ],
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(h)',
    leetcodeUrl: 'https://leetcode.com/problems/same-tree/',
    youtubeUrl: 'https://www.youtube.com/watch?v=vRbbcKXCxOw',
  },

  // EASY - Symmetric Tree
  {
    id: 'symmetric-tree',
    title: 'Symmetric Tree',
    difficulty: 'Easy',
    topic: 'Trees',
    description: `Given the \`root\` of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).`,
    examples: [
      {
        input: 'root = [1,2,2,3,4,4,3]',
        output: 'true',
      },
      {
        input: 'root = [1,2,2,null,3,null,3]',
        output: 'false',
      },
    ],
    constraints: [
      'The number of nodes in the tree is in the range [1, 1000]',
      '-100 <= Node.val <= 100',
    ],
    hints: [
      'A tree is symmetric if left subtree is mirror of right subtree',
      'Compare left.left with right.right, left.right with right.left',
    ],
    starterCode: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def is_symmetric(root: Optional[TreeNode]) -> bool:
    """
    Check if binary tree is symmetric.
    
    Args:
        root: Root of tree
        
    Returns:
        True if tree is symmetric
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 2, 2, 3, 4, 4, 3]],
        expected: true,
      },
      {
        input: [[1, 2, 2, null, 3, null, 3]],
        expected: false,
      },
      {
        input: [[1]],
        expected: true,
      },
    ],
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(h)',
    leetcodeUrl: 'https://leetcode.com/problems/symmetric-tree/',
    youtubeUrl: 'https://www.youtube.com/watch?v=Mao9uzxwvmc',
  },

  // EASY - Diameter of Binary Tree
  {
    id: 'diameter-binary-tree',
    title: 'Diameter of Binary Tree',
    difficulty: 'Easy',
    topic: 'Trees',
    description: `Given the \`root\` of a binary tree, return the length of the **diameter** of the tree.

The **diameter** of a binary tree is the **length** of the longest path between any two nodes in a tree. This path may or may not pass through the \`root\`.

The **length** of a path between two nodes is represented by the number of edges between them.`,
    examples: [
      {
        input: 'root = [1,2,3,4,5]',
        output: '3',
        explanation: 'The path [4,2,1,3] or [5,2,1,3] has length 3.',
      },
      {
        input: 'root = [1,2]',
        output: '1',
      },
    ],
    constraints: [
      'The number of nodes in the tree is in the range [1, 10^4]',
      '-100 <= Node.val <= 100',
    ],
    hints: [
      'For each node, diameter through it is left_height + right_height',
      'Track maximum diameter while calculating heights',
    ],
    starterCode: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def diameter_of_binary_tree(root: Optional[TreeNode]) -> int:
    """
    Find diameter of binary tree.
    
    Args:
        root: Root of tree
        
    Returns:
        Length of longest path
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 2, 3, 4, 5]],
        expected: 3,
      },
      {
        input: [[1, 2]],
        expected: 1,
      },
      {
        input: [[1]],
        expected: 0,
      },
    ],
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(h)',
    leetcodeUrl: 'https://leetcode.com/problems/diameter-of-binary-tree/',
    youtubeUrl: 'https://www.youtube.com/watch?v=bkxqA8Rfv04',
  },

  // MEDIUM - Lowest Common Ancestor of BST
  {
    id: 'lowest-common-ancestor-bst',
    title: 'Lowest Common Ancestor of a Binary Search Tree',
    difficulty: 'Medium',
    topic: 'Trees',
    description: `Given a binary search tree (BST), find the lowest common ancestor (LCA) node of two given nodes in the BST.

According to the definition of LCA: "The lowest common ancestor is defined between two nodes \`p\` and \`q\` as the lowest node in T that has both \`p\` and \`q\` as descendants (where we allow **a node to be a descendant of itself**)."`,
    examples: [
      {
        input: 'root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8',
        output: '6',
        explanation: 'The LCA of nodes 2 and 8 is 6.',
      },
      {
        input: 'root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4',
        output: '2',
      },
    ],
    constraints: [
      'The number of nodes in the tree is in the range [2, 10^5]',
      '-10^9 <= Node.val <= 10^9',
      'All Node.val are unique',
      'p != q',
      'p and q will exist in the BST',
    ],
    hints: [
      'Use BST property: left < root < right',
      'If both p and q are less than root, go left',
      'If both p and q are greater than root, go right',
      'Otherwise, current node is LCA',
    ],
    starterCode: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def lowest_common_ancestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    """
    Find LCA of two nodes in BST.
    
    Args:
        root: Root of BST
        p: First node
        q: Second node
        
    Returns:
        LCA node
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[6, 2, 8, 0, 4, 7, 9, null, null, 3, 5], 2, 8],
        expected: 6,
      },
      {
        input: [[6, 2, 8, 0, 4, 7, 9, null, null, 3, 5], 2, 4],
        expected: 2,
      },
    ],
    timeComplexity: 'O(h)',
    spaceComplexity: 'O(1)',
    leetcodeUrl:
      'https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/',
    youtubeUrl: 'https://www.youtube.com/watch?v=gs2LMfuOR9k',
  },

  // MEDIUM - Construct Binary Tree from Inorder and Preorder
  {
    id: 'construct-tree-preorder-inorder',
    title: 'Construct Binary Tree from Preorder and Inorder Traversal',
    difficulty: 'Medium',
    topic: 'Trees',
    description: `Given two integer arrays \`preorder\` and \`inorder\` where \`preorder\` is the preorder traversal of a binary tree and \`inorder\` is the inorder traversal of the same tree, construct and return the binary tree.`,
    examples: [
      {
        input: 'preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]',
        output: '[3,9,20,null,null,15,7]',
      },
      {
        input: 'preorder = [-1], inorder = [-1]',
        output: '[-1]',
      },
    ],
    constraints: [
      '1 <= preorder.length <= 3000',
      'inorder.length == preorder.length',
      '-3000 <= preorder[i], inorder[i] <= 3000',
      'preorder and inorder consist of unique values',
      'Each value of inorder also appears in preorder',
      'preorder is guaranteed to be the preorder traversal of the tree',
      'inorder is guaranteed to be the inorder traversal of the tree',
    ],
    hints: [
      'First element in preorder is always root',
      'Find root in inorder to split left and right subtrees',
      'Recursively build left and right subtrees',
    ],
    starterCode: `from typing import Optional, List

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def build_tree(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    """
    Construct binary tree from preorder and inorder traversals.
    
    Args:
        preorder: Preorder traversal
        inorder: Inorder traversal
        
    Returns:
        Root of constructed tree
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [3, 9, 20, 15, 7],
          [9, 3, 15, 20, 7],
        ],
        expected: [3, 9, 20, null, null, 15, 7],
      },
      {
        input: [[-1], [-1]],
        expected: [-1],
      },
    ],
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    leetcodeUrl:
      'https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/',
    youtubeUrl: 'https://www.youtube.com/watch?v=ihj4IQGZ2zc',
  },

  // MEDIUM - Lowest Common Ancestor of Binary Tree
  {
    id: 'lowest-common-ancestor-binary-tree',
    title: 'Lowest Common Ancestor of a Binary Tree',
    difficulty: 'Medium',
    topic: 'Trees',
    description: `Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the definition of LCA: "The lowest common ancestor is defined between two nodes \`p\` and \`q\` as the lowest node in T that has both \`p\` and \`q\` as descendants (where we allow **a node to be a descendant of itself**)."

**Note:** This is for a **general binary tree**, not a BST. You cannot use the ordering property.`,
    examples: [
      {
        input: 'root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1',
        output: '3',
        explanation: 'The LCA of nodes 5 and 1 is 3.',
      },
      {
        input: 'root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4',
        output: '5',
        explanation:
          'The LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself.',
      },
      {
        input: 'root = [1,2], p = 1, q = 2',
        output: '1',
        explanation: 'The LCA of nodes 1 and 2 is 1.',
      },
    ],
    constraints: [
      'The number of nodes in the tree is in the range [2, 10^5]',
      '-10^9 <= Node.val <= 10^9',
      'All Node.val are unique',
      'p != q',
      'p and q will exist in the tree',
    ],
    hints: [
      'Use recursion to search for both nodes',
      'If current node is p or q, return it',
      'If left and right subtrees both return non-null, current node is LCA',
      'Otherwise, return whichever subtree found something',
    ],
    starterCode: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def lowest_common_ancestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    """
    Find LCA of two nodes in a binary tree.
    
    Args:
        root: Root of binary tree
        p: First node
        q: Second node
        
    Returns:
        LCA node
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[3, 5, 1, 6, 2, 0, 8, null, null, 7, 4], 5, 1],
        expected: 3,
      },
      {
        input: [[3, 5, 1, 6, 2, 0, 8, null, null, 7, 4], 5, 4],
        expected: 5,
      },
      {
        input: [[1, 2], 1, 2],
        expected: 1,
      },
    ],
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(h)',
    solution: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def lowest_common_ancestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    """
    Find LCA of two nodes in a binary tree using recursive approach.
    
    Time: O(N) - might visit all nodes
    Space: O(H) - recursion depth (height of tree)
    
    Key Insight:
    - The LCA is the first node where p and q diverge into different subtrees
    - Use post-order traversal (process children before parent)
    - Return nodes upward and look for split point
    
    Algorithm:
    1. Base case: if root is None or root is p or q, return root
    2. Recursively search left and right subtrees
    3. If both subtrees return non-null → root is LCA (split point)
    4. If only one returns non-null → both nodes in that subtree
    """
    # Base case: empty tree or found one of the target nodes
    if not root or root == p or root == q:
        return root
    
    # Recursively search left and right subtrees
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    
    # Case 1: Found nodes in different subtrees
    # Current root is the split point (LCA)
    if left and right:
        return root
    
    # Case 2: Both nodes in one subtree
    # Return whichever subtree found something
    return left if left else right


# Example usage and explanation
def example():
    """
    Example tree:
            3
           / \\
          5   1
         / \\ / \\
        6  2 0  8
          / \\
         7   4
    
    Example 1: LCA(5, 1)
    - Searching from 3:
      - Left subtree (5): returns 5
      - Right subtree (1): returns 1
      - Both non-null → 3 is LCA
    
    Example 2: LCA(5, 4)
    - Searching from 3:
      - Left subtree (5):
        - Finds 5 immediately (base case)
        - Returns 5
      - Right subtree (1): returns None
      - Only left non-null → return 5
    - Result: 5 (node can be ancestor of itself)
    
    Example 3: LCA(7, 4)
    - Searching from 3:
      - Left subtree (5):
        - Neither 5 is target, search children
        - Left (6): returns None
        - Right (2):
          - Left (7): returns 7
          - Right (4): returns 4
          - Both non-null → 2 is LCA
        - Right returned 2 → return 2
      - Right subtree (1): returns None
      - Only left non-null → return 2
    - Result: 2
    """
    pass


# Alternative: Iterative with Parent Pointers (if nodes have parent)
def lca_with_parent(p: TreeNode, q: TreeNode) -> TreeNode:
    """
    Find LCA using parent pointers (if available).
    
    Time: O(H) - traverse up to root
    Space: O(H) - store ancestors in set
    
    Approach: Similar to finding intersection of two linked lists
    """
    # Store all ancestors of p
    ancestors = set()
    while p:
        ancestors.add(p)
        p = p.parent
    
    # Find first ancestor of q that's also ancestor of p
    while q:
        if q in ancestors:
            return q
        q = q.parent
    
    return None


# Comparison: BST vs Binary Tree LCA
"""
Binary Search Tree LCA:
- Can use BST ordering property
- If both nodes < root → go left
- If both nodes > root → go right
- Otherwise → root is LCA
- Time: O(H), Space: O(1) with iteration

Binary Tree LCA:
- No ordering property
- Must explore both subtrees
- Use recursive post-order traversal
- Time: O(N), Space: O(H) for recursion

Key Difference:
BST property allows us to determine direction without exploring
both subtrees, enabling iterative O(1) space solution.
"""`,
    difficulty: 'Medium',
    leetcodeUrl:
      'https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/',
    youtubeUrl: 'https://www.youtube.com/watch?v=13m9ZCB8gjw',
  },
  {
    id: 'serialize-deserialize-binary-tree',
    title: 'Serialize and Deserialize Binary Tree',
    difficulty: 'Hard',
    description: `Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

**Clarification:** The input/output format is the same as how LeetCode serializes a binary tree. You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.


**Why This Problem Matters:**

This is one of the most asked tree problems at FAANG companies because it tests:
- Deep understanding of tree traversal
- String manipulation skills
- Edge case handling (null nodes, single nodes, etc.)
- Design skills (choosing the right format)

**Key Concepts:**

1. **Serialization Strategy:** Convert tree to string representation
   - Can use pre-order, level-order, or post-order traversal
   - Must encode null nodes to preserve structure
   
2. **Deserialization Strategy:** Reconstruct tree from string
   - Parse the string and rebuild nodes
   - Must handle null markers correctly
   
3. **Format Choices:**
   - Delimiter-based: "1,2,3,null,null,4,5"
   - Bracket-based: "1(2)(3(4)(5))"
   - Level-order vs Pre-order

**Common Approaches:**

**Approach 1: Pre-order Traversal (Recommended)**
- Serialize: Root → Left → Right, mark nulls
- Deserialize: Build root, recursively build left & right
- Time: O(N), Space: O(N)
- Pros: Simple, intuitive, recursive

**Approach 2: Level-order Traversal (BFS)**
- Serialize: Use queue, process level by level
- Deserialize: Use queue, connect nodes level by level  
- Time: O(N), Space: O(N)
- Pros: Matches LeetCode format

**Approach 3: Post-order Traversal**
- Serialize: Left → Right → Root
- Deserialize: Build from end of string
- Time: O(N), Space: O(N)
- Less common in interviews`,
    examples: [
      {
        input: 'root = [1,2,3,null,null,4,5]',
        output: '[1,2,3,null,null,4,5]',
        explanation:
          'Tree structure is preserved after serialization and deserialization.',
      },
      {
        input: 'root = []',
        output: '[]',
        explanation: 'Empty tree case.',
      },
      {
        input: 'root = [1]',
        output: '[1]',
        explanation: 'Single node tree.',
      },
      {
        input: 'root = [1,2]',
        output: '[1,2]',
        explanation: 'Tree with only left child.',
      },
    ],
    constraints: [
      'The number of nodes in the tree is in the range [0, 10^4]',
      '-1000 <= Node.val <= 1000',
    ],
    hints: [
      'Use pre-order traversal for serialization (root, left, right)',
      'Mark null nodes explicitly with a special marker like "null" or "N"',
      'Use a delimiter like comma to separate values',
      'For deserialization, use an iterator or index to track position',
      'Build the tree recursively during deserialization',
      'Handle edge cases: empty tree, single node, only left/right children',
    ],
    starterCode: `# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Codec:
    def serialize(self, root: TreeNode) -> str:
        """Encodes a tree to a single string.
        
        Args:
            root: Root of binary tree
            
        Returns:
            String representation of the tree
        """
        pass
    
    def deserialize(self, data: str) -> TreeNode:
        """Decodes your encoded data to tree.
        
        Args:
            data: String representation from serialize()
            
        Returns:
            Root of reconstructed binary tree
        """
        pass

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))`,
    testCases: [
      {
        input: [1, 2, 3, null, null, 4, 5],
        expected: [1, 2, 3, null, null, 4, 5],
      },
      {
        input: [],
        expected: [],
      },
      {
        input: [1],
        expected: [1],
      },
      {
        input: [1, 2],
        expected: [1, 2],
      },
    ],
    solution: `# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Codec:
    """
    Serialize and Deserialize Binary Tree using Pre-order Traversal.
    
    Time Complexity: O(N) for both operations
    Space Complexity: O(N) for storing the string and recursion stack
    
    Strategy:
    - Serialize: Pre-order traversal (root, left, right) with null markers
    - Deserialize: Build tree recursively using iterator
    
    Format: "1,2,N,N,3,4,N,N,5,N,N"
    Where N represents null nodes and commas are delimiters
    """
    
    def serialize(self, root: TreeNode) -> str:
        """
        Encodes a tree to a single string using pre-order traversal.
        
        Example: Tree [1,2,3,null,null,4,5]
                 1
                / \\
               2   3
                  / \\
                 4   5
        
        Serialize as: "1,2,N,N,3,4,N,N,5,N,N"
        Breakdown:
        - Visit 1 → "1"
        - Visit 2 → "1,2"
        - 2.left is null → "1,2,N"
        - 2.right is null → "1,2,N,N"
        - Visit 3 → "1,2,N,N,3"
        - Visit 4 → "1,2,N,N,3,4"
        - 4.left is null → "1,2,N,N,3,4,N"
        - 4.right is null → "1,2,N,N,3,4,N,N"
        - Visit 5 → "1,2,N,N,3,4,N,N,5"
        - 5.left is null → "1,2,N,N,3,4,N,N,5,N"
        - 5.right is null → "1,2,N,N,3,4,N,N,5,N,N"
        """
        def dfs(node):
            if not node:
                return "N"
            
            # Pre-order: Root → Left → Right
            return f"{node.val},{dfs(node.left)},{dfs(node.right)}"
        
        return dfs(root)
    
    def deserialize(self, data: str) -> TreeNode:
        """
        Decodes string to tree using recursive pre-order reconstruction.
        
        Example: data = "1,2,N,N,3,4,N,N,5,N,N"
        
        Parse as list: ["1", "2", "N", "N", "3", "4", "N", "N", "5", "N", "N"]
        Use iterator to track current position:
        
        1. Read "1" → Create node(1)
        2. Build left subtree:
           - Read "2" → Create node(2)
           - Build 2.left: Read "N" → None
           - Build 2.right: Read "N" → None
           - Return node(2)
        3. Build right subtree:
           - Read "3" → Create node(3)
           - Build 3.left:
             - Read "4" → Create node(4)
             - Build 4.left: Read "N" → None
             - Build 4.right: Read "N" → None
             - Return node(4)
           - Build 3.right:
             - Read "5" → Create node(5)
             - Build 5.left: Read "N" → None
             - Build 5.right: Read "N" → None
             - Return node(5)
           - Return node(3)
        4. Return node(1) as root
        """
        def dfs():
            val = next(vals)
            
            # Base case: null marker
            if val == "N":
                return None
            
            # Create node and recursively build subtrees
            node = TreeNode(int(val))
            node.left = dfs()   # Build left subtree
            node.right = dfs()  # Build right subtree
            
            return node
        
        # Convert string to iterator for sequential access
        vals = iter(data.split(","))
        return dfs()


# Alternative Solution: Level-order (BFS) approach
class CodecBFS:
    """
    Alternative using level-order traversal (matches LeetCode format).
    
    Time: O(N), Space: O(N)
    
    Pros: Matches standard tree representation
    Cons: Slightly more complex with queue management
    """
    
    def serialize(self, root: TreeNode) -> str:
        if not root:
            return ""
        
        result = []
        queue = [root]
        
        while queue:
            node = queue.pop(0)
            
            if node:
                result.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                result.append("N")
        
        return ",".join(result)
    
    def deserialize(self, data: str) -> TreeNode:
        if not data:
            return None
        
        vals = data.split(",")
        root = TreeNode(int(vals[0]))
        queue = [root]
        i = 1
        
        while queue and i < len(vals):
            node = queue.pop(0)
            
            # Process left child
            if vals[i] != "N":
                node.left = TreeNode(int(vals[i]))
                queue.append(node.left)
            i += 1
            
            # Process right child
            if i < len(vals) and vals[i] != "N":
                node.right = TreeNode(int(vals[i]))
                queue.append(node.right)
            i += 1
        
        return root


"""
Comparison of Approaches:

Pre-order (Recommended for interviews):
✅ Simple, clean recursive solution
✅ Easy to explain and code
✅ Natural tree traversal
✅ Minimal state tracking (just iterator)

Level-order:
✅ Matches LeetCode format
✅ Intuitive (processes level by level)
⚠️ More complex with queue management
⚠️ More variables to track

Interview Tips:
1. Start with pre-order - cleaner and easier to code under pressure
2. Explain your choice of delimiter and null marker
3. Handle edge cases: empty tree, single node
4. Test with asymmetric trees (only left or right children)
5. Time/Space complexity: Both O(N)

Common Mistakes:
❌ Forgetting to mark null nodes → can't reconstruct structure
❌ Not using delimiter → can't parse multi-digit numbers
❌ Off-by-one errors in level-order deserialization
❌ Not handling empty tree case
"""`,
    timeComplexity: 'O(N) for both serialize and deserialize',
    spaceComplexity: 'O(N) for the string and recursion stack',
    order: 10,
    topic: 'Trees',
    leetcodeUrl:
      'https://leetcode.com/problems/serialize-and-deserialize-binary-tree/',
    youtubeUrl: 'https://www.youtube.com/watch?v=u4JAi2JJhI8',
  },
];
