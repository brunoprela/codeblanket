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
];
