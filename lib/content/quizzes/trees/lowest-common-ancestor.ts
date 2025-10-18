/**
 * Quiz questions for Lowest Common Ancestor (LCA) section
 */

export const lowestcommonancestorQuiz = [
  {
    id: 'q1',
    question:
      'Walk me through how the recursive LCA algorithm determines the lowest common ancestor. Why does returning root when both subtrees return non-null give us the LCA?',
    sampleAnswer:
      "The recursive LCA works bottom-up by returning found nodes upward. When we call lca(root, p, q), we recursively search left and right subtrees. If left subtree returns p and right subtree returns q (or vice versa), it means p and q are in different subtrees of root - so root must be their LCA (the split point). If both results come from one subtree, we pass that result up because both nodes are deeper. The base case 'if root == p or root == q: return root' ensures we return nodes when found. This works because the first ancestor where paths diverge is by definition the lowest common ancestor.",
    keyPoints: [
      'Works bottom-up, returning found nodes upward',
      'left and right both non-null → split point found',
      'Only one non-null → both nodes in that subtree',
      'Base case returns node when found',
      'First divergence point is LCA',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain why LCA for BST can be solved in O(1) space but general binary tree requires O(H) space. What property enables the optimization?',
    sampleAnswer:
      'BST LCA can use O(1) space with an iterative approach because the BST property (left < root < right) tells us exactly which direction to go - we never need to explore both subtrees. If both nodes are less than root, go left; if both are greater, go right; otherwise root is the LCA. We can iterate down without recursion. For general binary trees, we lack ordering information, so we must explore both subtrees to find where p and q are located. This requires recursion (or explicit stack), using O(H) space to store the call stack. The BST ordering property eliminates the need to explore both branches.',
    keyPoints: [
      'BST: ordering determines direction, no recursion needed',
      'Iterate down: O(1) space',
      'Binary tree: must check both subtrees',
      'Recursion needed: O(H) space',
      'BST property enables optimization',
    ],
  },
  {
    id: 'q3',
    question:
      'In the recursive LCA algorithm, why is the base case "if root == p or root == q: return root" correct? What about the case where one node is an ancestor of the other?',
    sampleAnswer:
      'The base case "return root" when we encounter p or q is correct because: 1) If we find p or q, we return it upward. 2) If one node (say p) is an ancestor of the other (q), when we encounter p, we return it immediately. The recursive calls below p will search for q, find it, and return it. But since we already returned p, the upper recursive call receives p from one subtree and null from the other, correctly returning p as the LCA. This handles the ancestor case elegantly - the higher node gets returned first, and it is indeed the LCA. The problem statement allows a node to be its own ancestor, so this works perfectly.',
    keyPoints: [
      'Base case returns first match encountered',
      'If one is ancestor: it gets returned first',
      "Other node found below, but we've already returned",
      'Upper call gets p from one side, null from other',
      'Correctly returns p as LCA',
    ],
  },
];
