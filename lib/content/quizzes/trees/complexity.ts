/**
 * Quiz questions for Complexity Analysis section
 */

export const complexityQuiz = [
  {
    id: 'q1',
    question:
      'Compare time complexity of BST operations in balanced vs unbalanced trees. Why does balance matter so much?',
    sampleAnswer:
      'In a balanced BST, operations are O(log n) because height is log n - each comparison eliminates half the remaining nodes. In an unbalanced (skewed) tree, operations degrade to O(n) because height can be n - the tree becomes like a linked list. For example, inserting 1, 2, 3, 4, 5 in order creates a right-skewed tree where searching for 5 requires checking all 5 nodes. Balance matters because it maintains the logarithmic efficiency that makes BSTs useful. This is why self-balancing trees like AVL and Red-Black exist - they guarantee O(log n) by rebalancing after insertions and deletions. Without balance, BSTs lose their advantage over arrays.',
    keyPoints: [
      'Balanced: O(log n), height is log n',
      'Unbalanced: O(n), height can be n',
      'Skewed tree like linked list',
      'Balance maintains logarithmic efficiency',
      'Self-balancing trees guarantee O(log n)',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain why DFS uses O(h) space while BFS uses O(w) space. When is each more space-efficient?',
    sampleAnswer:
      'DFS uses O(h) space because recursive call stack or explicit stack stores nodes along one path from root to current node - at most h nodes where h is height. BFS uses O(w) space because queue stores all nodes at current level - at most w nodes where w is width. DFS is more space-efficient for balanced trees (h = log n, w = n/2) or deep narrow trees. BFS is more space-efficient for shallow wide trees. For a complete binary tree, DFS uses O(log n) space, BFS uses O(n) space - DFS wins. For a skewed tree, DFS uses O(n), BFS uses O(1) - BFS wins. The choice depends on tree shape.',
    keyPoints: [
      'DFS: O(h) for path stack',
      'BFS: O(w) for level queue',
      'DFS better: balanced or deep narrow trees',
      'BFS better: shallow wide trees',
      'Complete tree: DFS wins, Skewed: BFS wins',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through why visiting all nodes in any traversal is O(n) time. What work is done at each node?',
    sampleAnswer:
      'Any complete traversal visits each of n nodes exactly once, giving O(n) time. At each node, we do constant work: process the node value, push/pop from stack or queue, make recursive calls. Even though we make two recursive calls (left and right), each call processes a disjoint subtree - no node is visited multiple times. The total work across all nodes is O(n) constant operations. For example, in preorder: visit node (O(1)), recurse left (processes left subtree once), recurse right (processes right subtree once). No overlap. The tree structure ensures each node is reached exactly once through parent links. This is why all basic traversals (inorder, preorder, postorder, level-order) are O(n).',
    keyPoints: [
      'Visit each of n nodes exactly once',
      'Constant work per node',
      'Two recursive calls process disjoint subtrees',
      'No node visited multiple times',
      'All traversals: O(n) time',
    ],
  },
];
