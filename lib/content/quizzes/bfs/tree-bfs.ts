/**
 * Quiz questions for BFS on Trees (Level-Order Traversal) section
 */

export const treebfsQuiz = [
  {
    id: 'q1',
    question:
      'Explain level-order traversal using BFS. How does it differ from DFS traversals?',
    sampleAnswer:
      'Level-order traversal visits nodes level by level (all at depth 0, then depth 1, etc.). BFS natural fit: queue ensures level-by-level processing. Algorithm: queue with root, loop: process all nodes at current level (track level size), enqueue children for next level. For tree [1, [2, [4, 5]], [3]]: level 0: [1], level 1: [2, 3], level 2: [4, 5]. DFS traversals (preorder, inorder, postorder) visit in depth-first order, no level separation. For example, preorder 1,2,4,5,3 mixes levels. Level-order applications: print tree by levels, find level with maximum sum, zigzag traversal. The key: BFS queue naturally groups by level, DFS stack/recursion does not.',
    keyPoints: [
      'Visit nodes level by level',
      'BFS with queue: natural level grouping',
      'Track level size to process level at a time',
      'vs DFS: no level separation',
      'Uses: print levels, level properties, zigzag',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through finding right-side view of binary tree using BFS. What is the pattern?',
    sampleAnswer:
      'Right-side view: values visible from right side (rightmost node at each level). BFS pattern: level-order, at each level take last node. Algorithm: queue with root, loop: process level (level_size = len(queue)), for each node in level add children, after level add last node value to result. For tree [1, [2, [4, 5]], [3]]: level 0 last is 1, level 1 last is 3, level 2 last is 5. Result: [1, 3, 5]. The pattern: level-order + track last node per level. Can also do DFS: preorder, recurse right before left, track depth, add first node seen at each depth (first from right is rightmost). BFS more intuitive for level-based problems.',
    keyPoints: [
      'Right-side view: rightmost at each level',
      'BFS: process level, take last node',
      'Track level size to know when level ends',
      'Alternative DFS: right-first, depth tracking',
      'BFS more intuitive for level problems',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe maximum width of binary tree using BFS. What is the challenge?',
    sampleAnswer:
      'Maximum width: max number of nodes at any level (including nulls between leftmost and rightmost). Challenge: need to track node positions to account for nulls. Algorithm: BFS with (node, position), left child position 2×pos, right child 2×pos+1. At each level, width = last_pos - first_pos + 1. For example, tree [1, [2, 4], 3]: level 0 width 1, level 1: node 2 at pos 0, node 3 at pos 1, width 2. Level 2: node 4 at pos 0 (2×0), width 1. Without position tracking, cannot account for missing nodes. Max width 2. The key: position indexing like array representation of binary tree, allows calculating width with gaps.',
    keyPoints: [
      'Width: includes nulls between leftmost and rightmost',
      'Track: (node, position) in queue',
      'Left: 2×pos, right: 2×pos+1',
      'Width per level: last_pos - first_pos + 1',
      'Accounts for missing nodes via positions',
    ],
  },
];
