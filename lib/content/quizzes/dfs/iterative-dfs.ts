/**
 * Quiz questions for Iterative DFS with Stack section
 */

export const iterativedfsQuiz = [
  {
    id: 'q1',
    question:
      'Explain iterative DFS using explicit stack. Why use iterative over recursive?',
    sampleAnswer:
      'Iterative DFS: use explicit stack instead of call stack. Push root, loop: pop node, process, push children (right then left for same order as recursive). For example, tree [1, [2, [4, 5]], 3]: stack=[1], pop 1 push 3,2, stack=[3,2], pop 2 push 5,4, etc. Result: 1,2,4,5,3 (same as recursive preorder). Use iterative when: deep trees (avoid stack overflow), need control over stack, language lacks good recursion support, debugging easier with explicit state. Tradeoff: more code (explicit stack management) vs recursive elegance. For graphs: same pattern, add visited set. Iterative gives you: no recursion depth limit, explicit stack inspection, easier to convert to BFS (replace stack with queue).',
    keyPoints: [
      'Explicit stack replaces call stack',
      'Push root, loop: pop, process, push children',
      'Push right before left for left-first order',
      'Use when: deep trees, stack overflow concerns',
      'Tradeoff: more code vs recursive elegance',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through converting recursive DFS to iterative. What changes?',
    sampleAnswer:
      'Conversion steps: 1) Replace recursion with explicit stack. 2) Push root to stack. 3) Loop while stack not empty. 4) Pop node, process. 5) Push children (order matters). For example, recursive max depth: depth (node) = 1 + max (depth (left), depth (right)). Iterative: stack = [(node, depth)], loop: pop (node, d), if leaf update max, push children with d+1. The state (current depth) that was in recursive parameters goes into stack tuples. Recursive return values become: accumulated in variables (max, sum) or checked in conditions. The key: what was in call stack (parameters, local vars) goes into explicit stack as tuples. Backtracking: happens automatically when popping.',
    keyPoints: [
      'Call stack → explicit stack',
      'Parameters → stack tuples (node, state)',
      'Return values → accumulate in variables',
      'Push children to continue exploration',
      'Order: push right before left',
    ],
  },
  {
    id: 'q3',
    question:
      'Compare recursive vs iterative DFS complexity. Which is more space-efficient?',
    sampleAnswer:
      'Time: both O(V+E) for graphs, O(n) for trees - same, visit each node once. Space: depends on tree shape. Recursive: O(h) where h is height (call stack). Iterative: O(h) for balanced, O(n) worst case (stack holds level). For balanced tree: both O(log n). For skewed tree: both O(n). The difference: recursive uses call stack (limited, typically few MB), iterative uses heap (more memory available). For very deep trees (100K nodes in chain): recursive stackoverflow, iterative works. Most cases: recursive simpler and same space. Use iterative only if: proven stack overflow, need explicit control, or converting to BFS.',
    keyPoints: [
      'Time: both O(V+E) or O(n)',
      'Space: both O(h) typically',
      'Recursive: call stack (limited)',
      'Iterative: heap stack (more memory)',
      'Very deep: iterative safer, else prefer recursive',
    ],
  },
];
