/**
 * Quiz questions for Common Queue Problems & Patterns section
 */

export const commonproblemsQuiz = [
  {
    id: 'q1',
    question:
      'Explain why BFS uses a queue instead of a stack. What would happen if you used a stack?',
    sampleAnswer:
      'BFS uses a queue because FIFO order ensures we explore all nodes at depth d before any node at depth d+1. Queue processes nodes in the order they were discovered. If we used a stack (LIFO), we would get DFS instead: we would go as deep as possible before backtracking. For example, in a tree, BFS with queue visits root, then all level 1 nodes, then all level 2 nodes (level-order). With a stack, we would visit root, then immediately go deep down one branch before exploring other branches at level 1. The traversal order would be completely different - depth-first rather than breadth-first.',
    keyPoints: [
      'Queue FIFO: process nodes in discovery order',
      'Ensures all depth d before depth d+1',
      'Stack would give DFS (depth-first)',
      'Stack goes deep immediately, queue goes wide',
      'Different traversal order: level-by-level vs deep-first',
    ],
  },
  {
    id: 'q2',
    question:
      'How do you implement a queue using two stacks? Explain the enqueue and dequeue operations.',
    sampleAnswer:
      "Two-stacks queue uses stack_in for enqueue and stack_out for dequeue. Enqueue: push to stack_in - O(1). Dequeue: if stack_out is empty, transfer all from stack_in to stack_out (reversing order), then pop from stack_out. Transfer is expensive O(n), but it's amortized O(1) because each element is transferred at most once. Example: enqueue 1,2,3 to stack_in [1,2,3]. First dequeue transfers to stack_out [3,2,1], then pops 1. Next dequeues just pop from stack_out (2, then 3) without transfers. The key insight: two reversals (stack_in to stack_out) restore FIFO order.",
    keyPoints: [
      'stack_in for enqueue, stack_out for dequeue',
      'Enqueue: push to stack_in, O(1)',
      'Dequeue: pop from stack_out, transfer if empty',
      'Transfer is O(n) but amortized O(1)',
      'Two stack reversals restore FIFO order',
    ],
  },
  {
    id: 'q3',
    question:
      'What is the pattern for "sliding window maximum" and why does it use a deque instead of a regular queue?',
    sampleAnswer:
      "Sliding window maximum finds the max element in each window of size k as it slides. We maintain a deque of indices in decreasing order of their values. For each element: (1) Remove indices outside window from front (dequeue from left), (2) Remove indices with smaller values from back (we pop from right because current element makes them useless), (3) Add current index to back, (4) Front of deque is the max for this window. We need deque not regular queue because we remove from BOTH ends: old indices from front (outside window), useless smaller indices from back (won't be max). This is O(n) because each element enters and leaves deque at most once.",
    keyPoints: [
      'Maintain deque of indices in decreasing value order',
      'Remove old indices from front (outside window)',
      'Remove smaller indices from back (useless)',
      'Need both-end operations: deque not queue',
      'O(n): each element processed once',
    ],
  },
];
