/**
 * Quiz questions for Time & Space Complexity section
 */

export const complexityQuiz = [
  {
    id: 'q1',
    question:
      'Why is list.pop(0) O(n) but deque.popleft() O(1)? Explain the underlying data structure difference.',
    sampleAnswer:
      'Lists in Python are implemented as dynamic arrays (contiguous memory). When you pop(0) from the front, all remaining elements must shift left by one position to fill the gap - this is O(n) copying. Deque is implemented as a doubly-linked list of blocks (not a single array). Each block contains multiple elements. Removing from the left just adjusts the head pointer to the next block or next element within the block - no shifting needed. This makes popleft() O(1). Similarly, appendleft() is O(1) because we just add a new block or element at the head. Lists optimize for index access O(1), deques optimize for both-end operations O(1).',
    keyPoints: [
      'List: dynamic array, contiguous memory',
      'list.pop(0): shift all elements, O(n)',
      'Deque: doubly-linked list of blocks',
      'deque.popleft(): adjust head pointer, O(1)',
      'Lists for indexing, deques for both-end operations',
    ],
  },
  {
    id: 'q2',
    question:
      'What is amortized O(1) and why does the two-stack queue implementation achieve it?',
    sampleAnswer:
      'Amortized O(1) means that while some operations are expensive, the average cost per operation over many operations is O(1). In two-stack queue, most dequeues are O(1) (just pop from stack_out). Occasionally, when stack_out is empty, we must transfer all n elements from stack_in to stack_out - this single operation is O(n). But here\'s the key: each element is enqueued once, transferred at most once, and dequeued once. So for n operations total, we do at most 3n single-element operations, which averages to O(1) per operation. The expensive transfers are rare and their cost is "spread out" (amortized) over many cheap operations.',
    keyPoints: [
      'Most operations cheap, occasional expensive one',
      'Average cost over many operations is O(1)',
      'Each element: enqueued once, transferred once, dequeued once',
      'Total 3n operations for n elements = O(1) average',
      'Expensive transfers are rare and cost is spread out',
    ],
  },
  {
    id: 'q3',
    question:
      'Compare the space complexity of BFS vs DFS. Which uses more memory and why?',
    sampleAnswer:
      'For a tree of depth d with branching factor b, BFS uses O(b^d) space (width of last level) because the queue holds all nodes at the current level. DFS uses O(d) space for the recursion stack or explicit stack (height of tree). BFS is wider, DFS is deeper. For a binary tree of depth 10, BFS can have 2^10 = 1024 nodes in queue at the deepest level. DFS only needs 10 stack frames. BFS uses exponentially more space as depth increases for trees with high branching factor. However, for graphs with cycles, DFS also needs O(V) visited set. In general: BFS = O(width), DFS = O(depth). Use DFS if memory is tight and tree is wide.',
    keyPoints: [
      'BFS: O(b^d) space, holds entire level in queue',
      'DFS: O(d) space, holds path in stack',
      'BFS wider, DFS deeper',
      'BFS can use exponentially more memory',
      'Choose DFS if memory tight and tree is wide',
    ],
  },
];
