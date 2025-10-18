/**
 * Quiz questions for Introduction to Queues section
 */

export const introductionQuiz = [
  {
    id: 'q1',
    question:
      "Explain the FIFO principle and how it differs from a Stack's LIFO principle.",
    sampleAnswer:
      'FIFO (First-In-First-Out) means the first element added to the queue is the first one removed, like a line at a store. Stack uses LIFO (Last-In-First-Out) where the most recently added element is removed first, like a stack of plates. In a queue, elements are added at the rear and removed from the front. In a stack, elements are added and removed from the same end (top). Queue preserves order of arrival, while stack reverses it. Example: Queue [1,2,3] removes 1 first; Stack [1,2,3] removes 3 first.',
    keyPoints: [
      'Queue: First-In-First-Out (FIFO)',
      'Stack: Last-In-First-Out (LIFO)',
      'Queue: add rear, remove front',
      'Stack: add/remove from top',
      'Queue preserves order, stack reverses',
    ],
  },
  {
    id: 'q2',
    question:
      'Give three real-world examples where a queue is the natural data structure choice.',
    sampleAnswer:
      '(1) Printer queue - print jobs are processed in order they were submitted, ensuring fairness. (2) Customer service call center - customers are helped in order they called, first caller gets helped first. (3) BFS algorithm in graphs - explores nodes level by level, processing neighbors in order they were discovered. Other examples: process scheduling in OS, breadth-first tree traversal, keyboard buffer, network packet routing.',
    keyPoints: [
      'Any "first-come, first-served" scenario',
      'BFS algorithm (level-order traversal)',
      'Task scheduling and job processing',
      'Buffering and data streaming',
      'Order matters and fairness is important',
    ],
  },
  {
    id: 'q3',
    question:
      'What are the performance characteristics of queue operations with deque vs list?',
    sampleAnswer:
      'With deque: enqueue (append) is O(1), dequeue (popleft) is O(1). With list: enqueue (append) is O(1), but dequeue (pop(0)) is O(n) because all remaining elements must be shifted. For queue operations, deque is significantly better because both operations are O(1). List as queue causes O(n) dequeue, making it inefficient for large queues. Always use collections.deque for queue implementation in Python.',
    keyPoints: [
      'deque: O(1) for both enqueue and dequeue',
      'list: O(1) enqueue, O(n) dequeue (pop(0))',
      'list.pop(0) shifts all elements',
      'deque is optimized for both ends',
      'Use deque for efficient queues',
    ],
  },
];
