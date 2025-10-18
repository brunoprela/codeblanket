/**
 * Quiz questions for Queue Operations & Implementation section
 */

export const operationsQuiz = [
  {
    id: 'q1',
    question:
      'Why is using a regular Python list for a queue inefficient? What specific operation causes the problem?',
    sampleAnswer:
      'Using a list for a queue is inefficient because dequeue (removing from front) requires pop(0), which is O(n). When we remove the first element, Python must shift all remaining n-1 elements one position to the left to fill the gap. So if the queue has 1000 elements, removing one requires 999 shift operations. This makes dequeue O(n) instead of O(1). In contrast, enqueue using append() is O(1) because it adds to the end. For a proper queue, use collections.deque which has O(1) operations for both ends.',
    keyPoints: [
      'pop(0) is O(n) - shifts all elements',
      'Must shift n-1 elements after removal',
      'Makes queue operations slow',
      'deque has O(1) for both ends',
      'List good for stack, bad for queue',
    ],
  },
  {
    id: 'q2',
    question:
      'How do you implement a queue using two stacks? Explain the amortized O(1) complexity.',
    sampleAnswer:
      'Use two stacks: stack_in for enqueue, stack_out for dequeue. Enqueue: push to stack_in (O(1)). Dequeue: if stack_out is empty, transfer all elements from stack_in to stack_out (reversing order), then pop from stack_out. This transfer happens rarely. Amortized analysis: each element is pushed once to stack_in and moved once to stack_out over its lifetime, giving O(1) amortized per operation. Example: enqueue(1,2,3) → stack_in=[3,2,1]. Dequeue → transfer to stack_out=[1,2,3], pop 1.',
    keyPoints: [
      'Two stacks: stack_in and stack_out',
      'Enqueue pushes to stack_in',
      'Dequeue pops from stack_out',
      'Transfer from stack_in when stack_out empty',
      'Amortized O(1) per operation',
      'Each element moved at most twice total',
    ],
  },
  {
    id: 'q3',
    question: 'What is a circular queue and when would you use it?',
    sampleAnswer:
      'A circular queue uses a fixed-size array with two pointers (front and rear) that wrap around to the beginning when reaching the end. It prevents wasted space from regular array queue where front pointer moves right, leaving unused space. Use cases: 1) Fixed buffer size known (streaming data, print spooler), 2) Prevent memory fragmentation, 3) Efficient ring buffer implementation. Implementation: use modulo operator for wrapping: rear = (rear + 1) % capacity. Must track size to distinguish empty (size==0) from full (size==capacity).',
    keyPoints: [
      'Fixed-size array with wraparound',
      'Two pointers: front and rear',
      'Use modulo for wraparound: (index + 1) % capacity',
      'Prevents wasted space from linear array queue',
      'Ideal for fixed buffer size',
      'Track size to distinguish empty vs full',
    ],
  },
];
