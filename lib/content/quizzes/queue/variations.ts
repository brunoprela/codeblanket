/**
 * Quiz questions for Queue Variations section
 */

export const variationsQuiz = [
  {
    id: 'q1',
    question:
      'Explain how a circular queue works and why it uses the modulo operator.',
    sampleAnswer:
      'A circular queue uses a fixed-size array where the rear wraps back to the front when it reaches the end. It uses modulo operator (index + 1) % capacity to wrap indices around. For example, in a queue of capacity 5, after index 4, the next index is (4 + 1) % 5 = 0. This reuses space from dequeued elements instead of shifting array elements or growing the array. Both front and rear pointers wrap around. This is efficient for fixed-size buffers where we want O(1) enqueue/dequeue without wasted space or expensive shifts.',
    keyPoints: [
      'Fixed-size array that wraps around',
      'Modulo operator: (index + 1) % capacity',
      'Reuses space from dequeued elements',
      'Both front and rear wrap around',
      'O(1) operations, no shifting or growing',
    ],
  },
  {
    id: 'q2',
    question:
      'What is the difference between a regular queue and a priority queue? When would you use each?',
    sampleAnswer:
      "Regular queue follows strict FIFO - first in is first out. Priority queue dequeues elements by priority, not insertion order. Regular queue uses deque with O(1) enqueue/dequeue. Priority queue uses a heap with O(log n) enqueue/dequeue. Use regular queue for BFS, task processing where order matters (like print queue). Use priority queue when some elements are more important: Dijkstra's shortest path (process closest node first), CPU scheduling (high priority tasks first), A* search, event simulation with timestamps. Key difference: FIFO vs priority-based.",
    keyPoints: [
      'Regular: strict FIFO order',
      'Priority: dequeue by priority, not order',
      'Regular: O(1) operations with deque',
      'Priority: O(log n) with heap',
      'Use priority when importance varies',
    ],
  },
  {
    id: 'q3',
    question:
      'Why is a deque better than a regular queue for sliding window problems?',
    sampleAnswer:
      'Sliding window problems often need to add/remove from both ends efficiently. Deque provides O(1) operations at both ends: append (add right), appendleft (add left), pop (remove right), popleft (remove left). Regular queue only efficiently removes from front. For example, in "sliding window maximum," we maintain indices in decreasing order of values. When window slides: remove expired indices from front (popleft), remove smaller values from rear before adding new (pop). Both operations are O(1) with deque. Using a regular list would be O(n) for removing from front. Deque is implemented as a doubly-linked list, enabling efficient operations on both ends.',
    keyPoints: [
      'Deque: O(1) add/remove from both ends',
      'Regular queue: only efficient at one end',
      'Sliding window: remove old (front), add new (rear)',
      'Maintain order/constraints by modifying both ends',
      'Deque is doubly-linked list internally',
    ],
  },
];
