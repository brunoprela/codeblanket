/**
 * Quiz questions for Introduction to Heaps section
 */

export const introductionQuiz = [
  {
    id: 'q1',
    question:
      'Explain what a heap is and what makes it different from other tree structures. Why is it always complete?',
    sampleAnswer:
      'A heap is a complete binary tree that satisfies the heap property: in a max heap, every parent is greater than or equal to its children; in a min heap, every parent is less than or equal to its children. The key difference from BST is that heap only maintains parent-child ordering, not left-right ordering - left child can be greater than right child. A heap must be complete (all levels filled except possibly last, which fills left to right) because this enables array representation: parent at i, children at 2i+1 and 2i+2. Completeness ensures balanced height of log n and efficient operations. Unlike BST which can become skewed, heaps always maintain log height.',
    keyPoints: [
      'Complete binary tree with heap property',
      'Max heap: parent ≥ children',
      'Min heap: parent ≤ children',
      'Complete enables array representation',
      'Always balanced: height log n',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe how heaps enable O(1) access to min/max element. Why is this useful for priority queues?',
    sampleAnswer:
      'In a heap, the root is always the min (min heap) or max (max heap) element due to the heap property - it is at index 0 in the array. No searching needed, just return heap[0] in O(1). This makes heaps perfect for priority queues where you repeatedly need the highest priority (max heap) or lowest priority (min heap) item. For example, in task scheduling, the highest priority task is always at the root. When you remove it with O(log n) pop, the next highest automatically becomes the new root after heapify. This is much better than unsorted array (O(n) find min) or sorted array (O(n) insert). Heaps balance find-min and insert both at O(log n).',
    keyPoints: [
      'Root always contains min/max',
      'O(1) access at index 0',
      'Perfect for priority queues',
      'Remove max/min: O(log n)',
      'Better than unsorted or sorted arrays',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through why heaps are commonly implemented as arrays rather than explicit node structures.',
    sampleAnswer:
      'Heaps use arrays because the complete binary tree structure maps perfectly to array indices. Parent at i, left child at 2i+1, right child at 2i+2. This arithmetic relationship means we can navigate parent-child without pointers, saving memory and improving cache locality. No null children to track since the tree is complete. Array implementation is more space-efficient (no pointer overhead) and faster (better cache performance from contiguous memory). For example, to bubble up from index 5, parent is at (5-1)//2 = 2, no pointer traversal needed. The only downside is fixed size, but Python heapq uses dynamic arrays. This is why standard libraries use array-based heaps.',
    keyPoints: [
      'Complete tree maps to array indices perfectly',
      'Parent at i, children at 2i+1, 2i+2',
      'No pointers needed: arithmetic navigation',
      'Space-efficient, better cache locality',
      'Standard library implementations use arrays',
    ],
  },
];
