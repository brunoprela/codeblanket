/**
 * Quiz questions for Collections Module - Advanced Data Structures section
 */

export const collectionsmoduleQuiz = [
  {
    id: 'q1',
    question: 'When would you use Counter vs defaultdict(int) for counting?',
    sampleAnswer:
      "Use Counter when you need counting-specific features like most_common(), arithmetic operations (+, -, &, |), or when you want to emphasize that you're counting. Counter also returns 0 for missing keys instead of raising KeyError. Use defaultdict(int) when counting is part of a larger operation and you need the auto-initialization behavior for general integer operations. Counter is more expressive for pure counting tasks: Counter([1,2,1]).most_common() is clearer than manually sorting defaultdict items.",
    keyPoints: [
      'Counter: pure counting, has most_common()',
      'Counter: math operations between counters',
      'Counter: more explicit intent',
      'defaultdict(int): part of larger logic',
      'Both avoid KeyError for missing keys',
    ],
  },
  {
    id: 'q2',
    question: 'Why is deque better than list for implementing a queue?',
    sampleAnswer:
      'deque.popleft() is O(1) while list.pop(0) is O(n). When removing from the front of a list, Python must shift all remaining elements left, taking O(n) time. With deque, both ends are optimized for O(1) operations using a doubly-linked list of blocks internally. For a queue with 10,000 operations, list would take ~50 million operations total (O(n²)) while deque takes just 10,000 (O(n)). This makes deque essential for BFS, sliding windows, and any FIFO data structure.',
    keyPoints: [
      'deque.popleft(): O(1)',
      'list.pop(0): O(n) - shifts elements',
      'Huge performance difference for queues',
      'Essential for BFS and sliding windows',
      'deque optimized for both ends',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain the practical differences between OrderedDict and regular dict in Python 3.7+.',
    sampleAnswer:
      'In Python 3.7+, regular dicts maintain insertion order, making OrderedDict less critical. However, OrderedDict still has unique features: (1) move_to_end(key) to reorder items, (2) popitem(last=False/True) to remove from specific end, (3) Explicit ordering semantics - code intent is clearer, (4) Equality checks consider order: OrderedDict(a=1, b=2) != OrderedDict(b=2, a=1), but regular dicts with same items are equal regardless of order. Use OrderedDict when you need these operations or want to explicitly signal that order matters for correctness (e.g., LRU cache implementation).',
    keyPoints: [
      'Both maintain insertion order in Python 3.7+',
      'OrderedDict has move_to_end() and directional popitem()',
      'OrderedDict equality considers order',
      'Regular dict equality ignores order',
      'Use OrderedDict when order operations are needed',
    ],
  },
];
