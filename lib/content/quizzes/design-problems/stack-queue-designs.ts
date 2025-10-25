/**
 * Quiz questions for Stack & Queue Designs section
 */

export const stackqueuedesignsQuiz = [
  {
    id: 'q1',
    question:
      'Explain how the Min Stack maintains O(1) getMin() while supporting push and pop.',
    sampleAnswer:
      'Min Stack maintains O(1) getMin by storing the minimum value at each level in a parallel min_stack. When we push (val), we compare val with current min and push the smaller one to min_stack. When we pop(), we pop from both stacks to keep them synchronized. The key insight: at any point, min_stack.top() tells us "what is the minimum among all elements currently in the stack?" When we pop an element, the min_stack also pops, revealing the min of the remaining elements. For example, after push(3), push(1), push(5): main=[3,1,5], min=[3,1,1]. getMin() returns min[-1]=1 in O(1). After pop(): main=[3,1], min=[3,1], getMin() still works correctly returning 1.',
    keyPoints: [
      'Parallel min_stack tracks min at each level',
      'Push: store min (new_val, current_min)',
      'Pop: remove from both stacks (stay synchronized)',
      'min_stack.top() = min of current elements',
      'Always O(1) - just array access',
    ],
  },
  {
    id: 'q2',
    question:
      'Why is Queue using Stacks amortized O(1) for dequeue? Explain the amortization.',
    sampleAnswer:
      "Individual dequeue can be O(N) when transferring all elements from stack_in to stack_out, but amortized cost is O(1) because each element is transferred at most once. Consider N enqueues followed by N dequeues: First dequeue transfers N elements (O(N)), but subsequent N-1 dequeues just pop from stack_out (O(1) each). Total: O(N) + O(N-1) = O(2N-1) for N dequeues = O(1) average. Key point: once an element moves to stack_out, it never moves back. You can't count O(N) for every dequeue - elements aren't transferred repeatedly. Amortized analysis spreads the expensive operation over all operations.",
    keyPoints: [
      'Individual dequeue can be O(N) (transfer)',
      'But each element transferred at most once',
      'N dequeues = O(N) total work = O(1) average',
      'Elements never move back to stack_in',
      'Spread expensive operation over all ops',
    ],
  },
  {
    id: 'q3',
    question:
      'Compare the two approaches for Min Stack: two stacks vs. stack with tuples. Which is better?',
    sampleAnswer:
      "Two-stack approach: Pros: cleaner separation, easier to understand, each stack stores single values. Cons: manage two data structures, remember to sync on pop. Stack-with-tuples approach: Pros: single data structure, impossible to desync, cleaner code. Cons: every element stores tuple (more memory per element), tuple unpacking overhead. In practice, I prefer two stacks for interviews because it's clearer to explain and reason about. For production, stack-with-tuples might be cleaner due to single data structure. Neither has better time complexity (both O(1)). Memory is similar (two stacks: 2N ints, tuples: N tuples). Choice is about code clarity and maintainability.",
    keyPoints: [
      'Two stacks: cleaner separation, easier to explain',
      'Tuples: single structure, cannot desync',
      'Both O(1) time complexity',
      'Memory similar (2N ints vs N tuples)',
      'Choice based on clarity, not performance',
    ],
  },
];
