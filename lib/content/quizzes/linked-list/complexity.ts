/**
 * Quiz questions for Complexity Analysis section
 */

export const complexityQuiz = [
  {
    id: 'q1',
    question:
      'Compare the complexity of inserting at head vs tail in a singly linked list. Why is there such a difference?',
    sampleAnswer:
      'Inserting at head is O(1) because we have direct access to the head pointer - create new node, point it to current head, update head to new node. Done. Inserting at tail is O(n) because we must traverse the entire list to find the last node, then append. Each step of traversal takes constant time but we need n steps. However, if we maintain a tail pointer, insertion at tail becomes O(1) too - just update tail.next and move tail pointer. The difference comes from whether we need to search for the insertion point or have direct access to it. This is why doubly linked lists with tail pointers are more versatile.',
    keyPoints: [
      'Head insert: O(1) - direct access',
      'Tail insert: O(n) - must traverse',
      'With tail pointer: O(1) for tail insert',
      'Difference: need to search vs direct access',
      'Doubly linked with tail pointer: O(1) both ends',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain why recursive linked list solutions use O(n) space. What exactly is stored on the call stack?',
    sampleAnswer:
      'Recursive solutions use O(n) space because each recursive call adds a stack frame to the call stack. For a list of n nodes, we make n recursive calls before starting to return. Each stack frame stores local variables like the current node pointer, return address, and any other local state. For example, recursive reverse: we recursively call on next node, storing current node on each frame, until we reach the end, then unwind the stack updating pointers. This is in contrast to iterative solutions which use O(1) space with just a few pointer variables. The space comes from the recursion depth matching list length, not from the data itself.',
    keyPoints: [
      'Each recursive call adds stack frame',
      'n nodes = n recursive calls',
      'Each frame: local variables, return address',
      'Stack depth matches list length',
      'Iterative: O(1) with pointer variables',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through why Floyd cycle detection is O(n) time and O(1) space. How do the pointers traverse the list?',
    sampleAnswer:
      'Floyd cycle detection uses two pointers: slow (one step) and fast (two steps). In worst case with no cycle, fast reaches the end after n/2 iterations, giving O(n) time. With a cycle, once both enter the cycle, fast catches slow in at most cycle length iterations. Total is still O(n) because we traverse at most n nodes plus some cycle iterations. Space is O(1) because we only need two pointer variables, regardless of list size. Compare to hash set approach: also O(n) time but O(n) space to store visited nodes. The brilliance of Floyd is achieving cycle detection with constant space by using the mathematical property that faster pointer will catch slower in a cycle.',
    keyPoints: [
      'Two pointers: O(1) space',
      'No cycle: fast reaches end in n/2 iterations',
      'With cycle: fast catches slow within cycle',
      'Total: O(n) time',
      'vs Hash set: O(n) space',
    ],
  },
];
