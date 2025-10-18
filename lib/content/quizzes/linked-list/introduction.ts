/**
 * Quiz questions for Introduction to Linked Lists section
 */

export const introductionQuiz = [
  {
    id: 'q1',
    question:
      'Explain the fundamental difference between linked lists and arrays. What trade-offs do you make when choosing one over the other?',
    sampleAnswer:
      'The fundamental difference is memory layout and access pattern. Arrays store elements contiguously in memory with O(1) random access by index - I can jump to any element instantly. Linked lists store elements as nodes scattered in memory, connected by pointers, with O(n) access - I must traverse from head to reach an element. Arrays have better cache locality, making them faster for sequential access. Linked lists excel at insertions and deletions, especially at the beginning: O(1) vs O(n) for arrays which must shift elements. The trade-off: arrays for fast access and iteration, linked lists for dynamic size and frequent insertions/deletions. Memory overhead: arrays have minimal overhead, linked lists need extra pointer storage per node.',
    keyPoints: [
      'Arrays: contiguous memory, O(1) random access',
      'Linked lists: scattered memory, O(n) access',
      'Arrays: better cache locality',
      'Linked lists: O(1) insert/delete at beginning',
      'Trade-off: access speed vs insertion flexibility',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through the mechanics of inserting a node in the middle of a linked list. Why is this O(1) once you have the reference?',
    sampleAnswer:
      'To insert a new node after a given node, I create the new node, set its next pointer to point to the current node next, then update the current node next to point to the new node. This is just two pointer updates, so O(1) time once I have the reference. The key is "once you have the reference" - finding the insertion point takes O(n) traversal. But if I already have a pointer to the node, insertion is constant time because I am only changing pointers, not shifting elements like in arrays. For example, to insert B between A and C: B.next = A.next (B points to C), then A.next = B (A points to B). No other nodes are affected.',
    keyPoints: [
      'Create new node',
      'New node points to current node next',
      'Update current node to point to new node',
      'Two pointer updates: O(1)',
      'Finding position: O(n), insertion itself: O(1)',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe the concept of a dummy node. When would you use it and why does it simplify code?',
    sampleAnswer:
      'A dummy node is a placeholder node placed before the actual head of the list. Instead of tracking the real head, I work with dummy.next as the head. This simplifies edge cases because now the head is no longer special - I can insert or delete at the head using the same logic as any other position. Without a dummy, I need special case handling when the head changes. For example, deleting the first node requires updating the head pointer separately. With a dummy, deleting the first node is just dummy.next = dummy.next.next, same as any deletion. At the end, return dummy.next as the new head. This is a common interview trick to avoid messy conditional logic.',
    keyPoints: [
      'Placeholder node before real head',
      'Work with dummy.next as head',
      'Eliminates special case for head operations',
      'Same logic for all positions',
      'Return dummy.next at end',
    ],
  },
];
