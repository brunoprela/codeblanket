/**
 * Quiz questions for Advanced Techniques section
 */

export const advancedQuiz = [
  {
    id: 'q1',
    question:
      'Explain how to find the cycle start after detecting a cycle. Why does the mathematical relationship work?',
    sampleAnswer:
      'After fast and slow meet in a cycle, reset one pointer to head and keep the other at meeting point. Move both one step at a time - they will meet at the cycle start. The math: let distance from head to cycle start be x, distance from start to meeting point be y, and remaining cycle be z. When they meet, slow traveled x+y, fast traveled x+y+z+y (went around once more). Since fast is twice as fast, 2(x+y) = x+y+z+y, simplifying to x = z. So distance from head to start equals distance from meeting point to start. This elegant property lets us find the start with O(1) space. Beautiful application of algebra to pointer manipulation.',
    keyPoints: [
      'After meeting: reset one to head',
      'Both move one step until meeting',
      'They meet at cycle start',
      'Math: x = z (head to start = meeting to start)',
      'O(1) space solution',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through the approach to reverse nodes in k-groups. What makes this problem more complex than simple reversal?',
    sampleAnswer:
      'Reversing in k-groups requires tracking multiple boundaries. For each group of k nodes, I reverse that segment, then connect it back to the previous group and forward to the next group. The complexity comes from managing these connections: need to save the node before the group, reverse the k nodes, connect previous group tail to reversed group head, and connect reversed group tail to next group. If fewer than k nodes remain, leave them as-is. I typically use a helper function to reverse k nodes and return new head and tail, making the main logic cleaner. The challenge is correctly updating all pointer connections without losing references.',
    keyPoints: [
      'Reverse each k-node segment',
      'Track boundaries: before, after each group',
      'Connect reversed segments back together',
      'Handle remainder less than k',
      'Helper function for k-node reversal',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe the technique for sorting a linked list. Why is merge sort preferred over quick sort for linked lists?',
    sampleAnswer:
      'Merge sort is ideal for linked lists because it does not require random access. The algorithm: find middle with fast-slow pointers, recursively sort left and right halves, merge sorted halves. This gives O(n log n) time and O(log n) space for recursion. Quick sort needs efficient partitioning which requires random access - linked lists lack this. Partitioning a linked list is O(n) but awkward with many pointer updates. Merge sort natural operations (split, merge) work elegantly with linked list structure. The merge step is especially clean with linked lists - just pointer manipulation, no array copying. This is why merge sort is the go-to for linked list sorting.',
    keyPoints: [
      'Merge sort: O(n log n) time, O(log n) space',
      'Works without random access',
      'Find middle, recursively sort, merge',
      'Quick sort needs random access for efficient partition',
      'Merge step elegant with pointers',
    ],
  },
];
