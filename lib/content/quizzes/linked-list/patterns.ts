/**
 * Quiz questions for Essential Linked List Patterns section
 */

export const patternsQuiz = [
  {
    id: 'q1',
    question:
      'Explain the fast and slow pointer technique. Why does the slow pointer end up at the middle when fast reaches the end?',
    sampleAnswer:
      'Fast and slow pointer uses two pointers moving at different speeds: slow moves one step per iteration, fast moves two steps. When fast reaches the end (or null), slow is at the middle. This works because fast travels twice the distance of slow. If the list has n nodes, when fast has moved n steps (reached end), slow has moved n/2 steps (at middle). For example, in list 1→2→3→4→5, when fast reaches 5, slow is at 3. This is elegant because we find the middle in one pass without knowing the length beforehand. Used for finding middle node, detecting cycles, or problems requiring simultaneous traversal at different speeds.',
    keyPoints: [
      'Slow moves one step, fast moves two steps',
      'Fast travels twice the distance of slow',
      'When fast at end, slow at middle',
      'One pass without knowing length',
      'Used for: middle, cycle detection, offset problems',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through cycle detection with Floyd algorithm. How does meeting of two pointers prove a cycle exists?',
    sampleAnswer:
      'Floyd cycle detection uses fast and slow pointers. If there is a cycle, fast will eventually catch up to slow inside the cycle because fast gains one position per iteration. Think of it like a circular track: the faster runner will lap the slower runner. If no cycle, fast reaches null. Once they meet, we know a cycle exists. To find the cycle start, reset one pointer to head and move both one step at a time - they meet at the cycle entrance. This works due to mathematical properties of the distances. The brilliance is using O(1) space instead of O(n) hash set to track visited nodes. Time is O(n) as each pointer traverses at most n nodes.',
    keyPoints: [
      'Fast and slow pointers in cycle',
      'Fast eventually catches slow (laps on circular track)',
      'Meeting proves cycle exists',
      'To find start: reset one to head, both move one step',
      'O(1) space vs O(n) hash set',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe the two-pointer technique for removing the nth node from end. Why move first pointer n steps ahead?',
    sampleAnswer:
      'To remove nth node from end, I use two pointers with n-node gap between them. First, move the first pointer n steps ahead. Then move both pointers together until first reaches the end. Now second pointer is n nodes from the end - exactly at the node before the one to delete. This works because maintaining constant gap of n nodes means when first reaches end (0 from end), second is n from end. I then do second.next = second.next.next to remove the target. Using a dummy node handles the edge case of removing the first node. This is one-pass solution, very elegant compared to two-pass (find length, then remove).',
    keyPoints: [
      'Two pointers with n-node gap',
      'Move first n steps ahead',
      'Move both until first at end',
      'Second now at node before target',
      'One pass vs two-pass solution',
    ],
  },
];
