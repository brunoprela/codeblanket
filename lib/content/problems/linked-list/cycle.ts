/**
 * Linked List Cycle
 * Problem ID: linked-list-cycle
 * Order: 2
 */

import { Problem } from '../../../types';

export const cycleProblem: Problem = {
  id: 'linked-list-cycle',
  title: 'Linked List Cycle',
  difficulty: 'Medium',
  description: `Given \`head\`, the head of a linked list, determine if the linked list has a **cycle** in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the \`next\` pointer. Internally, \`pos\` is used to denote the index of the node that tail's \`next\` pointer is connected to. **Note that \`pos\` is not passed as a parameter**.

Return \`true\` if there is a cycle in the linked list. Otherwise, return \`false\`.


**Approach:**
Use Floyd Cycle Detection Algorithm (tortoise and hare). Use two pointers: slow moves one step at a time, fast moves two steps. If there is a cycle, they will eventually meet. If fast reaches None, there is no cycle.`,
  examples: [
    {
      input: 'head = [3,2,0,-4], pos = 1',
      output: 'true',
      explanation:
        'There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).',
    },
    {
      input: 'head = [1,2], pos = 0',
      output: 'true',
      explanation:
        'There is a cycle in the linked list, where the tail connects to the 0th node.',
    },
    {
      input: 'head = [1], pos = -1',
      output: 'false',
      explanation: 'There is no cycle in the linked list.',
    },
  ],
  constraints: [
    'The number of the nodes in the list is in the range [0, 10^4]',
    '-10^5 <= Node.val <= 10^5',
    'pos is -1 or a valid index in the linked-list',
  ],
  hints: [
    'Use two pointers: slow and fast',
    'Move slow one step, fast two steps in each iteration',
    'If fast reaches None, there is no cycle',
    'If slow equals fast at any point, there is a cycle',
    "This is Floyd's Cycle Detection Algorithm",
  ],
  starterCode: `from typing import Optional

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

def has_cycle(head: Optional[ListNode]) -> bool:
    """
    Detect if linked list has a cycle.
    
    Args:
        head: Head of the linked list
        
    Returns:
        True if cycle exists, False otherwise
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[[3, 2, 0, -4], 1]],
      expected: true,
    },
    {
      input: [[[1, 2], 0]],
      expected: true,
    },
    {
      input: [[[1], -1]],
      expected: false,
    },
    {
      input: [[[], -1]],
      expected: false,
    },
  ],
  solution: `from typing import Optional

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


def has_cycle(head: Optional[ListNode]) -> bool:
    """
    Floyd's Cycle Detection (Tortoise and Hare).
    Time: O(N), Space: O(1)
    """
    if not head or not head.next:
        return False
    
    slow = head
    fast = head
    
    while fast and fast.next:
        slow = slow.next          # Move 1 step
        fast = fast.next.next     # Move 2 steps
        
        if slow == fast:          # They met - cycle exists
            return True
    
    return False  # fast reached end - no cycle


# Alternative: Using hash set (less optimal)
def has_cycle_set(head: Optional[ListNode]) -> bool:
    """
    Hash set approach.
    Time: O(N), Space: O(N)
    """
    visited = set()
    curr = head
    
    while curr:
        if curr in visited:
            return True
        visited.add(curr)
        curr = curr.next
    
    return False


# Bonus: Find where cycle starts
def detect_cycle_start(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Find the node where cycle begins.
    Time: O(N), Space: O(1)
    """
    if not head or not head.next:
        return None
    
    # Phase 1: Detect if cycle exists
    slow = fast = head
    has_cycle = False
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            has_cycle = True
            break
    
    if not has_cycle:
        return None
    
    # Phase 2: Find cycle start
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow  # This is where cycle starts`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',

  leetcodeUrl: 'https://leetcode.com/problems/linked-list-cycle/',
  youtubeUrl: 'https://www.youtube.com/watch?v=gBTe7lFR3vc',
  order: 2,
  topic: 'Linked List',
};
