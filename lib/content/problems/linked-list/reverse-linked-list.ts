/**
 * Reverse Linked List
 * Problem ID: reverse-linked-list
 * Order: 1
 */

import { Problem } from '../../../types';

export const reverse_linked_listProblem: Problem = {
  id: 'reverse-linked-list',
  title: 'Reverse Linked List',
  difficulty: 'Easy',
  description: `Given the \`head\` of a singly linked list, reverse the list, and return **the reversed list**.


**Approach:**
Iterate through the list, reversing the \`next\` pointer of each node. Use three pointers: \`prev\`, \`curr\`, and \`next_temp\` to safely reverse links without losing references.

**Alternative:** Can also be solved recursively, but uses O(N) stack space.`,
  examples: [
    {
      input: 'head = [1,2,3,4,5]',
      output: '[5,4,3,2,1]',
      explanation: 'The list is reversed.',
    },
    {
      input: 'head = [1,2]',
      output: '[2,1]',
    },
    {
      input: 'head = []',
      output: '[]',
    },
  ],
  constraints: [
    'The number of nodes in the list is the range [0, 5000]',
    '-5000 <= Node.val <= 5000',
  ],
  hints: [
    'Use three pointers: prev, curr, and next_temp',
    'Save the next node before changing curr.next',
    'Reverse the pointer: curr.next = prev',
    'Move all pointers forward: prev = curr, curr = next_temp',
    'Return prev at the end (new head)',
  ],
  starterCode: `from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Reverse a singly linked list.
    
    Args:
        head: Head of the linked list
        
    Returns:
        Head of the reversed list
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[[1, 2, 3, 4, 5]]],
      expected: [5, 4, 3, 2, 1],
    },
    {
      input: [[[1, 2]]],
      expected: [2, 1],
    },
    {
      input: [[[]]],
      expected: [],
    },
    {
      input: [[[1]]],
      expected: [1],
    },
  ],
  solution: `from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Iterative solution.
    Time: O(N), Space: O(1)
    """
    prev = None
    curr = head
    
    while curr:
        next_temp = curr.next  # Save next
        curr.next = prev       # Reverse link
        prev = curr            # Move prev forward
        curr = next_temp       # Move curr forward
    
    return prev  # New head


# Alternative: Recursive solution
def reverse_list_recursive(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Recursive solution.
    Time: O(N), Space: O(N) for call stack
    """
    # Base case
    if not head or not head.next:
        return head
    
    # Recursively reverse the rest
    new_head = reverse_list_recursive(head.next)
    
    # Reverse the link
    head.next.next = head
    head.next = None
    
    return new_head`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1) iterative, O(n) recursive',

  leetcodeUrl: 'https://leetcode.com/problems/reverse-linked-list/',
  youtubeUrl: 'https://www.youtube.com/watch?v=G0_I-ZF0S38',
  order: 1,
  topic: 'Linked List',
};
