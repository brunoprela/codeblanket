/**
 * Middle of the Linked List
 * Problem ID: middle-of-linked-list
 * Order: 4
 */

import { Problem } from '../../../types';

export const middle_of_linked_listProblem: Problem = {
  id: 'middle-of-linked-list',
  title: 'Middle of the Linked List',
  difficulty: 'Easy',
  topic: 'Linked List',
  order: 4,
  description: `Given the \`head\` of a singly linked list, return the middle node of the linked list.

If there are two middle nodes, return **the second middle** node.`,
  examples: [
    {
      input: 'head = [1,2,3,4,5]',
      output: '[3,4,5]',
      explanation: 'The middle node of the list is node 3.',
    },
    {
      input: 'head = [1,2,3,4,5,6]',
      output: '[4,5,6]',
      explanation:
        'Since the list has two middle nodes with values 3 and 4, we return the second one.',
    },
  ],
  constraints: [
    'The number of nodes in the list is in the range [1, 100]',
    '1 <= Node.val <= 100',
  ],
  hints: [
    'Use slow and fast pointers (tortoise and hare)',
    'Fast moves twice as fast as slow',
    'When fast reaches end, slow is at middle',
  ],
  starterCode: `from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def middle_node(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Find the middle node of linked list.
    
    Args:
        head: Head of the linked list
        
    Returns:
        Middle node (second middle if even length)
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 2, 3, 4, 5]],
      expected: [3, 4, 5],
    },
    {
      input: [[1, 2, 3, 4, 5, 6]],
      expected: [4, 5, 6],
    },
    {
      input: [[1]],
      expected: [1],
    },
  ],
  solution: `from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def middle_node(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Two-pointer approach (slow and fast).
    Time: O(n), Space: O(1)
    """
    slow = fast = head
    
    # Fast moves twice as fast as slow
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    # Slow is at middle
    return slow
`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/middle-of-the-linked-list/',
  youtubeUrl: 'https://www.youtube.com/watch?v=A2_ldqM4QcY',
};
