/**
 * Remove Nth Node From End of List
 * Problem ID: remove-nth-node-from-end
 * Order: 12
 */

import { Problem } from '../../../types';

export const remove_nth_node_from_endProblem: Problem = {
  id: 'remove-nth-node-from-end',
  title: 'Remove Nth Node From End of List',
  difficulty: 'Medium',
  topic: 'Linked List',
  description: `Given the \`head\` of a linked list, remove the \`n-th\` node from the end of the list and return its head.`,
  examples: [
    {
      input: 'head = [1,2,3,4,5], n = 2',
      output: '[1,2,3,5]',
    },
    {
      input: 'head = [1], n = 1',
      output: '[]',
    },
    {
      input: 'head = [1,2], n = 1',
      output: '[1]',
    },
  ],
  constraints: [
    'The number of nodes in the list is sz',
    '1 <= sz <= 30',
    '0 <= Node.val <= 100',
    '1 <= n <= sz',
  ],
  hints: [
    'Use two pointers with n gap between them',
    'Move both until first reaches end',
    'Second pointer will be at node before the one to delete',
  ],
  starterCode: `from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """
    Remove nth node from end of list.
    
    Args:
        head: Head of linked list
        n: Position from end (1-indexed)
        
    Returns:
        New head after removal
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 2, 3, 4, 5], 2],
      expected: [1, 2, 3, 5],
    },
    {
      input: [[1], 1],
      expected: [],
    },
    {
      input: [[1, 2], 1],
      expected: [1],
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl:
    'https://leetcode.com/problems/remove-nth-node-from-end-of-list/',
  youtubeUrl: 'https://www.youtube.com/watch?v=XVuQxVej6y8',
};
