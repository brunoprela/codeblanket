/**
 * Reorder List
 * Problem ID: reorder-list
 * Order: 13
 */

import { Problem } from '../../../types';

export const reorder_listProblem: Problem = {
  id: 'reorder-list',
  title: 'Reorder List',
  difficulty: 'Medium',
  topic: 'Linked List',
  description: `You are given the head of a singly linked-list. The list can be represented as:

L0 → L1 → … → Ln - 1 → Ln

Reorder the list to be on the following form:

L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …

You may not modify the values in the list nodes. Only nodes themselves may be changed.`,
  examples: [
    {
      input: 'head = [1,2,3,4]',
      output: '[1,4,2,3]',
    },
    {
      input: 'head = [1,2,3,4,5]',
      output: '[1,5,2,4,3]',
    },
  ],
  constraints: [
    'The number of nodes in the list is in the range [1, 5 * 10^4]',
    '1 <= Node.val <= 1000',
  ],
  hints: [
    'Find middle of list',
    'Reverse second half',
    'Merge two halves alternately',
  ],
  starterCode: `from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reorder_list(head: Optional[ListNode]) -> None:
    """
    Reorder list in-place.
    
    Args:
        head: Head of linked list
        
    Returns:
        None, modifies list in-place
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 2, 3, 4]],
      expected: [1, 4, 2, 3],
    },
    {
      input: [[1, 2, 3, 4, 5]],
      expected: [1, 5, 2, 4, 3],
    },
    {
      input: [[1]],
      expected: [1],
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/reorder-list/',
  youtubeUrl: 'https://www.youtube.com/watch?v=S5bfdUTrKLM',
};
