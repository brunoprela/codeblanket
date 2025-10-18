/**
 * Remove Linked List Elements
 * Problem ID: remove-linked-list-elements
 * Order: 10
 */

import { Problem } from '../../../types';

export const remove_elementsProblem: Problem = {
  id: 'remove-linked-list-elements',
  title: 'Remove Linked List Elements',
  difficulty: 'Easy',
  topic: 'Linked List',
  description: `Given the \`head\` of a linked list and an integer \`val\`, remove all the nodes of the linked list that has \`Node.val == val\`, and return the new head.`,
  examples: [
    {
      input: 'head = [1,2,6,3,4,5,6], val = 6',
      output: '[1,2,3,4,5]',
    },
    {
      input: 'head = [], val = 1',
      output: '[]',
    },
    {
      input: 'head = [7,7,7,7], val = 7',
      output: '[]',
    },
  ],
  constraints: [
    'The number of nodes in the list is in the range [0, 10^4]',
    '1 <= Node.val <= 50',
    '0 <= val <= 50',
  ],
  hints: ['Use a dummy node to handle edge cases', 'Keep a previous pointer'],
  starterCode: `from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def remove_elements(head: Optional[ListNode], val: int) -> Optional[ListNode]:
    """
    Remove all nodes with given value.
    
    Args:
        head: Head of linked list
        val: Value to remove
        
    Returns:
        New head after removal
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 2, 6, 3, 4, 5, 6], 6],
      expected: [1, 2, 3, 4, 5],
    },
    {
      input: [[], 1],
      expected: [],
    },
    {
      input: [[7, 7, 7, 7], 7],
      expected: [],
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/remove-linked-list-elements/',
  youtubeUrl: 'https://www.youtube.com/watch?v=JI71sxtHTng',
};
