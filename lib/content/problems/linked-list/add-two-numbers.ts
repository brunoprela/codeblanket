/**
 * Add Two Numbers
 * Problem ID: add-two-numbers
 * Order: 14
 */

import { Problem } from '../../../types';

export const add_two_numbersProblem: Problem = {
  id: 'add-two-numbers',
  title: 'Add Two Numbers',
  difficulty: 'Medium',
  topic: 'Linked List',
  description: `You are given two **non-empty** linked lists representing two non-negative integers. The digits are stored in **reverse order**, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.`,
  examples: [
    {
      input: 'l1 = [2,4,3], l2 = [5,6,4]',
      output: '[7,0,8]',
      explanation: '342 + 465 = 807.',
    },
    {
      input: 'l1 = [0], l2 = [0]',
      output: '[0]',
    },
    {
      input: 'l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]',
      output: '[8,9,9,9,0,0,0,1]',
    },
  ],
  constraints: [
    'The number of nodes in each linked list is in the range [1, 100]',
    '0 <= Node.val <= 9',
    'It is guaranteed that the list represents a number that does not have leading zeros',
  ],
  hints: [
    'Track carry while adding digits',
    'Handle different lengths',
    'Do not forget carry at the end',
  ],
  starterCode: `from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def add_two_numbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    """
    Add two numbers represented by linked lists.
    
    Args:
        l1: First number (reversed)
        l2: Second number (reversed)
        
    Returns:
        Sum as linked list (reversed)
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [2, 4, 3],
        [5, 6, 4],
      ],
      expected: [7, 0, 8],
    },
    {
      input: [[0], [0]],
      expected: [0],
    },
    {
      input: [
        [9, 9, 9, 9, 9, 9, 9],
        [9, 9, 9, 9],
      ],
      expected: [8, 9, 9, 9, 0, 0, 0, 1],
    },
  ],
  timeComplexity: 'O(max(m, n))',
  spaceComplexity: 'O(max(m, n))',
  leetcodeUrl: 'https://leetcode.com/problems/add-two-numbers/',
  youtubeUrl: 'https://www.youtube.com/watch?v=wgFPrzTjm7s',
};
