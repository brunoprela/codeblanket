/**
 * Convert Binary Number in Linked List to Integer
 * Problem ID: convert-binary-linked-list-to-integer
 * Order: 11
 */

import { Problem } from '../../../types';

export const convert_binary_to_integerProblem: Problem = {
  id: 'convert-binary-linked-list-to-integer',
  title: 'Convert Binary Number in Linked List to Integer',
  difficulty: 'Easy',
  topic: 'Linked List',
  description: `Given \`head\` which is a reference node to a singly-linked list. The value of each node in the linked list is either \`0\` or \`1\`. The linked list holds the binary representation of a number.

Return the decimal value of the number in the linked list.

The **most significant bit** is at the head of the linked list.`,
  examples: [
    {
      input: 'head = [1,0,1]',
      output: '5',
      explanation: '(101) in base 2 = (5) in base 10',
    },
    {
      input: 'head = [0]',
      output: '0',
    },
  ],
  constraints: [
    'The Linked List is not empty',
    'Number of nodes will not exceed 30',
    'Each node value is either 0 or 1',
  ],
  hints: [
    'Traverse the list and build the number',
    'For each node, result = result * 2 + node.val',
  ],
  starterCode: `from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def get_decimal_value(head: Optional[ListNode]) -> int:
    """
    Convert binary linked list to decimal integer.
    
    Args:
        head: Head of binary linked list
        
    Returns:
        Decimal value
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 0, 1]],
      expected: 5,
    },
    {
      input: [[0]],
      expected: 0,
    },
    {
      input: [[1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]],
      expected: 18880,
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl:
    'https://leetcode.com/problems/convert-binary-number-in-a-linked-list-to-integer/',
  youtubeUrl: 'https://www.youtube.com/watch?v=zhhlTLfP4WU',
};
