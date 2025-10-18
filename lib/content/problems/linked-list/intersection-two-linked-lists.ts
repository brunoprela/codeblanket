/**
 * Intersection of Two Linked Lists
 * Problem ID: intersection-two-linked-lists
 * Order: 9
 */

import { Problem } from '../../../types';

export const intersection_two_linked_listsProblem: Problem = {
  id: 'intersection-two-linked-lists',
  title: 'Intersection of Two Linked Lists',
  difficulty: 'Easy',
  topic: 'Linked List',
  description: `Given the heads of two singly linked-lists \`headA\` and \`headB\`, return the node at which the two lists intersect. If the two linked lists have no intersection at all, return \`null\`.

It is guaranteed that there are no cycles anywhere in the entire linked structure.

**Note** that the linked lists must retain their original structure after the function returns.`,
  examples: [
    {
      input:
        'intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3',
      output: 'Intersected at node with value 8',
    },
    {
      input: 'intersectVal = 0, listA = [2,6,4], listB = [1,5]',
      output: 'No intersection',
    },
  ],
  constraints: [
    'The number of nodes of listA is in the m',
    'The number of nodes of listB is in the n',
    '1 <= m, n <= 3 * 10^4',
    '1 <= Node.val <= 10^5',
  ],
  hints: [
    'Use two pointers',
    'When a pointer reaches end, redirect it to the other head',
    'They will meet at intersection or both be null',
  ],
  starterCode: `from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def get_intersection_node(headA: Optional[ListNode], headB: Optional[ListNode]) -> Optional[ListNode]:
    """
    Find intersection node of two linked lists.
    
    Args:
        headA: Head of first list
        headB: Head of second list
        
    Returns:
        Intersection node or None
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[4, 1, 8, 4, 5], [5, 6, 1, 8, 4, 5], 2, 3],
      expected: 8,
    },
    {
      input: [[2, 6, 4], [1, 5], 3, 2],
      expected: null,
    },
  ],
  timeComplexity: 'O(m + n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl:
    'https://leetcode.com/problems/intersection-of-two-linked-lists/',
  youtubeUrl: 'https://www.youtube.com/watch?v=D0X0BONOQhI',
};
