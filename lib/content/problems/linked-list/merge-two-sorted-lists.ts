/**
 * Merge Two Sorted Lists
 * Problem ID: merge-two-sorted-lists
 * Order: 6
 */

import { Problem } from '../../../types';

export const merge_two_sorted_listsProblem: Problem = {
  id: 'merge-two-sorted-lists',
  title: 'Merge Two Sorted Lists',
  difficulty: 'Easy',
  topic: 'Linked List',
  order: 6,
  description: `You are given the heads of two sorted linked lists \`list1\` and \`list2\`.

Merge the two lists into one **sorted** list. The list should be made by splicing together the nodes of the first two lists.

Return the head of the merged linked list.`,
  examples: [
    {
      input: 'list1 = [1,2,4], list2 = [1,3,4]',
      output: '[1,1,2,3,4,4]',
    },
    {
      input: 'list1 = [], list2 = []',
      output: '[]',
    },
    {
      input: 'list1 = [], list2 = [0]',
      output: '[0]',
    },
  ],
  constraints: [
    'The number of nodes in both lists is in the range [0, 50]',
    '-100 <= Node.val <= 100',
    'Both list1 and list2 are sorted in non-decreasing order',
  ],
  hints: [
    'Use a dummy head to simplify edge cases',
    'Compare values from both lists',
    'Append smaller value to result',
    'Attach remaining list at the end',
  ],
  starterCode: `from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge_two_lists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    """
    Merge two sorted linked lists.
    
    Args:
        list1: Head of first sorted list
        list2: Head of second sorted list
        
    Returns:
        Head of merged sorted list
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [1, 2, 4],
        [1, 3, 4],
      ],
      expected: [1, 1, 2, 3, 4, 4],
    },
    {
      input: [[], []],
      expected: [],
    },
    {
      input: [[], [0]],
      expected: [0],
    },
  ],
  solution: `from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge_two_lists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    """
    Iterative merge with dummy head.
    Time: O(n + m), Space: O(1)
    """
    # Dummy head simplifies edge cases
    dummy = ListNode(0)
    current = dummy
    
    # Merge while both lists have nodes
    while list1 and list2:
        if list1.val <= list2.val:
            current.next = list1
            list1 = list1.next
        else:
            current.next = list2
            list2 = list2.next
        current = current.next
    
    # Attach remaining list
    current.next = list1 if list1 else list2
    
    return dummy.next

# Alternative: Recursive approach
def merge_two_lists_recursive(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    """
    Recursive merge.
    Time: O(n + m), Space: O(n + m) for recursion stack
    """
    if not list1:
        return list2
    if not list2:
        return list1
    
    if list1.val <= list2.val:
        list1.next = merge_two_lists_recursive(list1.next, list2)
        return list1
    else:
        list2.next = merge_two_lists_recursive(list1, list2.next)
        return list2
`,
  timeComplexity: 'O(n + m)',
  spaceComplexity: 'O(1) iterative, O(n + m) recursive',
  leetcodeUrl: 'https://leetcode.com/problems/merge-two-sorted-lists/',
  youtubeUrl: 'https://www.youtube.com/watch?v=XIdigk956u0',
};
