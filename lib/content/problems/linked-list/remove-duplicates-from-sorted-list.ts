/**
 * Remove Duplicates from Sorted List
 * Problem ID: remove-duplicates-from-sorted-list
 * Order: 7
 */

import { Problem } from '../../../types';

export const remove_duplicates_from_sorted_listProblem: Problem = {
  id: 'remove-duplicates-from-sorted-list',
  title: 'Remove Duplicates from Sorted List',
  difficulty: 'Easy',
  topic: 'Linked List',
  order: 7,
  description: `Given the \`head\` of a sorted linked list, delete all duplicates such that each element appears only once. Return the linked list **sorted** as well.`,
  examples: [
    {
      input: 'head = [1,1,2]',
      output: '[1,2]',
    },
    {
      input: 'head = [1,1,2,3,3]',
      output: '[1,2,3]',
    },
  ],
  constraints: [
    'The number of nodes in the list is in the range [0, 300]',
    '-100 <= Node.val <= 100',
    'The list is guaranteed to be sorted in ascending order',
  ],
  hints: [
    'Traverse the list',
    'If current value equals next value, skip next',
    'Otherwise move to next node',
  ],
  starterCode: `from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def delete_duplicates(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Remove duplicates from sorted linked list.
    
    Args:
        head: Head of sorted linked list
        
    Returns:
        Head of list with duplicates removed
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 1, 2]],
      expected: [1, 2],
    },
    {
      input: [[1, 1, 2, 3, 3]],
      expected: [1, 2, 3],
    },
    {
      input: [[]],
      expected: [],
    },
  ],
  solution: `from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def delete_duplicates(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Single pass to remove duplicates.
    Time: O(n), Space: O(1)
    """
    if not head:
        return head
    
    current = head
    
    while current and current.next:
        if current.val == current.next.val:
            # Skip duplicate node
            current.next = current.next.next
        else:
            # Move to next distinct value
            current = current.next
    
    return head
`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl:
    'https://leetcode.com/problems/remove-duplicates-from-sorted-list/',
  youtubeUrl: 'https://www.youtube.com/watch?v=p10f-VpO4nE',
};
