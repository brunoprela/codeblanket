/**
 * Palindrome Linked List
 * Problem ID: palindrome-linked-list
 * Order: 8
 */

import { Problem } from '../../../types';

export const palindrome_linked_listProblem: Problem = {
  id: 'palindrome-linked-list',
  title: 'Palindrome Linked List',
  difficulty: 'Easy',
  topic: 'Linked List',
  order: 8,
  description: `Given the \`head\` of a singly linked list, return \`true\` if it is a **palindrome** or \`false\` otherwise.`,
  examples: [
    {
      input: 'head = [1,2,2,1]',
      output: 'true',
    },
    {
      input: 'head = [1,2]',
      output: 'false',
    },
  ],
  constraints: [
    'The number of nodes in the list is in the range [1, 10^5]',
    '0 <= Node.val <= 9',
  ],
  hints: [
    'Find middle of list using slow/fast pointers',
    'Reverse second half of list',
    'Compare first half with reversed second half',
    'Can you do it in O(n) time and O(1) space?',
  ],
  starterCode: `from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def is_palindrome(head: Optional[ListNode]) -> bool:
    """
    Check if linked list is a palindrome.
    
    Args:
        head: Head of linked list
        
    Returns:
        True if palindrome, False otherwise
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 2, 2, 1]],
      expected: true,
    },
    {
      input: [[1, 2]],
      expected: false,
    },
    {
      input: [[1]],
      expected: true,
    },
  ],
  solution: `from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def is_palindrome(head: Optional[ListNode]) -> bool:
    """
    Find middle, reverse second half, compare.
    Time: O(n), Space: O(1)
    """
    if not head or not head.next:
        return True
    
    # Find middle using slow/fast pointers
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    # Reverse second half
    prev = None
    while slow:
        next_node = slow.next
        slow.next = prev
        prev = slow
        slow = next_node
    
    # Compare first half with reversed second half
    left, right = head, prev
    while right:  # Only need to check right (shorter or equal)
        if left.val != right.val:
            return False
        left = left.next
        right = right.next
    
    return True

# Alternative: Using extra space
def is_palindrome_array(head: Optional[ListNode]) -> bool:
    """
    Convert to array and check palindrome.
    Time: O(n), Space: O(n)
    """
    values = []
    current = head
    while current:
        values.append(current.val)
        current = current.next
    
    return values == values[::-1]
`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1) for in-place, O(n) for array',
  leetcodeUrl: 'https://leetcode.com/problems/palindrome-linked-list/',
  youtubeUrl: 'https://www.youtube.com/watch?v=yOzXms1J6Nk',
};
