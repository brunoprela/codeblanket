/**
 * Sort List (Merge Sort)
 * Problem ID: sort-list
 * Order: 4
 */

import { Problem } from '../../../types';

export const sort_listProblem: Problem = {
  id: 'sort-list',
  title: 'Sort List (Merge Sort)',
  difficulty: 'Medium',
  topic: 'Sorting Algorithms',

  leetcodeUrl: 'https://leetcode.com/problems/sort-list/',
  youtubeUrl: 'https://www.youtube.com/watch?v=TGveA1oFhrc',
  order: 4,
  description: `Given the \`head\` of a linked list, return the list after sorting it in **ascending order**.

**Challenge:** Can you sort the linked list in O(n log n) time and O(1) space?

**Hint:** Use merge sort, which works perfectly for linked lists!`,
  examples: [
    {
      input: 'head = [4,2,1,3]',
      output: '[1,2,3,4]',
    },
    {
      input: 'head = [-1,5,3,4,0]',
      output: '[-1,0,3,4,5]',
    },
    {
      input: 'head = []',
      output: '[]',
    },
  ],
  constraints: [
    'The number of nodes in the list is in the range [0, 5 * 10^4]',
    '-10^5 <= Node.val <= 10^5',
  ],
  hints: [
    'Merge sort is ideal for linked lists',
    'Use slow/fast pointers to find the middle',
    'Split the list in half, sort each half, then merge',
  ],
  starterCode: `class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def sort_list(head: ListNode) -> ListNode:
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[4, 2, 1, 3]],
      expected: [1, 2, 3, 4],
    },
    {
      input: [[-1, 5, 3, 4, 0]],
      expected: [-1, 0, 3, 4, 5],
    },
    {
      input: [[]],
      expected: [],
    },
    {
      input: [[1]],
      expected: [1],
    },
    {
      input: [[2, 1]],
      expected: [1, 2],
    },
  ],
  solution: `class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def sort_list(head):
    # Base case
    if not head or not head.next:
        return head
    
    # Find middle using slow/fast pointers
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    # Split the list
    mid = slow.next
    slow.next = None
    
    # Recursively sort both halves
    left = sort_list(head)
    right = sort_list(mid)
    
    # Merge sorted halves
    return merge(left, right)

def merge(l1, l2):
    dummy = ListNode(0)
    current = dummy
    
    while l1 and l2:
        if l1.val <= l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    
    # Attach remaining nodes
    current.next = l1 if l1 else l2
    
    return dummy.next

# Helper functions for testing
def list_to_linked_list(arr):
    if not arr:
        return None
    head = ListNode(arr[0])
    current = head
    for val in arr[1:]:
        current.next = ListNode(val)
        current = current.next
    return head

def linked_list_to_list(head):
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result
`,
  timeComplexity:
    'O(n log n) - divide list log n times, merge takes O(n) each time',
  spaceComplexity:
    'O(log n) for recursion stack - in-place for the linked list itself',
};
