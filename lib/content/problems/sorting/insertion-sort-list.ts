/**
 * Insertion Sort List
 * Problem ID: insertion-sort-list
 * Order: 3
 */

import { Problem } from '../../../types';

export const insertion_sort_listProblem: Problem = {
  id: 'insertion-sort-list',
  title: 'Insertion Sort List',
  difficulty: 'Easy',
  topic: 'Sorting Algorithms',

  leetcodeUrl: 'https://leetcode.com/problems/insertion-sort-list/',
  youtubeUrl: 'https://www.youtube.com/watch?v=Kk6mXAzQ3zs',
  order: 3,
  description: `Given the \`head\` of a singly linked list, sort the list using **insertion sort**, and return the sorted list's head.

The steps of the **insertion sort** algorithm:
1. Start with one element (the first node) as the sorted portion
2. Take the next element and insert it into the correct position in the sorted portion
3. Repeat until all elements are sorted

For this problem, implement insertion sort on a linked list.`,
  examples: [
    {
      input: 'head = [4,2,1,3]',
      output: '[1,2,3,4]',
    },
    {
      input: 'head = [-1,5,3,4,0]',
      output: '[-1,0,3,4,5]',
    },
  ],
  constraints: [
    'The number of nodes in the list is in the range [1, 5000]',
    '-5000 <= Node.val <= 5000',
  ],
  hints: [
    'Maintain a separate sorted list and insert each node into it',
    'For each node, find where it should go in the sorted portion',
    'Use a dummy node to simplify edge cases',
  ],
  starterCode: `class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def insertion_sort_list(head: ListNode) -> ListNode:
    # Write your code here
    pass

# Helper function to convert list to linked list (for testing)
def list_to_linked_list(arr):
    if not arr:
        return None
    head = ListNode(arr[0])
    current = head
    for val in arr[1:]:
        current.next = ListNode(val)
        current = current.next
    return head

# Helper function to convert linked list to list (for testing)
def linked_list_to_list(head):
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result
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
      input: [[1]],
      expected: [1],
    },
    {
      input: [[3, 2, 1]],
      expected: [1, 2, 3],
    },
    {
      input: [[1, 1, 1]],
      expected: [1, 1, 1],
    },
  ],
  solution: `class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def insertion_sort_list(head):
    if not head or not head.next:
        return head
    
    # Dummy node to simplify insertion
    dummy = ListNode(0)
    current = head
    
    while current:
        # Save next node before we modify current
        next_node = current.next
        
        # Find position to insert current node
        prev = dummy
        while prev.next and prev.next.val < current.val:
            prev = prev.next
        
        # Insert current node
        current.next = prev.next
        prev.next = current
        
        # Move to next node
        current = next_node
    
    return dummy.next

# Helper functions
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
    'O(n²) - for each of n nodes, we may scan through sorted portion (worst case n nodes)',
  spaceComplexity:
    'O(1) - we only rearrange pointers, no extra data structures needed',
};
