/**
 * Merge k Sorted Lists
 * Problem ID: merge-k-sorted-lists
 * Order: 3
 */

import { Problem } from '../../../types';

export const merge_k_sorted_listsProblem: Problem = {
  id: 'merge-k-sorted-lists',
  title: 'Merge k Sorted Lists',
  difficulty: 'Hard',
  description: `You are given an array of \`k\` linked-lists \`lists\`, each linked-list is sorted in ascending order.

**Merge all the linked-lists into one sorted linked-list and return it.**


**Approach:**
Multiple approaches possible:

1. **Min Heap**: Use a min heap to track the smallest current element from each list. Time: O(N log K), Space: O(K)
2. **Divide & Conquer**: Recursively merge pairs of lists. Time: O(N log K), Space: O(log K)
3. **Naive**: Merge one list at a time. Time: O(NK), Space: O(1)

The min heap approach is most intuitive and efficient.`,
  examples: [
    {
      input: 'lists = [[1,4,5],[1,3,4],[2,6]]',
      output: '[1,1,2,3,4,4,5,6]',
      explanation:
        'The linked-lists are:\n[\n  1->4->5,\n  1->3->4,\n  2->6\n]\nmerging them into one sorted list:\n1->1->2->3->4->4->5->6',
    },
    {
      input: 'lists = []',
      output: '[]',
    },
    {
      input: 'lists = [[]]',
      output: '[]',
    },
  ],
  constraints: [
    'k == lists.length',
    '0 <= k <= 10^4',
    '0 <= lists[i].length <= 500',
    '-10^4 <= lists[i][j] <= 10^4',
    'lists[i] is sorted in ascending order',
    'The sum of lists[i].length will not exceed 10^4',
  ],
  hints: [
    'Use a min heap to efficiently find the smallest element among k lists',
    'Store (value, list_index, node) in the heap',
    'Extract min, add to result, and push the next node from that list',
    'Alternative: Divide and conquer by merging pairs recursively',
    'Python heapq module provides min heap functionality',
  ],
  starterCode: `from typing import List, Optional
import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge_k_lists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    Merge k sorted linked lists.
    
    Args:
        lists: Array of k sorted linked lists
        
    Returns:
        Head of the merged sorted list
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [
          [
            [1, 4, 5],
            [1, 3, 4],
            [2, 6],
          ],
        ],
      ],
      expected: [1, 1, 2, 3, 4, 4, 5, 6],
    },
    {
      input: [[[[]]]],
      expected: [],
    },
    {
      input: [[[]]],
      expected: [],
    },
    {
      input: [[[[1], [0]]]],
      expected: [0, 1],
    },
  ],
  solution: `from typing import List, Optional
import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def merge_k_lists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    Min Heap approach.
    Time: O(N log K) where N is total nodes, K is number of lists
    Space: O(K) for the heap
    """
    if not lists:
        return None
    
    # Min heap: (value, index, node)
    # Index prevents comparison of nodes when values are equal
    heap = []
    
    # Add first node from each list
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))
    
    dummy = ListNode(0)
    curr = dummy
    
    while heap:
        val, i, node = heapq.heappop(heap)
        
        # Add to result
        curr.next = node
        curr = curr.next
        
        # Add next node from same list
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    
    return dummy.next


# Alternative: Divide and Conquer
def merge_k_lists_divide_conquer(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    Divide and conquer approach.
    Time: O(N log K), Space: O(log K) for recursion stack
    """
    if not lists:
        return None
    if len(lists) == 1:
        return lists[0]
    
    def merge_two(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        """Merge two sorted lists."""
        dummy = ListNode(0)
        curr = dummy
        
        while l1 and l2:
            if l1.val <= l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next
        
        curr.next = l1 if l1 else l2
        return dummy.next
    
    def merge_range(lists: List[Optional[ListNode]], left: int, right: int) -> Optional[ListNode]:
        """Merge lists in range [left, right]."""
        if left == right:
            return lists[left]
        
        mid = (left + right) // 2
        l1 = merge_range(lists, left, mid)
        l2 = merge_range(lists, mid + 1, right)
        
        return merge_two(l1, l2)
    
    return merge_range(lists, 0, len(lists) - 1)


# Alternative: Sequential merge (naive)
def merge_k_lists_naive(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    Merge one list at a time.
    Time: O(N * K), Space: O(1)
    """
    if not lists:
        return None
    
    def merge_two(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        curr = dummy
        
        while l1 and l2:
            if l1.val <= l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next
        
        curr.next = l1 if l1 else l2
        return dummy.next
    
    result = lists[0]
    for i in range(1, len(lists)):
        result = merge_two(result, lists[i])
    
    return result`,
  timeComplexity: 'O(N log K) where N is total nodes, K is number of lists',
  spaceComplexity: 'O(K) for min heap, O(log K) for divide & conquer',

  leetcodeUrl: 'https://leetcode.com/problems/merge-k-sorted-lists/',
  youtubeUrl: 'https://www.youtube.com/watch?v=q5a5OiGbT6Q',
  order: 3,
  topic: 'Linked List',
};
