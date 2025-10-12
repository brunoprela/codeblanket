import { Problem } from '../types';

export const linkedListProblems: Problem[] = [
  {
    id: 'reverse-linked-list',
    title: 'Reverse Linked List',
    difficulty: 'Easy',
    description: `Given the \`head\` of a singly linked list, reverse the list, and return **the reversed list**.

**LeetCode:** [206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)
**YouTube:** [NeetCode - Reverse Linked List](https://www.youtube.com/watch?v=G0_I-ZF0S38)

**Approach:**
Iterate through the list, reversing the \`next\` pointer of each node. Use three pointers: \`prev\`, \`curr\`, and \`next_temp\` to safely reverse links without losing references.

**Alternative:** Can also be solved recursively, but uses O(N) stack space.`,
    examples: [
      {
        input: 'head = [1,2,3,4,5]',
        output: '[5,4,3,2,1]',
        explanation: 'The list is reversed.',
      },
      {
        input: 'head = [1,2]',
        output: '[2,1]',
      },
      {
        input: 'head = []',
        output: '[]',
      },
    ],
    constraints: [
      'The number of nodes in the list is the range [0, 5000]',
      '-5000 <= Node.val <= 5000',
    ],
    hints: [
      'Use three pointers: prev, curr, and next_temp',
      'Save the next node before changing curr.next',
      'Reverse the pointer: curr.next = prev',
      'Move all pointers forward: prev = curr, curr = next_temp',
      'Return prev at the end (new head)',
    ],
    starterCode: `from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Reverse a singly linked list.
    
    Args:
        head: Head of the linked list
        
    Returns:
        Head of the reversed list
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[[1, 2, 3, 4, 5]]],
        expected: [5, 4, 3, 2, 1],
      },
      {
        input: [[[1, 2]]],
        expected: [2, 1],
      },
      {
        input: [[[]]],
        expected: [],
      },
      {
        input: [[[1]]],
        expected: [1],
      },
    ],
    solution: `from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Iterative solution.
    Time: O(N), Space: O(1)
    """
    prev = None
    curr = head
    
    while curr:
        next_temp = curr.next  # Save next
        curr.next = prev       # Reverse link
        prev = curr            # Move prev forward
        curr = next_temp       # Move curr forward
    
    return prev  # New head


# Alternative: Recursive solution
def reverse_list_recursive(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Recursive solution.
    Time: O(N), Space: O(N) for call stack
    """
    # Base case
    if not head or not head.next:
        return head
    
    # Recursively reverse the rest
    new_head = reverse_list_recursive(head.next)
    
    # Reverse the link
    head.next.next = head
    head.next = None
    
    return new_head`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1) iterative, O(n) recursive',
    order: 1,
    topic: 'Linked List',
    leetcodeUrl: 'https://leetcode.com/problems/reverse-linked-list/',
    youtubeUrl: 'https://www.youtube.com/watch?v=G0_I-ZF0S38',
  },
  {
    id: 'linked-list-cycle',
    title: 'Linked List Cycle',
    difficulty: 'Medium',
    description: `Given \`head\`, the head of a linked list, determine if the linked list has a **cycle** in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the \`next\` pointer. Internally, \`pos\` is used to denote the index of the node that tail's \`next\` pointer is connected to. **Note that \`pos\` is not passed as a parameter**.

Return \`true\` if there is a cycle in the linked list. Otherwise, return \`false\`.

**LeetCode:** [141. Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)
**YouTube:** [NeetCode - Linked List Cycle](https://www.youtube.com/watch?v=gBTe7lFR3vc)

**Approach:**
Use Floyd's Cycle Detection Algorithm (tortoise and hare). Use two pointers: slow moves one step at a time, fast moves two steps. If there's a cycle, they will eventually meet. If fast reaches None, there's no cycle.`,
    examples: [
      {
        input: 'head = [3,2,0,-4], pos = 1',
        output: 'true',
        explanation:
          'There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).',
      },
      {
        input: 'head = [1,2], pos = 0',
        output: 'true',
        explanation:
          'There is a cycle in the linked list, where the tail connects to the 0th node.',
      },
      {
        input: 'head = [1], pos = -1',
        output: 'false',
        explanation: 'There is no cycle in the linked list.',
      },
    ],
    constraints: [
      'The number of the nodes in the list is in the range [0, 10^4]',
      '-10^5 <= Node.val <= 10^5',
      'pos is -1 or a valid index in the linked-list',
    ],
    hints: [
      'Use two pointers: slow and fast',
      'Move slow one step, fast two steps in each iteration',
      'If fast reaches None, there is no cycle',
      'If slow equals fast at any point, there is a cycle',
      "This is Floyd's Cycle Detection Algorithm",
    ],
    starterCode: `from typing import Optional

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

def has_cycle(head: Optional[ListNode]) -> bool:
    """
    Detect if linked list has a cycle.
    
    Args:
        head: Head of the linked list
        
    Returns:
        True if cycle exists, False otherwise
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[[3, 2, 0, -4], 1]],
        expected: true,
      },
      {
        input: [[[1, 2], 0]],
        expected: true,
      },
      {
        input: [[[1], -1]],
        expected: false,
      },
      {
        input: [[[], -1]],
        expected: false,
      },
    ],
    solution: `from typing import Optional

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


def has_cycle(head: Optional[ListNode]) -> bool:
    """
    Floyd's Cycle Detection (Tortoise and Hare).
    Time: O(N), Space: O(1)
    """
    if not head or not head.next:
        return False
    
    slow = head
    fast = head
    
    while fast and fast.next:
        slow = slow.next          # Move 1 step
        fast = fast.next.next     # Move 2 steps
        
        if slow == fast:          # They met - cycle exists
            return True
    
    return False  # fast reached end - no cycle


# Alternative: Using hash set (less optimal)
def has_cycle_set(head: Optional[ListNode]) -> bool:
    """
    Hash set approach.
    Time: O(N), Space: O(N)
    """
    visited = set()
    curr = head
    
    while curr:
        if curr in visited:
            return True
        visited.add(curr)
        curr = curr.next
    
    return False


# Bonus: Find where cycle starts
def detect_cycle_start(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Find the node where cycle begins.
    Time: O(N), Space: O(1)
    """
    if not head or not head.next:
        return None
    
    # Phase 1: Detect if cycle exists
    slow = fast = head
    has_cycle = False
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            has_cycle = True
            break
    
    if not has_cycle:
        return None
    
    # Phase 2: Find cycle start
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow  # This is where cycle starts`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    order: 2,
    topic: 'Linked List',
    leetcodeUrl: 'https://leetcode.com/problems/linked-list-cycle/',
    youtubeUrl: 'https://www.youtube.com/watch?v=gBTe7lFR3vc',
  },
  {
    id: 'merge-k-sorted-lists',
    title: 'Merge k Sorted Lists',
    difficulty: 'Hard',
    description: `You are given an array of \`k\` linked-lists \`lists\`, each linked-list is sorted in ascending order.

**Merge all the linked-lists into one sorted linked-list and return it.**

**LeetCode:** [23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)
**YouTube:** [NeetCode - Merge k Sorted Lists](https://www.youtube.com/watch?v=q5a5OiGbT6Q)

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
    order: 3,
    topic: 'Linked List',
    leetcodeUrl: 'https://leetcode.com/problems/merge-k-sorted-lists/',
    youtubeUrl: 'https://www.youtube.com/watch?v=q5a5OiGbT6Q',
  },
];
