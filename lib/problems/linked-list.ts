import { Problem } from '../types';

export const linkedListProblems: Problem[] = [
  {
    id: 'reverse-linked-list',
    title: 'Reverse Linked List',
    difficulty: 'Easy',
    description: `Given the \`head\` of a singly linked list, reverse the list, and return **the reversed list**.


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


**Approach:**
Use Floyd Cycle Detection Algorithm (tortoise and hare). Use two pointers: slow moves one step at a time, fast moves two steps. If there is a cycle, they will eventually meet. If fast reaches None, there is no cycle.`,
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
  // EASY - Middle of the Linked List
  {
    id: 'middle-of-linked-list',
    title: 'Middle of the Linked List',
    difficulty: 'Easy',
    topic: 'Linked List',
    order: 4,
    description: `Given the \`head\` of a singly linked list, return the middle node of the linked list.

If there are two middle nodes, return **the second middle** node.`,
    examples: [
      {
        input: 'head = [1,2,3,4,5]',
        output: '[3,4,5]',
        explanation: 'The middle node of the list is node 3.',
      },
      {
        input: 'head = [1,2,3,4,5,6]',
        output: '[4,5,6]',
        explanation: 'Since the list has two middle nodes with values 3 and 4, we return the second one.',
      },
    ],
    constraints: [
      'The number of nodes in the list is in the range [1, 100]',
      '1 <= Node.val <= 100',
    ],
    hints: [
      'Use slow and fast pointers (tortoise and hare)',
      'Fast moves twice as fast as slow',
      'When fast reaches end, slow is at middle',
    ],
    starterCode: `from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def middle_node(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Find the middle node of linked list.
    
    Args:
        head: Head of the linked list
        
    Returns:
        Middle node (second middle if even length)
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 2, 3, 4, 5]],
        expected: [3, 4, 5],
      },
      {
        input: [[1, 2, 3, 4, 5, 6]],
        expected: [4, 5, 6],
      },
      {
        input: [[1]],
        expected: [1],
      },
    ],
    solution: `from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def middle_node(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Two-pointer approach (slow and fast).
    Time: O(n), Space: O(1)
    """
    slow = fast = head
    
    # Fast moves twice as fast as slow
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    # Slow is at middle
    return slow
`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/middle-of-the-linked-list/',
    youtubeUrl: 'https://www.youtube.com/watch?v=A2_ldqM4QcY',
  },
  // EASY - Delete Node in a Linked List
  {
    id: 'delete-node-in-linked-list',
    title: 'Delete Node in a Linked List',
    difficulty: 'Easy',
    topic: 'Linked List',
    order: 5,
    description: `There is a singly-linked list \`head\` and we want to delete a node \`node\` in it.

You are given the node to be deleted \`node\`. You will **not be given access** to the first node of \`head\`.

All the values of the linked list are **unique**, and it is guaranteed that the given node \`node\` is **not the last node** in the linked list.

Delete the given node. Note that by deleting the node, we do not mean removing it from memory. We mean:
- The value of the given node should not exist in the linked list.
- The number of nodes in the linked list should decrease by one.
- All the values before \`node\` should be in the same order.
- All the values after \`node\` should be in the same order.`,
    examples: [
      {
        input: 'head = [4,5,1,9], node = 5',
        output: '[4,1,9]',
        explanation: 'You are given the second node with value 5. After deleting it, the linked list becomes 4 -> 1 -> 9.',
      },
      {
        input: 'head = [4,5,1,9], node = 1',
        output: '[4,5,9]',
      },
    ],
    constraints: [
      'The number of the nodes in the given list is in the range [2, 1000]',
      '-1000 <= Node.val <= 1000',
      'The value of each node in the list is unique',
      'The node to be deleted is in the list and is not a tail node',
    ],
    hints: [
      'Copy the value from the next node to current node',
      'Then delete the next node instead',
      'This effectively "deletes" the current node',
    ],
    starterCode: `class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def delete_node(node: ListNode) -> None:
    """
    Delete given node (not given head!).
    
    Args:
        node: The node to delete
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[4, 5, 1, 9], 5],
        expected: [4, 1, 9],
      },
      {
        input: [[4, 5, 1, 9], 1],
        expected: [4, 5, 9],
      },
    ],
    solution: `class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def delete_node(node: ListNode) -> None:
    """
    Copy next node's value and skip next node.
    Time: O(1), Space: O(1)
    """
    # Copy value from next node
    node.val = node.next.val
    
    # Skip the next node
    node.next = node.next.next
`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/delete-node-in-a-linked-list/',
    youtubeUrl: 'https://www.youtube.com/watch?v=f1r_jFWRbH8',
  },
  // EASY - Merge Two Sorted Lists
  {
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
        input: [[1, 2, 4], [1, 3, 4]],
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
  },
  // EASY - Remove Duplicates from Sorted List
  {
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
    leetcodeUrl: 'https://leetcode.com/problems/remove-duplicates-from-sorted-list/',
    youtubeUrl: 'https://www.youtube.com/watch?v=p10f-VpO4nE',
  },
  // EASY - Palindrome Linked List
  {
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
  },
];
