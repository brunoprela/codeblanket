import { Problem } from '@/lib/types';

export const sortingProblems: Problem[] = [
  // EASY #1 - Merge Sorted Arrays
  {
    id: 'merge-sorted-arrays',
    title: 'Merge Two Sorted Arrays',
    difficulty: 'Easy',
    topic: 'Sorting Algorithms',
    order: 1,
    description: `You are given two integer arrays \`nums1\` and \`nums2\`, sorted in **non-decreasing order**.

Merge \`nums2\` into \`nums1\` as one sorted array and return the result.

**Note:** You may assume that both arrays are already sorted.`,
    examples: [
      {
        input: 'nums1 = [1,2,3], nums2 = [2,5,6]',
        output: '[1,2,2,3,5,6]',
        explanation: 'The arrays we are merging are [1,2,3] and [2,5,6].',
      },
      {
        input: 'nums1 = [1], nums2 = []',
        output: '[1]',
        explanation: 'Only one array to merge.',
      },
      {
        input: 'nums1 = [], nums2 = [1]',
        output: '[1]',
        explanation: 'Only one array to merge.',
      },
    ],
    constraints: [
      'nums1.length == m',
      'nums2.length == n',
      '0 <= m, n <= 200',
      '-10^9 <= nums1[i], nums2[i] <= 10^9',
    ],
    hints: [
      'Use two pointers, one for each array',
      'Compare elements and take the smaller one',
      'This is the merge step from merge sort!',
    ],
    starterCode: `from typing import List

def merge_sorted_arrays(nums1: List[int], nums2: List[int]) -> List[int]:
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [1, 2, 3],
          [2, 5, 6],
        ],
        expected: [1, 2, 2, 3, 5, 6],
      },
      {
        input: [[1], []],
        expected: [1],
      },
      {
        input: [[], [1]],
        expected: [1],
      },
      {
        input: [
          [1, 3, 5],
          [2, 4, 6],
        ],
        expected: [1, 2, 3, 4, 5, 6],
      },
      {
        input: [
          [-5, -2, 0, 3],
          [-3, -1, 1, 4],
        ],
        expected: [-5, -3, -2, -1, 0, 1, 3, 4],
      },
    ],
    solution: `def merge_sorted_arrays(nums1, nums2):
    result = []
    i = j = 0
    
    # Merge while both arrays have elements
    while i < len(nums1) and j < len(nums2):
        if nums1[i] <= nums2[j]:
            result.append(nums1[i])
            i += 1
        else:
            result.append(nums2[j])
            j += 1
    
    # Add remaining elements from nums1
    while i < len(nums1):
        result.append(nums1[i])
        i += 1
    
    # Add remaining elements from nums2
    while j < len(nums2):
        result.append(nums2[j])
        j += 1
    
    return result

# Alternative: using Python's extend
def merge_sorted_arrays_v2(nums1, nums2):
    result = []
    i = j = 0
    
    while i < len(nums1) and j < len(nums2):
        if nums1[i] <= nums2[j]:
            result.append(nums1[i])
            i += 1
        else:
            result.append(nums2[j])
            j += 1
    
    result.extend(nums1[i:])
    result.extend(nums2[j:])
    
    return result
`,
    timeComplexity:
      'O(n + m) - we visit each element in both arrays exactly once',
    spaceComplexity:
      'O(n + m) - we create a new array to store the merged result',
    leetcodeUrl: 'https://leetcode.com/problems/merge-sorted-array/',
  },

  // EASY #2 - Sort Array By Parity
  {
    id: 'sort-array-parity',
    title: 'Sort Array By Parity',
    difficulty: 'Easy',
    topic: 'Sorting Algorithms',

    leetcodeUrl: 'https://leetcode.com/problems/sort-array-by-parity/',
    youtubeUrl: 'https://www.youtube.com/watch?v=6YZn-z5jkrg',
    order: 2,
    description: `Given an integer array \`nums\`, move all the even integers at the beginning of the array followed by all the odd integers.

Return **any array** that satisfies this condition.

**Challenge:** Can you do it in-place with O(1) extra space?`,
    examples: [
      {
        input: 'nums = [3,1,2,4]',
        output: '[2,4,3,1]',
        explanation:
          'The outputs [4,2,3,1], [2,4,1,3], and [4,2,1,3] would also be accepted.',
      },
      {
        input: 'nums = [0]',
        output: '[0]',
      },
    ],
    constraints: ['1 <= nums.length <= 5000', '0 <= nums[i] <= 5000'],
    hints: [
      'Two pointers: one at the start, one at the end',
      'Swap when left is odd and right is even',
      'Similar to the partition step in quicksort',
    ],
    starterCode: `from typing import List

def sort_array_by_parity(nums: List[int]) -> List[int]:
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[3, 1, 2, 4]],
        expected: [2, 4, 3, 1],
      },
      {
        input: [[0]],
        expected: [0],
      },
      {
        input: [[1, 2, 3, 4]],
        expected: [2, 4, 1, 3],
      },
      {
        input: [[2, 4, 6]],
        expected: [2, 4, 6],
      },
      {
        input: [[1, 3, 5]],
        expected: [1, 3, 5],
      },
    ],
    solution: `# Two-pointer in-place: O(n) time, O(1) space
def sort_array_by_parity(nums):
    left, right = 0, len(nums) - 1
    
    while left < right:
        # If left is even, move forward
        if nums[left] % 2 == 0:
            left += 1
        # If right is odd, move backward
        elif nums[right] % 2 == 1:
            right -= 1
        # Both in wrong position, swap
        else:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
    
    return nums

# Alternative: create new array (easier but uses O(n) space)
def sort_array_by_parity_v2(nums):
    even = [x for x in nums if x % 2 == 0]
    odd = [x for x in nums if x % 2 == 1]
    return even + odd

# Using Python's sort with custom key
def sort_array_by_parity_v3(nums):
    return sorted(nums, key=lambda x: x % 2)
`,
    timeComplexity: 'O(n) - single pass with two pointers',
    spaceComplexity: 'O(1) for in-place solution, O(n) for creating new arrays',
  },

  // EASY #3 - Insertion Sort List
  {
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
  },

  // MEDIUM #1 - Sort List
  {
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
  },

  // MEDIUM #2 - Wiggle Sort
  {
    id: 'wiggle-sort',
    title: 'Wiggle Sort',
    difficulty: 'Medium',
    topic: 'Sorting Algorithms',
    order: 5,
    description: `Given an integer array \`nums\`, reorder it such that \`nums[0] <= nums[1] >= nums[2] <= nums[3]...\`

You may assume the input array always has a valid answer.

**Challenge:** Can you do it in O(n) time without fully sorting?`,
    examples: [
      {
        input: 'nums = [3,5,2,1,6,4]',
        output: '[3,5,1,6,2,4]',
        explanation: '[1,6,2,5,3,4] is also accepted.',
      },
      {
        input: 'nums = [6,6,5,6,3,8]',
        output: '[6,6,5,6,3,8]',
      },
    ],
    constraints: ['1 <= nums.length <= 5 * 10^4', '0 <= nums[i] <= 10^4'],
    hints: [
      'Naive: sort then swap adjacent pairs',
      'Optimal: ensure even indices are <= next, odd indices are >= next',
      'One pass with local swaps is enough!',
    ],
    starterCode: `from typing import List

def wiggle_sort(nums: List[int]) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[[3, 5, 2, 1, 6, 4]]],
        expected: [3, 5, 1, 6, 2, 4],
      },
      {
        input: [[[6, 6, 5, 6, 3, 8]]],
        expected: [6, 6, 5, 6, 3, 8],
      },
      {
        input: [[[1, 2, 3]]],
        expected: [1, 3, 2],
      },
      {
        input: [[[1, 1, 1]]],
        expected: [1, 1, 1],
      },
      {
        input: [[[5, 3, 1, 2, 6, 7, 8, 5, 5]]],
        expected: [3, 5, 1, 6, 2, 7, 5, 8, 5],
      },
    ],
    solution: `# Optimal: O(n) time, O(1) space
def wiggle_sort(nums):
    for i in range(len(nums) - 1):
        if i % 2 == 0:
            # Even index: should be <= next
            if nums[i] > nums[i + 1]:
                nums[i], nums[i + 1] = nums[i + 1], nums[i]
        else:
            # Odd index: should be >= next
            if nums[i] < nums[i + 1]:
                nums[i], nums[i + 1] = nums[i + 1], nums[i]
    
    return nums  # For testing

# Naive approach: O(n log n) time
def wiggle_sort_sort(nums):
    nums.sort()
    
    # Swap pairs: 0-1, 2-3, 4-5, etc.
    for i in range(1, len(nums) - 1, 2):
        nums[i], nums[i + 1] = nums[i + 1], nums[i]
    
    return nums
`,
    timeComplexity:
      'O(n) optimal vs O(n log n) sorting - demonstrates avoiding unnecessary sorting',
    spaceComplexity: 'O(1) - in-place with local swaps',
    leetcodeUrl: 'https://leetcode.com/problems/wiggle-sort/',
  },

  // HARD - Count of Smaller Numbers After Self
  {
    id: 'count-smaller',
    title: 'Count of Smaller Numbers After Self',
    difficulty: 'Hard',
    topic: 'Sorting Algorithms',

    leetcodeUrl:
      'https://leetcode.com/problems/count-of-smaller-numbers-after-self/',
    youtubeUrl: 'https://www.youtube.com/watch?v=2SVLYsq5W8M',
    order: 6,
    description: `Given an integer array \`nums\`, return an integer array \`counts\` where \`counts[i]\` is the number of smaller elements to the right of \`nums[i]\`.

**Challenge:** Can you do better than O(n²)?

**Hint:** Modified merge sort can count inversions during the merge process!`,
    examples: [
      {
        input: 'nums = [5,2,6,1]',
        output: '[2,1,1,0]',
        explanation: `To the right of 5 there are 2 smaller elements (2 and 1).
To the right of 2 there is 1 smaller element (1).
To the right of 6 there is 1 smaller element (1).
To the right of 1 there is 0 smaller element.`,
      },
      {
        input: 'nums = [-1]',
        output: '[0]',
      },
      {
        input: 'nums = [-1,-1]',
        output: '[0,0]',
      },
    ],
    constraints: ['1 <= nums.length <= 10^5', '-10^4 <= nums[i] <= 10^4'],
    hints: [
      'Brute force: for each element, count how many after it are smaller - O(n²)',
      'Key insight: counting inversions is similar to merge sort',
      'During merge, track original indices',
    ],
    starterCode: `from typing import List

def count_smaller(nums: List[int]) -> List[int]:
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[5, 2, 6, 1]],
        expected: [2, 1, 1, 0],
      },
      {
        input: [[-1]],
        expected: [0],
      },
      {
        input: [[-1, -1]],
        expected: [0, 0],
      },
      {
        input: [[1, 2, 3, 4, 5]],
        expected: [0, 0, 0, 0, 0],
      },
      {
        input: [[5, 4, 3, 2, 1]],
        expected: [4, 3, 2, 1, 0],
      },
    ],
    solution: `# Optimal: Modified Merge Sort - O(n log n)
def count_smaller(nums):
    def merge_sort(arr):
        if len(arr) <= 1:
            return arr
        
        mid = len(arr) // 2
        left = merge_sort(arr[:mid])
        right = merge_sort(arr[mid:])
        
        return merge(left, right)
    
    def merge(left, right):
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i][1] <= right[j][1]:
                # Count how many from right are smaller
                counts[left[i][0]] += j
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        # Process remaining left elements
        while i < len(left):
            counts[left[i][0]] += j
            result.append(left[i])
            i += 1
        
        # Process remaining right elements
        while j < len(right):
            result.append(right[j])
            j += 1
        
        return result
    
    # Create array of (index, value) pairs
    indexed_nums = [(i, num) for i, num in enumerate(nums)]
    counts = [0] * len(nums)
    
    merge_sort(indexed_nums)
    
    return counts

# Brute Force: O(n²) - for comparison
def count_smaller_brute(nums):
    counts = []
    for i in range(len(nums)):
        count = 0
        for j in range(i + 1, len(nums)):
            if nums[j] < nums[i]:
                count += 1
        counts.append(count)
    return counts
`,
    timeComplexity:
      'O(n log n) with modified merge sort vs O(n²) brute force - demonstrates advanced sorting applications',
    spaceComplexity: 'O(n) for auxiliary arrays during merge sort',
  },
];
