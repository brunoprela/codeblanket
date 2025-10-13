import { Problem } from '@/lib/types';

export const timeSpaceComplexityProblems: Problem[] = [
  // EASY #1 - Find Pivot Index
  {
    id: 'pivot-index',
    title: 'Find Pivot Index',
    difficulty: 'Easy',
    topic: 'Time & Space Complexity',
    order: 2,
    description: `Given an array of integers \`nums\`, calculate the **pivot index** of this array.

The pivot index is the index where the sum of all the numbers **strictly** to the left of the index is equal to the sum of all the numbers **strictly** to the index's right.

If the index is on the left edge of the array, then the left sum is \`0\` because there are no elements to the left. This also applies to the right edge of the array.

Return the **leftmost pivot index**. If no such index exists, return \`-1\`.`,
    examples: [
      {
        input: 'nums = [1,7,3,6,5,6]',
        output: '3',
        explanation:
          'The pivot index is 3. Left sum = 1 + 7 + 3 = 11. Right sum = 5 + 6 = 11.',
      },
      {
        input: 'nums = [1,2,3]',
        output: '-1',
        explanation: 'There is no index that satisfies the conditions.',
      },
      {
        input: 'nums = [2,1,-1]',
        output: '0',
        explanation:
          'The pivot index is 0. Left sum = 0. Right sum = 1 + (-1) = 0.',
      },
    ],
    constraints: ['1 <= nums.length <= 10^4', '-1000 <= nums[i] <= 1000'],
    hints: [
      'Can you compute left and right sums for each index? What complexity?',
      'Better: compute total sum once, then track left sum as you iterate',
      'At each position: right_sum = total_sum - left_sum - nums[i]',
    ],
    starterCode: `from typing import List

def pivot_index(nums: List[int]) -> int:
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 7, 3, 6, 5, 6]],
        expected: 3,
      },
      {
        input: [[1, 2, 3]],
        expected: -1,
      },
      {
        input: [[2, 1, -1]],
        expected: 0,
      },
      {
        input: [[1, 1, 1, 1, 1, 1]],
        expected: -1,
      },
      {
        input: [[-1, -1, -1, -1, -1, 0]],
        expected: 2,
      },
    ],
    solution: `# Optimal: O(n) time, O(1) space
def pivot_index(nums):
    total_sum = sum(nums)
    left_sum = 0
    
    for i in range(len(nums)):
        # Right sum = total - left - current
        right_sum = total_sum - left_sum - nums[i]
        
        if left_sum == right_sum:
            return i
        
        left_sum += nums[i]
    
    return -1

# Naive: Compute sums for each index - O(n²) time
def pivot_index_naive(nums):
    for i in range(len(nums)):
        left_sum = sum(nums[:i])
        right_sum = sum(nums[i+1:])
        if left_sum == right_sum:
            return i
    return -1
`,
    timeComplexity:
      'O(n) optimal vs O(n²) naive - demonstrates precomputation optimization',
    spaceComplexity: 'O(1) - only tracking sums',
    leetcodeUrl: 'https://leetcode.com/problems/find-pivot-index/',
    youtubeUrl: 'https://www.youtube.com/watch?v=u89i60lYx8U',
  },

  // EASY #3 - First Unique Character
  {
    id: 'first-unique-char',
    title: 'First Unique Character in a String',
    difficulty: 'Easy',
    topic: 'Time & Space Complexity',
    order: 3,
    description: `Given a string \`s\`, find the first non-repeating character in it and return its index. If it does not exist, return \`-1\`.

**Complexity Analysis:** What's the time and space complexity of using a hash map?`,
    examples: [
      {
        input: 's = "leetcode"',
        output: '0',
      },
      {
        input: 's = "loveleetcode"',
        output: '2',
      },
      {
        input: 's = "aabb"',
        output: '-1',
      },
    ],
    constraints: [
      '1 <= s.length <= 10^5',
      's consists of only lowercase English letters',
    ],
    hints: [
      'First pass: count character frequencies',
      'Second pass: find first character with count = 1',
      'Can you do it in O(n) time with O(1) space? (26 letters is constant!)',
    ],
    starterCode: `def first_uniq_char(s: str) -> int:
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['leetcode'],
        expected: 0,
      },
      {
        input: ['loveleetcode'],
        expected: 2,
      },
      {
        input: ['aabb'],
        expected: -1,
      },
      {
        input: ['z'],
        expected: 0,
      },
      {
        input: ['aabbccddee'],
        expected: -1,
      },
    ],
    solution: `# Optimal: Hash Map - O(n) time, O(1) space (26 letters max)
def first_uniq_char(s):
    # Count frequencies
    char_count = {}
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1
    
    # Find first unique
    for i, char in enumerate(s):
        if char_count[char] == 1:
            return i
    
    return -1

# Using Counter
from collections import Counter
def first_uniq_char_counter(s):
    count = Counter(s)
    for i, char in enumerate(s):
        if count[char] == 1:
            return i
    return -1

# Naive: Check each character - O(n²) time
def first_uniq_char_naive(s):
    for i in range(len(s)):
        is_unique = True
        for j in range(len(s)):
            if i != j and s[i] == s[j]:
                is_unique = False
                break
        if is_unique:
            return i
    return -1
`,
    timeComplexity:
      'O(n) with hash map vs O(n²) naive - two passes vs nested loops',
    spaceComplexity: 'O(1) - at most 26 lowercase English letters in hash map',
    leetcodeUrl:
      'https://leetcode.com/problems/first-unique-character-in-a-string/',
    youtubeUrl: 'https://www.youtube.com/watch?v=5co5Gvp_-S0',
  },

  // MEDIUM #1 - Find the Duplicate Number
  {
    id: 'duplicate-number',
    title: 'Find the Duplicate Number',
    difficulty: 'Medium',
    topic: 'Time & Space Complexity',
    order: 4,
    description: `Given an array of integers \`nums\` containing \`n + 1\` integers where each integer is in the range \`[1, n]\` inclusive.

There is only **one repeated number** in \`nums\`, return this repeated number.

You must solve the problem **without** modifying the array \`nums\` and uses only **constant** extra space.

**Challenge:** Can you solve it in O(n) time and O(1) space?`,
    examples: [
      {
        input: 'nums = [1,3,4,2,2]',
        output: '2',
      },
      {
        input: 'nums = [3,1,3,4,2]',
        output: '3',
      },
    ],
    constraints: [
      '1 <= n <= 10^5',
      'nums.length == n + 1',
      '1 <= nums[i] <= n',
      'All the integers in nums appear only once except for precisely one integer which appears two or more times',
    ],
    hints: [
      'Hash set uses O(n) space - violates constraint',
      'Sorting modifies array - violates constraint',
      'Think of it as a linked list cycle detection problem!',
      "Floyd's Tortoise and Hare algorithm: O(n) time, O(1) space",
    ],
    starterCode: `from typing import List

def find_duplicate(nums: List[int]) -> int:
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 3, 4, 2, 2]],
        expected: 2,
      },
      {
        input: [[3, 1, 3, 4, 2]],
        expected: 3,
      },
      {
        input: [[2, 5, 9, 6, 9, 3, 8, 9, 7, 1, 4]],
        expected: 9,
      },
      {
        input: [[1, 1]],
        expected: 1,
      },
      {
        input: [[1, 1, 2]],
        expected: 1,
      },
    ],
    solution: `# Optimal: Floyd's Cycle Detection - O(n) time, O(1) space
def find_duplicate(nums):
    # Phase 1: Find intersection point in the cycle
    slow = fast = nums[0]
    
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    
    # Phase 2: Find entrance to the cycle (the duplicate)
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    
    return slow

# Hash Set - O(n) time, O(n) space (violates space constraint)
def find_duplicate_set(nums):
    seen = set()
    for num in nums:
        if num in seen:
            return num
        seen.add(num)
    return -1

# Sorting - O(n log n) time, O(1) space (but modifies array - violates constraint)
def find_duplicate_sort(nums):
    nums.sort()
    for i in range(1, len(nums)):
        if nums[i] == nums[i-1]:
            return nums[i]
    return -1
`,
    timeComplexity: "O(n) optimal with Floyd's vs O(n log n) with sorting",
    spaceComplexity:
      'O(1) optimal - demonstrates clever algorithm avoiding extra space',
    leetcodeUrl: 'https://leetcode.com/problems/find-the-duplicate-number/',
    youtubeUrl: 'https://www.youtube.com/watch?v=wjYnzkAhcNk',
  },

  // MEDIUM #2 - Array Partition
  {
    id: 'array-partition',
    title: 'Array Partition',
    difficulty: 'Medium',
    topic: 'Time & Space Complexity',
    order: 5,
    description: `Given an integer array \`nums\` of \`2n\` integers, group these integers into \`n\` pairs \`(a1, b1), (a2, b2), ..., (an, bn)\` such that the sum of \`min(ai, bi)\` for all \`i\` is **maximized**. Return the maximized sum.

**Key Insight:** To maximize the sum of minimums, pair adjacent elements after sorting!`,
    examples: [
      {
        input: 'nums = [1,4,3,2]',
        output: '4',
        explanation:
          'All possible pairings:\n1. (1, 4), (2, 3) -> min(1, 4) + min(2, 3) = 1 + 2 = 3\n2. (1, 3), (2, 4) -> min(1, 3) + min(2, 4) = 1 + 2 = 3\n3. (1, 2), (3, 4) -> min(1, 2) + min(3, 4) = 1 + 3 = 4 (maximum)',
      },
      {
        input: 'nums = [6,2,6,5,1,2]',
        output: '9',
        explanation:
          'Optimal pairing: (2, 1), (2, 5), (6, 6). Sum of mins = 1 + 2 + 6 = 9.',
      },
    ],
    constraints: [
      '1 <= n <= 10^4',
      'nums.length == 2 * n',
      '-10^4 <= nums[i] <= 10^4',
    ],
    hints: [
      'To maximize sum of minimums, avoid wasting large numbers',
      'Sort the array first',
      'After sorting, pair adjacent elements',
    ],
    starterCode: `from typing import List

def array_pair_sum(nums: List[int]) -> int:
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 4, 3, 2]],
        expected: 4,
      },
      {
        input: [[6, 2, 6, 5, 1, 2]],
        expected: 9,
      },
      {
        input: [[1, 2]],
        expected: 1,
      },
      {
        input: [[0, 0, 0, 0]],
        expected: 0,
      },
      {
        input: [[-1, -2, -3, -4]],
        expected: -4,
      },
    ],
    solution: `# Optimal: Sort and sum alternating elements - O(n log n) time, O(1) space
def array_pair_sum(nums):
    nums.sort()
    return sum(nums[::2])  # Sum elements at even indices

# Expanded version
def array_pair_sum_verbose(nums):
    nums.sort()
    result = 0
    for i in range(0, len(nums), 2):
        result += nums[i]
    return result
`,
    timeComplexity: 'O(n log n) - dominated by sorting',
    spaceComplexity: 'O(1) or O(log n) depending on sorting algorithm',
    leetcodeUrl: 'https://leetcode.com/problems/array-partition/',
  },

  // HARD - Valid Number
  {
    id: 'valid-number',
    title: 'Valid Number',
    difficulty: 'Hard',
    topic: 'Time & Space Complexity',

    leetcodeUrl: 'https://leetcode.com/problems/valid-number/',
    youtubeUrl: 'https://www.youtube.com/watch?v=QfRSeibcugw',
    order: 6,
    description: `A **valid number** can be split up into these components (in order):

1. A **decimal number** or an **integer**.
2. (Optional) An \`'e'\` or \`'E'\`, followed by an **integer**.

A **decimal number** can be split up into these components (in order):
1. (Optional) A sign character (either \`'+'\` or \`'-'\`).
2. One of the following formats:
   - One or more digits, followed by a dot \`'.'\`.
   - One or more digits, followed by a dot \`'.'\`, followed by one or more digits.
   - A dot \`'.'\`, followed by one or more digits.

An **integer** can be split up into these components (in order):
1. (Optional) A sign character (either \`'+'\` or \`'-'\`).
2. One or more digits.

Given a string \`s\`, return \`true\` if \`s\` is a **valid number**.

**Complexity Focus:** This problem tests your ability to implement a state machine with O(n) time and O(1) space.`,
    examples: [
      {
        input: 's = "0"',
        output: 'true',
      },
      {
        input: 's = "e"',
        output: 'false',
      },
      {
        input: 's = "."',
        output: 'false',
      },
      {
        input: 's = "0.1"',
        output: 'true',
      },
    ],
    constraints: [
      '1 <= s.length <= 20',
      "s consists of only English letters (both uppercase and lowercase), digits (0-9), plus '+', minus '-', or dot '.'.",
    ],
    hints: [
      'Use a finite state machine',
      'Track: seen digit, seen dot, seen exponent',
      'Process character by character - O(n) time',
    ],
    starterCode: `def is_number(s: str) -> bool:
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['0'],
        expected: true,
      },
      {
        input: ['e'],
        expected: false,
      },
      {
        input: ['.'],
        expected: false,
      },
      {
        input: ['0.1'],
        expected: true,
      },
      {
        input: ['2e10'],
        expected: true,
      },
      {
        input: ['-90e3'],
        expected: true,
      },
      {
        input: ['1e'],
        expected: false,
      },
      {
        input: ['e3'],
        expected: false,
      },
      {
        input: ['6e-1'],
        expected: true,
      },
      {
        input: ['.1'],
        expected: true,
      },
    ],
    solution: `# State Machine - O(n) time, O(1) space
def is_number(s):
    seen_digit = False
    seen_exponent = False
    seen_dot = False
    
    for i, char in enumerate(s):
        if char.isdigit():
            seen_digit = True
        elif char in ['+', '-']:
            # Sign must be at start or right after exponent
            if i > 0 and s[i-1] not in ['e', 'E']:
                return False
        elif char in ['e', 'E']:
            # Can't have two exponents, must have seen digit before
            if seen_exponent or not seen_digit:
                return False
            seen_exponent = True
            seen_digit = False  # Must have digit after exponent
        elif char == '.':
            # Can't have two dots or dot after exponent
            if seen_dot or seen_exponent:
                return False
            seen_dot = True
        else:
            # Invalid character
            return False
    
    return seen_digit

# Alternative: Try parsing with try-except (simpler but less educational)
def is_number_simple(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
`,
    timeComplexity: 'O(n) - single pass through the string',
    spaceComplexity: 'O(1) - only tracking a few boolean flags',
    leetcodeUrl: 'https://leetcode.com/problems/valid-number/',
  },
];
