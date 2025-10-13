/**
 * Arrays & Hashing problems
 */

import { Problem } from '../types';

export const arraysHashingProblems: Problem[] = [
  {
    id: 'contains-duplicate',
    title: 'Contains Duplicate',
    difficulty: 'Easy',
    description: `Given an integer array \`nums\`, return \`true\` if any value appears **at least twice** in the array, and return \`false\` if every element is distinct.


**Approach:**
Use a hash set to track seen numbers. If you encounter a number already in the set, you've found a duplicate.`,
    examples: [
      {
        input: 'nums = [1,2,3,1]',
        output: 'true',
        explanation: 'The number 1 appears twice.',
      },
      {
        input: 'nums = [1,2,3,4]',
        output: 'false',
        explanation: 'All elements are distinct.',
      },
      {
        input: 'nums = [1,1,1,3,3,4,3,2,4,2]',
        output: 'true',
        explanation: 'Multiple numbers appear more than once.',
      },
    ],
    constraints: ['1 <= nums.length <= 10^5', '-10^9 <= nums[i] <= 10^9'],
    hints: [
      "Use a hash set to track numbers you've seen",
      'If you find a number already in the set, return true immediately',
      'Time complexity should be O(n), space complexity O(n)',
    ],
    starterCode: `from typing import List

def contains_duplicate(nums: List[int]) -> bool:
    """
    Determine if array contains any duplicates.
    
    Args:
        nums: List of integers
        
    Returns:
        True if any value appears at least twice, False otherwise
    """
    # Your code here
    pass
`,
    testCases: [
      {
        input: [[1, 2, 3, 1]],
        expected: true,
      },
      {
        input: [[1, 2, 3, 4]],
        expected: false,
      },
      {
        input: [[1, 1, 1, 3, 3, 4, 3, 2, 4, 2]],
        expected: true,
      },
      {
        input: [[1]],
        expected: false,
      },
    ],
    solution: `def contains_duplicate(nums: List[int]) -> bool:
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False

# Alternative: Use set length comparison
def contains_duplicate_alt(nums: List[int]) -> bool:
    return len(nums) != len(set(nums))`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 1,
    topic: 'Arrays & Hashing',
    leetcodeUrl: 'https://leetcode.com/problems/contains-duplicate/',
    youtubeUrl: 'https://www.youtube.com/watch?v=3OamzN90kPg',
  },
  {
    id: 'two-sum',
    title: 'Two Sum',
    difficulty: 'Medium',
    description: `Given an array of integers \`nums\` and an integer \`target\`, return **indices** of the two numbers such that they add up to \`target\`.

You may assume that each input would have **exactly one solution**, and you may not use the same element twice.

You can return the answer in any order.


**Approach:**
Use a hash map to store each number and its index. For each number, check if its complement (target - num) exists in the map.`,
    examples: [
      {
        input: 'nums = [2,7,11,15], target = 9',
        output: '[0,1]',
        explanation: 'Because nums[0] + nums[1] == 9, we return [0, 1].',
      },
      {
        input: 'nums = [3,2,4], target = 6',
        output: '[1,2]',
        explanation: 'nums[1] + nums[2] == 6',
      },
      {
        input: 'nums = [3,3], target = 6',
        output: '[0,1]',
        explanation: 'Both elements sum to the target.',
      },
    ],
    constraints: [
      '2 <= nums.length <= 10^4',
      '-10^9 <= nums[i] <= 10^9',
      '-10^9 <= target <= 10^9',
      'Only one valid answer exists',
    ],
    hints: [
      'For each number, calculate what its complement should be (target - num)',
      'Use a hash map to check if the complement exists in O(1) time',
      'Store both the number and its index in the hash map',
      "Don't use the same element twice",
    ],
    starterCode: `from typing import List

def two_sum(nums: List[int], target: int) -> List[int]:
    """
    Find indices of two numbers that add up to target.
    
    Args:
        nums: List of integers
        target: Target sum
        
    Returns:
        List of two indices [i, j] where nums[i] + nums[j] == target
    """
    # Your code here
    pass
`,
    testCases: [
      {
        input: [[2, 7, 11, 15], 9],
        expected: [0, 1],
      },
      {
        input: [[3, 2, 4], 6],
        expected: [1, 2],
      },
      {
        input: [[3, 3], 6],
        expected: [0, 1],
      },
      {
        input: [[1, 2, 3, 4, 5], 9],
        expected: [3, 4],
      },
    ],
    solution: `def two_sum(nums: List[int], target: int) -> List[int]:
    seen = {}  # value -> index
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    
    return []  # Should never reach if input is valid`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 2,
    topic: 'Arrays & Hashing',
    leetcodeUrl: 'https://leetcode.com/problems/two-sum/',
    youtubeUrl: 'https://www.youtube.com/watch?v=KLlXCFG5TnA',
  },
  {
    id: 'group-anagrams',
    title: 'Group Anagrams',
    difficulty: 'Hard',
    description: `Given an array of strings \`strs\`, group **the anagrams** together. You can return the answer in **any order**.

An **Anagram** is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.


**Approach:**
Use a hash map where the key is a signature of the anagram (e.g., sorted string or character count), and the value is a list of strings matching that signature.`,
    examples: [
      {
        input: 'strs = ["eat","tea","tan","ate","nat","bat"]',
        output: '[["bat"],["nat","tan"],["ate","eat","tea"]]',
        explanation: 'All anagrams are grouped together.',
      },
      {
        input: 'strs = [""]',
        output: '[[""]]',
        explanation: 'Single empty string.',
      },
      {
        input: 'strs = ["a"]',
        output: '[["a"]]',
        explanation: 'Single character.',
      },
    ],
    constraints: [
      '1 <= strs.length <= 10^4',
      '0 <= strs[i].length <= 100',
      'strs[i] consists of lowercase English letters',
    ],
    hints: [
      'Anagrams have the same characters, just in different order',
      'Use sorted string as a key: sorted("eat") == sorted("tea") == "aet"',
      'Alternative: Use character count as key (more efficient)',
      'Use defaultdict(list) to group strings by their signature',
    ],
    starterCode: `from typing import List

def group_anagrams(strs: List[str]) -> List[List[str]]:
    """
    Group strings that are anagrams of each other.
    
    Args:
        strs: List of strings
        
    Returns:
        List of groups, where each group contains anagram strings
    """
    # Your code here
    pass
`,
    testCases: [
      {
        input: [['eat', 'tea', 'tan', 'ate', 'nat', 'bat']],
        expected: [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']],
      },
      {
        input: [['']],
        expected: [['']],
      },
      {
        input: [['a']],
        expected: [['a']],
      },
      {
        input: [['abc', 'bca', 'cab', 'xyz', 'zyx', 'yxz']],
        expected: [
          ['abc', 'bca', 'cab'],
          ['xyz', 'zyx', 'yxz'],
        ],
      },
    ],
    solution: `def group_anagrams(strs: List[str]) -> List[List[str]]:
    from collections import defaultdict
    
    groups = defaultdict(list)
    
    for s in strs:
        # Use sorted string as key
        key = ''.join(sorted(s))
        groups[key].append(s)
    
    return list(groups.values())

# Alternative: Use character count (faster for long strings)
def group_anagrams_count(strs: List[str]) -> List[List[str]]:
    from collections import defaultdict
    
    groups = defaultdict(list)
    
    for s in strs:
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1
        key = tuple(count)
        groups[key].append(s)
    
    return list(groups.values())`,
    timeComplexity:
      'O(n * k log k) where n is number of strings and k is max length',
    spaceComplexity: 'O(n * k)',
    order: 3,
    topic: 'Arrays & Hashing',
    leetcodeUrl: 'https://leetcode.com/problems/group-anagrams/',
    youtubeUrl: 'https://www.youtube.com/watch?v=vzdNOK2oB2E',
  },
  // EASY - Valid Anagram
  {
    id: 'valid-anagram',
    title: 'Valid Anagram',
    difficulty: 'Easy',
    topic: 'Arrays & Hashing',
    order: 4,
    description: `Given two strings \`s\` and \`t\`, return \`true\` if \`t\` is an anagram of \`s\`, and \`false\` otherwise.

An **Anagram** is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.`,
    examples: [
      {
        input: 's = "anagram", t = "nagaram"',
        output: 'true',
      },
      {
        input: 's = "rat", t = "car"',
        output: 'false',
      },
    ],
    constraints: [
      '1 <= s.length, t.length <= 5 * 10^4',
      's and t consist of lowercase English letters',
    ],
    hints: [
      'Sort both strings and compare',
      'Or use a hash map to count character frequencies',
      'Both strings must have same length to be anagrams',
    ],
    starterCode: `def is_anagram(s: str, t: str) -> bool:
    """
    Check if t is an anagram of s.
    
    Args:
        s: First string
        t: Second string
        
    Returns:
        True if t is an anagram of s
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['anagram', 'nagaram'],
        expected: true,
      },
      {
        input: ['rat', 'car'],
        expected: false,
      },
    ],
    solution: `def is_anagram(s: str, t: str) -> bool:
    """
    Hash map approach to count frequencies.
    Time: O(n), Space: O(1) - at most 26 letters
    """
    if len(s) != len(t):
        return False
    
    count = {}
    
    # Count characters in s
    for c in s:
        count[c] = count.get(c, 0) + 1
    
    # Decrement for characters in t
    for c in t:
        if c not in count:
            return False
        count[c] -= 1
        if count[c] < 0:
            return False
    
    return True

# Alternative: Sorting approach
def is_anagram_sort(s: str, t: str) -> bool:
    """
    Sort both strings and compare.
    Time: O(n log n), Space: O(1)
    """
    return sorted(s) == sorted(t)
`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/valid-anagram/',
    youtubeUrl: 'https://www.youtube.com/watch?v=9UtInBqnCgA',
  },
  // EASY - Majority Element
  {
    id: 'majority-element',
    title: 'Majority Element',
    difficulty: 'Easy',
    topic: 'Arrays & Hashing',
    order: 5,
    description: `Given an array \`nums\` of size \`n\`, return the **majority element**.

The majority element is the element that appears **more than ⌊n / 2⌋ times**. You may assume that the majority element always exists in the array.`,
    examples: [
      {
        input: 'nums = [3,2,3]',
        output: '3',
      },
      {
        input: 'nums = [2,2,1,1,1,2,2]',
        output: '2',
      },
    ],
    constraints: [
      'n == nums.length',
      '1 <= n <= 5 * 10^4',
      '-10^9 <= nums[i] <= 10^9',
    ],
    hints: [
      'Use a hash map to count frequencies',
      'Return the element with count > n/2',
      'Can you solve it in O(1) space? (Boyer-Moore Voting)',
    ],
    starterCode: `from typing import List

def majority_element(nums: List[int]) -> int:
    """
    Find the majority element.
    
    Args:
        nums: Array of integers
        
    Returns:
        The majority element
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[3, 2, 3]],
        expected: 3,
      },
      {
        input: [[2, 2, 1, 1, 1, 2, 2]],
        expected: 2,
      },
      {
        input: [[1]],
        expected: 1,
      },
    ],
    solution: `from typing import List

def majority_element(nums: List[int]) -> int:
    """
    Hash map approach to count frequencies.
    Time: O(n), Space: O(n)
    """
    count = {}
    for num in nums:
        count[num] = count.get(num, 0) + 1
        if count[num] > len(nums) // 2:
            return num
    return -1

# Alternative: Boyer-Moore Voting Algorithm
def majority_element_voting(nums: List[int]) -> int:
    """
    Boyer-Moore Voting: O(n) time, O(1) space.
    """
    candidate = None
    count = 0
    
    # Find candidate
    for num in nums:
        if count == 0:
            candidate = num
        count += 1 if num == candidate else -1
    
    return candidate
`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n) for hash map, O(1) for voting',
    leetcodeUrl: 'https://leetcode.com/problems/majority-element/',
    youtubeUrl: 'https://www.youtube.com/watch?v=7pnhv842keE',
  },
  // EASY - Intersection of Two Arrays
  {
    id: 'intersection-two-arrays',
    title: 'Intersection of Two Arrays',
    difficulty: 'Easy',
    topic: 'Arrays & Hashing',
    order: 6,
    description: `Given two integer arrays \`nums1\` and \`nums2\`, return an array of their intersection. Each element in the result must be **unique** and you may return the result in **any order**.`,
    examples: [
      {
        input: 'nums1 = [1,2,2,1], nums2 = [2,2]',
        output: '[2]',
      },
      {
        input: 'nums1 = [4,9,5], nums2 = [9,4,9,8,4]',
        output: '[9,4]',
        explanation: '[4,9] is also accepted.',
      },
    ],
    constraints: [
      '1 <= nums1.length, nums2.length <= 1000',
      '0 <= nums1[i], nums2[i] <= 1000',
    ],
    hints: [
      'Use a set to store unique elements from first array',
      'Check which elements from second array exist in the set',
      'Return the intersection as a list',
    ],
    starterCode: `from typing import List

def intersection(nums1: List[int], nums2: List[int]) -> List[int]:
    """
    Find intersection of two arrays.
    
    Args:
        nums1: First array
        nums2: Second array
        
    Returns:
        Array of unique intersection elements
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [1, 2, 2, 1],
          [2, 2],
        ],
        expected: [2],
      },
      {
        input: [
          [4, 9, 5],
          [9, 4, 9, 8, 4],
        ],
        expected: [9, 4],
      },
    ],
    solution: `from typing import List

def intersection(nums1: List[int], nums2: List[int]) -> List[int]:
    """
    Set intersection approach.
    Time: O(n + m), Space: O(n)
    """
    set1 = set(nums1)
    result = set()
    
    for num in nums2:
        if num in set1:
            result.add(num)
    
    return list(result)

# Alternative: Built-in set operations
def intersection_builtin(nums1: List[int], nums2: List[int]) -> List[int]:
    """
    Using Python set intersection.
    Time: O(n + m), Space: O(n + m)
    """
    return list(set(nums1) & set(nums2))
`,
    timeComplexity: 'O(n + m)',
    spaceComplexity: 'O(min(n, m))',
    leetcodeUrl: 'https://leetcode.com/problems/intersection-of-two-arrays/',
    youtubeUrl: 'https://www.youtube.com/watch?v=Yz4V1RdPJx8',
  },
  // EASY - Concatenation of Array
  {
    id: 'concatenation-of-array',
    title: 'Concatenation of Array',
    difficulty: 'Easy',
    topic: 'Arrays & Hashing',
    order: 7,
    description: `Given an integer array \`nums\` of length \`n\`, you want to create an array \`ans\` of length \`2n\` where \`ans[i] == nums[i]\` and \`ans[i + n] == nums[i]\` for \`0 <= i < n\` (**0-indexed**).

Specifically, \`ans\` is the **concatenation** of two \`nums\` arrays.

Return the array \`ans\`.`,
    examples: [
      {
        input: 'nums = [1,2,1]',
        output: '[1,2,1,1,2,1]',
        explanation:
          'The array ans is formed as follows: ans = [nums[0],nums[1],nums[2],nums[0],nums[1],nums[2]] = [1,2,1,1,2,1]',
      },
      {
        input: 'nums = [1,3,2,1]',
        output: '[1,3,2,1,1,3,2,1]',
      },
    ],
    constraints: ['n == nums.length', '1 <= n <= 1000', '1 <= nums[i] <= 1000'],
    hints: [
      'Create a new array of size 2n',
      'Copy elements from nums twice',
      'Or use array concatenation built-ins',
    ],
    starterCode: `from typing import List

def get_concatenation(nums: List[int]) -> List[int]:
    """
    Concatenate array with itself.
    
    Args:
        nums: Input array
        
    Returns:
        Array concatenated with itself
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 2, 1]],
        expected: [1, 2, 1, 1, 2, 1],
      },
      {
        input: [[1, 3, 2, 1]],
        expected: [1, 3, 2, 1, 1, 3, 2, 1],
      },
      {
        input: [[1]],
        expected: [1, 1],
      },
    ],
    solution: `from typing import List

def get_concatenation(nums: List[int]) -> List[int]:
    """
    Simple concatenation approach.
    Time: O(n), Space: O(n)
    """
    return nums + nums

# Alternative: Manual approach
def get_concatenation_manual(nums: List[int]) -> List[int]:
    """
    Manually build result array.
    Time: O(n), Space: O(n)
    """
    n = len(nums)
    ans = [0] * (2 * n)
    
    for i in range(n):
        ans[i] = nums[i]
        ans[i + n] = nums[i]
    
    return ans

# Alternative: List comprehension
def get_concatenation_comprehension(nums: List[int]) -> List[int]:
    """
    Using list comprehension.
    Time: O(n), Space: O(n)
    """
    return [nums[i % len(nums)] for i in range(2 * len(nums))]
`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    leetcodeUrl: 'https://leetcode.com/problems/concatenation-of-array/',
    youtubeUrl: 'https://www.youtube.com/watch?v=68a1Dc_qVq4',
  },
  // EASY - Find All Numbers Disappeared in an Array
  {
    id: 'find-disappeared-numbers',
    title: 'Find All Numbers Disappeared in an Array',
    difficulty: 'Easy',
    topic: 'Arrays & Hashing',
    order: 8,
    description: `Given an array \`nums\` of \`n\` integers where \`nums[i]\` is in the range \`[1, n]\`, return an array of all the integers in the range \`[1, n]\` that do not appear in \`nums\`.`,
    examples: [
      {
        input: 'nums = [4,3,2,7,8,2,3,1]',
        output: '[5,6]',
      },
      {
        input: 'nums = [1,1]',
        output: '[2]',
      },
    ],
    constraints: ['n == nums.length', '1 <= n <= 10^5', '1 <= nums[i] <= n'],
    hints: [
      'Use a set to track which numbers appear',
      'Then check which numbers from [1, n] are missing',
      'Can you do it without extra space by marking in-place?',
    ],
    starterCode: `from typing import List

def find_disappeared_numbers(nums: List[int]) -> List[int]:
    """
    Find all numbers that disappeared.
    
    Args:
        nums: Array with some numbers in [1, n]
        
    Returns:
        List of missing numbers
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[4, 3, 2, 7, 8, 2, 3, 1]],
        expected: [5, 6],
      },
      {
        input: [[1, 1]],
        expected: [2],
      },
      {
        input: [[1, 2, 3, 4, 5]],
        expected: [],
      },
    ],
    solution: `from typing import List

def find_disappeared_numbers(nums: List[int]) -> List[int]:
    """
    Set approach to find missing numbers.
    Time: O(n), Space: O(n)
    """
    num_set = set(nums)
    result = []
    
    for i in range(1, len(nums) + 1):
        if i not in num_set:
            result.append(i)
    
    return result

# Alternative: In-place marking
def find_disappeared_numbers_inplace(nums: List[int]) -> List[int]:
    """
    Mark presence by negating values at indices.
    Time: O(n), Space: O(1)
    """
    # Mark presence by negating value at index
    for num in nums:
        index = abs(num) - 1
        if nums[index] > 0:
            nums[index] = -nums[index]
    
    # Positive indices indicate missing numbers
    result = []
    for i in range(len(nums)):
        if nums[i] > 0:
            result.append(i + 1)
    
    return result
`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n) for set, O(1) for in-place',
    leetcodeUrl:
      'https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/',
    youtubeUrl: 'https://www.youtube.com/watch?v=8i-f24YFWC4',
  },
  // EASY - Plus One
  {
    id: 'plus-one',
    title: 'Plus One',
    difficulty: 'Easy',
    topic: 'Arrays & Hashing',
    order: 9,
    description: `You are given a **large integer** represented as an integer array \`digits\`, where each \`digits[i]\` is the \`ith\` digit of the integer. The digits are ordered from most significant to least significant in left-to-right order. The large integer does not contain any leading \`0\`'s.

Increment the large integer by one and return the resulting array of digits.`,
    examples: [
      {
        input: 'digits = [1,2,3]',
        output: '[1,2,4]',
        explanation:
          'The array represents the integer 123. Incrementing by one gives 123 + 1 = 124.',
      },
      {
        input: 'digits = [9,9,9]',
        output: '[1,0,0,0]',
        explanation:
          'The array represents the integer 999. Incrementing by one gives 999 + 1 = 1000.',
      },
    ],
    constraints: [
      '1 <= digits.length <= 100',
      '0 <= digits[i] <= 9',
      'digits does not contain any leading 0s',
    ],
    hints: [
      'Start from the rightmost digit',
      'Handle carry when digit is 9',
      'If all digits are 9, you need to add a 1 at the beginning',
    ],
    starterCode: `from typing import List

def plus_one(digits: List[int]) -> List[int]:
    """
    Add one to the number represented by the array.
    
    Args:
        digits: Array representing a large integer
        
    Returns:
        Array representing the incremented number
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 2, 3]],
        expected: [1, 2, 4],
      },
      {
        input: [[9, 9, 9]],
        expected: [1, 0, 0, 0],
      },
      {
        input: [[4, 3, 2, 1]],
        expected: [4, 3, 2, 2],
      },
    ],
    solution: `from typing import List

def plus_one(digits: List[int]) -> List[int]:
    """
    Handle carry from right to left.
    Time: O(n), Space: O(1) or O(n) if all 9s
    """
    n = len(digits)
    
    # Start from rightmost digit
    for i in range(n - 1, -1, -1):
        # If not 9, just increment and return
        if digits[i] < 9:
            digits[i] += 1
            return digits
        
        # If 9, set to 0 and continue carry
        digits[i] = 0
    
    # If we're here, all digits were 9
    # Need to add 1 at the beginning
    return [1] + digits
`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1) or O(n) if all 9s',
    leetcodeUrl: 'https://leetcode.com/problems/plus-one/',
    youtubeUrl: 'https://www.youtube.com/watch?v=jIaA8boiG1s',
  },
  // EASY - Pascal's Triangle
  {
    id: 'pascals-triangle',
    title: "Pascal's Triangle",
    difficulty: 'Easy',
    topic: 'Arrays & Hashing',
    order: 10,
    description: `Given an integer \`numRows\`, return the first numRows of **Pascal's triangle**.

In Pascal's triangle, each number is the sum of the two numbers directly above it.`,
    examples: [
      {
        input: 'numRows = 5',
        output: '[[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]',
      },
      {
        input: 'numRows = 1',
        output: '[[1]]',
      },
    ],
    constraints: ['1 <= numRows <= 30'],
    hints: [
      'First and last elements of each row are always 1',
      'Each middle element is the sum of two elements from the previous row',
      'Build row by row',
    ],
    starterCode: `from typing import List

def generate_pascals_triangle(num_rows: int) -> List[List[int]]:
    """
    Generate Pascal's Triangle.
    
    Args:
        num_rows: Number of rows to generate
        
    Returns:
        Pascal's Triangle as a 2D list
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [5],
        expected: [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]],
      },
      {
        input: [1],
        expected: [[1]],
      },
      {
        input: [3],
        expected: [[1], [1, 1], [1, 2, 1]],
      },
    ],
    solution: `from typing import List

def generate_pascals_triangle(num_rows: int) -> List[List[int]]:
    """
    Build triangle row by row.
    Time: O(numRows^2), Space: O(1) excluding output
    """
    triangle = []
    
    for row_num in range(num_rows):
        # Start with all 1s
        row = [1] * (row_num + 1)
        
        # Fill in middle values
        for j in range(1, row_num):
            row[j] = triangle[row_num - 1][j - 1] + triangle[row_num - 1][j]
        
        triangle.append(row)
    
    return triangle
`,
    timeComplexity: 'O(numRows^2)',
    spaceComplexity: 'O(1) excluding output',
    leetcodeUrl: 'https://leetcode.com/problems/pascals-triangle/',
    youtubeUrl: 'https://www.youtube.com/watch?v=nPVEaB3AjUM',
  },
  // EASY - Sort Array By Parity
  {
    id: 'sort-array-by-parity',
    title: 'Sort Array By Parity',
    difficulty: 'Easy',
    topic: 'Arrays & Hashing',
    order: 11,
    description: `Given an integer array \`nums\`, move all the even integers at the beginning of the array followed by all the odd integers.

Return **any array** that satisfies this condition.`,
    examples: [
      {
        input: 'nums = [3,1,2,4]',
        output: '[2,4,3,1]',
        explanation:
          '[4,2,3,1], [2,4,1,3], and [4,2,1,3] would also be accepted.',
      },
      {
        input: 'nums = [0]',
        output: '[0]',
      },
    ],
    constraints: ['1 <= nums.length <= 5000', '0 <= nums[i] <= 5000'],
    hints: [
      'Use two pointers approach',
      'One pointer for even position, one for odd',
      'Swap when necessary',
    ],
    starterCode: `from typing import List

def sort_array_by_parity(nums: List[int]) -> List[int]:
    """
    Sort array so evens come before odds.
    
    Args:
        nums: Array of integers
        
    Returns:
        Array with evens first, then odds
    """
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
    ],
    solution: `from typing import List

def sort_array_by_parity(nums: List[int]) -> List[int]:
    """
    Two-pointer in-place sort.
    Time: O(n), Space: O(1)
    """
    left = 0
    right = len(nums) - 1
    
    while left < right:
        # If left is odd and right is even, swap
        if nums[left] % 2 > nums[right] % 2:
            nums[left], nums[right] = nums[right], nums[left]
        
        # Move left pointer if even
        if nums[left] % 2 == 0:
            left += 1
        
        # Move right pointer if odd
        if nums[right] % 2 == 1:
            right -= 1
    
    return nums
`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/sort-array-by-parity/',
    youtubeUrl: 'https://www.youtube.com/watch?v=6YZn-z5jkrg',
  },
  // EASY - Find Common Characters
  {
    id: 'find-common-characters',
    title: 'Find Common Characters',
    difficulty: 'Easy',
    topic: 'Arrays & Hashing',
    order: 12,
    description: `Given a string array \`words\`, return an array of all characters that show up in all strings within the \`words\` (including duplicates). You may return the answer in **any order**.`,
    examples: [
      {
        input: 'words = ["bella","label","roller"]',
        output: '["e","l","l"]',
      },
      {
        input: 'words = ["cool","lock","cook"]',
        output: '["c","o"]',
      },
    ],
    constraints: [
      '1 <= words.length <= 100',
      '1 <= words[i].length <= 100',
      'words[i] consists of lowercase English letters',
    ],
    hints: [
      'Count character frequencies for each word',
      'Find the minimum frequency for each character across all words',
      'Add characters to result based on minimum frequency',
    ],
    starterCode: `from typing import List

def common_chars(words: List[str]) -> List[str]:
    """
    Find characters common to all words.
    
    Args:
        words: List of words
        
    Returns:
        List of common characters (with duplicates)
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [['bella', 'label', 'roller']],
        expected: ['e', 'l', 'l'],
      },
      {
        input: [['cool', 'lock', 'cook']],
        expected: ['c', 'o'],
      },
    ],
    solution: `from typing import List
from collections import Counter

def common_chars(words: List[str]) -> List[str]:
    """
    Find minimum frequency of each character.
    Time: O(n * k) where n = number of words, k = avg length
    Space: O(1) - at most 26 letters
    """
    # Start with frequency count of first word
    common_count = Counter(words[0])
    
    # For each subsequent word, keep minimum counts
    for word in words[1:]:
        word_count = Counter(word)
        
        # Update common_count to minimum frequencies
        for char in common_count:
            common_count[char] = min(common_count[char], word_count.get(char, 0))
    
    # Build result from common_count
    result = []
    for char, count in common_count.items():
        result.extend([char] * count)
    
    return result
`,
    timeComplexity: 'O(n * k)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/find-common-characters/',
    youtubeUrl: 'https://www.youtube.com/watch?v=kolXfMZ4kZY',
  },
  // EASY - Rank Transform of an Array
  {
    id: 'rank-transform-of-array',
    title: 'Rank Transform of an Array',
    difficulty: 'Easy',
    topic: 'Arrays & Hashing',
    order: 13,
    description: `Given an array of integers \`arr\`, replace each element with its rank.

The rank represents how large the element is. The rank has the following rules:
- Rank is an integer starting from 1.
- The larger the element, the larger the rank. If two elements are equal, their rank must be the same.
- Rank should be as small as possible.`,
    examples: [
      {
        input: 'arr = [40,10,20,30]',
        output: '[4,1,2,3]',
        explanation:
          '40 is the largest element. 10 is the smallest. 20 is the second smallest. 30 is the third smallest.',
      },
      {
        input: 'arr = [100,100,100]',
        output: '[1,1,1]',
        explanation: 'Same elements share the same rank.',
      },
    ],
    constraints: ['0 <= arr.length <= 10^5', '-10^9 <= arr[i] <= 10^9'],
    hints: [
      'Sort the unique values',
      'Create a mapping from value to rank',
      'Map each element to its rank',
    ],
    starterCode: `from typing import List

def array_rank_transform(arr: List[int]) -> List[int]:
    """
    Transform array to ranks.
    
    Args:
        arr: Array of integers
        
    Returns:
        Array where each element is replaced by its rank
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[40, 10, 20, 30]],
        expected: [4, 1, 2, 3],
      },
      {
        input: [[100, 100, 100]],
        expected: [1, 1, 1],
      },
      {
        input: [[37, 12, 28, 9, 100, 56, 80, 5, 12]],
        expected: [5, 3, 4, 2, 8, 6, 7, 1, 3],
      },
    ],
    solution: `from typing import List

def array_rank_transform(arr: List[int]) -> List[int]:
    """
    Sort and create rank mapping.
    Time: O(n log n), Space: O(n)
    """
    # Create mapping from value to rank
    rank = {}
    for value in sorted(set(arr)):
        rank[value] = len(rank) + 1
    
    # Map each element to its rank
    return [rank[value] for value in arr]
`,
    timeComplexity: 'O(n log n)',
    spaceComplexity: 'O(n)',
    leetcodeUrl: 'https://leetcode.com/problems/rank-transform-of-an-array/',
    youtubeUrl: 'https://www.youtube.com/watch?v=yccJ_V7Q7DA',
  },
  // MEDIUM - Product of Array Except Self
  {
    id: 'product-except-self',
    title: 'Product of Array Except Self',
    difficulty: 'Medium',
    topic: 'Arrays & Hashing',
    order: 14,
    description: `Given an integer array \`nums\`, return an array \`answer\` such that \`answer[i]\` is equal to the product of all the elements of \`nums\` except \`nums[i]\`.

The product of any prefix or suffix of \`nums\` is **guaranteed** to fit in a **32-bit** integer.

You must write an algorithm that runs in **O(n)** time and without using the division operation.`,
    examples: [
      {
        input: 'nums = [1,2,3,4]',
        output: '[24,12,8,6]',
      },
      {
        input: 'nums = [-1,1,0,-3,3]',
        output: '[0,0,9,0,0]',
      },
    ],
    constraints: ['2 <= nums.length <= 10^5', '-30 <= nums[i] <= 30'],
    hints: [
      'Use prefix and suffix products',
      'First pass: calculate prefix products',
      'Second pass: calculate suffix products and combine',
      'Can you do it with O(1) extra space?',
    ],
    starterCode: `from typing import List

def product_except_self(nums: List[int]) -> List[int]:
    """
    Calculate product of array except self.
    
    Args:
        nums: Array of integers
        
    Returns:
        Array where each element is product of all others
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 2, 3, 4]],
        expected: [24, 12, 8, 6],
      },
      {
        input: [[-1, 1, 0, -3, 3]],
        expected: [0, 0, 9, 0, 0],
      },
    ],
    solution: `from typing import List

def product_except_self(nums: List[int]) -> List[int]:
    """
    Two-pass with prefix and suffix products.
    Time: O(n), Space: O(1) excluding output
    """
    n = len(nums)
    result = [1] * n
    
    # First pass: prefix products
    prefix = 1
    for i in range(n):
        result[i] = prefix
        prefix *= nums[i]
    
    # Second pass: suffix products
    suffix = 1
    for i in range(n - 1, -1, -1):
        result[i] *= suffix
        suffix *= nums[i]
    
    return result
`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1) excluding output',
    leetcodeUrl: 'https://leetcode.com/problems/product-of-array-except-self/',
    youtubeUrl: 'https://www.youtube.com/watch?v=bNvIQI2wAjk',
  },
  // MEDIUM - Subarray Sum Equals K
  {
    id: 'subarray-sum-equals-k',
    title: 'Subarray Sum Equals K',
    difficulty: 'Medium',
    topic: 'Arrays & Hashing',
    order: 15,
    description: `Given an array of integers \`nums\` and an integer \`k\`, return the total number of subarrays whose sum equals to \`k\`.

A subarray is a contiguous **non-empty** sequence of elements within an array.`,
    examples: [
      {
        input: 'nums = [1,1,1], k = 2',
        output: '2',
      },
      {
        input: 'nums = [1,2,3], k = 3',
        output: '2',
      },
    ],
    constraints: [
      '1 <= nums.length <= 2 * 10^4',
      '-1000 <= nums[i] <= 1000',
      '-10^7 <= k <= 10^7',
    ],
    hints: [
      'Use prefix sum with hash map',
      'Store frequency of prefix sums',
      'If (current_sum - k) exists in map, we found valid subarrays',
    ],
    starterCode: `from typing import List

def subarray_sum(nums: List[int], k: int) -> int:
    """
    Count subarrays with sum equal to k.
    
    Args:
        nums: Array of integers
        k: Target sum
        
    Returns:
        Number of subarrays with sum k
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 1, 1], 2],
        expected: 2,
      },
      {
        input: [[1, 2, 3], 3],
        expected: 2,
      },
    ],
    solution: `from typing import List

def subarray_sum(nums: List[int], k: int) -> int:
    """
    Prefix sum with hash map.
    Time: O(n), Space: O(n)
    """
    count = 0
    prefix_sum = 0
    sum_freq = {0: 1}  # Base case: empty prefix
    
    for num in nums:
        prefix_sum += num
        
        # If (prefix_sum - k) exists, we found subarrays
        if prefix_sum - k in sum_freq:
            count += sum_freq[prefix_sum - k]
        
        # Add current prefix_sum to map
        sum_freq[prefix_sum] = sum_freq.get(prefix_sum, 0) + 1
    
    return count
`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    leetcodeUrl: 'https://leetcode.com/problems/subarray-sum-equals-k/',
    youtubeUrl: 'https://www.youtube.com/watch?v=fFVZt-6sgyo',
  },
  // MEDIUM - Longest Consecutive Sequence
  {
    id: 'longest-consecutive-sequence',
    title: 'Longest Consecutive Sequence',
    difficulty: 'Medium',
    topic: 'Arrays & Hashing',
    order: 16,
    description: `Given an unsorted array of integers \`nums\`, return the length of the longest consecutive elements sequence.

You must write an algorithm that runs in **O(n)** time.`,
    examples: [
      {
        input: 'nums = [100,4,200,1,3,2]',
        output: '4',
        explanation:
          'The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.',
      },
      {
        input: 'nums = [0,3,7,2,5,8,4,6,0,1]',
        output: '9',
      },
    ],
    constraints: ['0 <= nums.length <= 10^5', '-10^9 <= nums[i] <= 10^9'],
    hints: [
      'Use a hash set for O(1) lookups',
      'Only start counting from the beginning of a sequence',
      'Check if num-1 exists to know if it is a sequence start',
    ],
    starterCode: `from typing import List

def longest_consecutive(nums: List[int]) -> int:
    """
    Find longest consecutive sequence length.
    
    Args:
        nums: Array of integers
        
    Returns:
        Length of longest consecutive sequence
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[100, 4, 200, 1, 3, 2]],
        expected: 4,
      },
      {
        input: [[0, 3, 7, 2, 5, 8, 4, 6, 0, 1]],
        expected: 9,
      },
    ],
    solution: `from typing import List

def longest_consecutive(nums: List[int]) -> int:
    """
    Hash set with intelligent sequence detection.
    Time: O(n), Space: O(n)
    """
    if not nums:
        return 0
    
    num_set = set(nums)
    max_length = 0
    
    for num in num_set:
        # Only start counting from sequence beginning
        if num - 1 not in num_set:
            current_num = num
            current_length = 1
            
            # Count consecutive numbers
            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1
            
            max_length = max(max_length, current_length)
    
    return max_length
`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    leetcodeUrl: 'https://leetcode.com/problems/longest-consecutive-sequence/',
    youtubeUrl: 'https://www.youtube.com/watch?v=P6RZZMu_maU',
  },
  // MEDIUM - Find All Duplicates in an Array
  {
    id: 'find-all-duplicates',
    title: 'Find All Duplicates in an Array',
    difficulty: 'Medium',
    topic: 'Arrays & Hashing',
    order: 17,
    description: `Given an integer array \`nums\` of length \`n\` where all the integers of \`nums\` are in the range \`[1, n]\` and each integer appears **once** or **twice**, return an array of all the integers that appears **twice**.

You must write an algorithm that runs in **O(n)** time and uses only constant extra space.`,
    examples: [
      {
        input: 'nums = [4,3,2,7,8,2,3,1]',
        output: '[2,3]',
      },
      {
        input: 'nums = [1,1,2]',
        output: '[1]',
      },
    ],
    constraints: [
      'n == nums.length',
      '1 <= n <= 10^5',
      '1 <= nums[i] <= n',
      'Each element appears once or twice',
    ],
    hints: [
      'Use the array itself as a hash map',
      'Mark visited numbers by negating values at indices',
      'If value at index is already negative, we found a duplicate',
    ],
    starterCode: `from typing import List

def find_duplicates(nums: List[int]) -> List[int]:
    """
    Find all duplicates in array.
    
    Args:
        nums: Array where each element appears once or twice
        
    Returns:
        List of elements that appear twice
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[4, 3, 2, 7, 8, 2, 3, 1]],
        expected: [2, 3],
      },
      {
        input: [[1, 1, 2]],
        expected: [1],
      },
    ],
    solution: `from typing import List

def find_duplicates(nums: List[int]) -> List[int]:
    """
    In-place marking using sign.
    Time: O(n), Space: O(1)
    """
    result = []
    
    for num in nums:
        index = abs(num) - 1
        
        # If already negative, this is a duplicate
        if nums[index] < 0:
            result.append(abs(num))
        else:
            # Mark as visited by negating
            nums[index] = -nums[index]
    
    return result
`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl:
      'https://leetcode.com/problems/find-all-duplicates-in-an-array/',
    youtubeUrl: 'https://www.youtube.com/watch?v=aMsSF1Il3IY',
  },
  // MEDIUM - Set Matrix Zeroes
  {
    id: 'set-matrix-zeroes',
    title: 'Set Matrix Zeroes',
    difficulty: 'Medium',
    topic: 'Arrays & Hashing',
    order: 18,
    description: `Given an \`m x n\` integer matrix \`matrix\`, if an element is \`0\`, set its entire row and column to \`0\`'s.

You must do it **in place**.`,
    examples: [
      {
        input: 'matrix = [[1,1,1],[1,0,1],[1,1,1]]',
        output: '[[1,0,1],[0,0,0],[1,0,1]]',
      },
      {
        input: 'matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]',
        output: '[[0,0,0,0],[0,4,5,0],[0,3,1,0]]',
      },
    ],
    constraints: [
      'm == matrix.length',
      'n == matrix[0].length',
      '1 <= m, n <= 200',
      '-2^31 <= matrix[i][j] <= 2^31 - 1',
    ],
    hints: [
      'Use first row and first column as markers',
      'Need separate variable for first cell',
      'Process matrix in two passes',
    ],
    starterCode: `from typing import List

def set_zeroes(matrix: List[List[int]]) -> None:
    """
    Set entire row and column to zero if element is zero.
    Modify matrix in-place.
    
    Args:
        matrix: m x n matrix
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
          ],
        ],
        expected: [
          [1, 0, 1],
          [0, 0, 0],
          [1, 0, 1],
        ],
      },
      {
        input: [
          [
            [0, 1, 2, 0],
            [3, 4, 5, 2],
            [1, 3, 1, 5],
          ],
        ],
        expected: [
          [0, 0, 0, 0],
          [0, 4, 5, 0],
          [0, 3, 1, 0],
        ],
      },
    ],
    solution: `from typing import List

def set_zeroes(matrix: List[List[int]]) -> None:
    """
    Use first row/col as markers.
    Time: O(m * n), Space: O(1)
    """
    m, n = len(matrix), len(matrix[0])
    first_row_zero = False
    first_col_zero = False
    
    # Check if first row has zero
    for j in range(n):
        if matrix[0][j] == 0:
            first_row_zero = True
            break
    
    # Check if first column has zero
    for i in range(m):
        if matrix[i][0] == 0:
            first_col_zero = True
            break
    
    # Use first row and column as markers
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0
    
    # Set zeros based on markers
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0
    
    # Handle first row and column
    if first_row_zero:
        for j in range(n):
            matrix[0][j] = 0
    
    if first_col_zero:
        for i in range(m):
            matrix[i][0] = 0
`,
    timeComplexity: 'O(m * n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/set-matrix-zeroes/',
    youtubeUrl: 'https://www.youtube.com/watch?v=T41rL0L3Pnw',
  },
  // HARD - First Missing Positive
  {
    id: 'first-missing-positive',
    title: 'First Missing Positive',
    difficulty: 'Hard',
    topic: 'Arrays & Hashing',
    order: 19,
    description: `Given an unsorted integer array \`nums\`, return the smallest missing positive integer.

You must implement an algorithm that runs in **O(n)** time and uses **O(1)** auxiliary space.`,
    examples: [
      {
        input: 'nums = [1,2,0]',
        output: '3',
      },
      {
        input: 'nums = [3,4,-1,1]',
        output: '2',
      },
      {
        input: 'nums = [7,8,9,11,12]',
        output: '1',
      },
    ],
    constraints: ['1 <= nums.length <= 10^5', '-2^31 <= nums[i] <= 2^31 - 1'],
    hints: [
      'Use the array itself as a hash table',
      'Place each number at its correct index: nums[i] = i + 1',
      'First index where nums[i] != i + 1 is the answer',
    ],
    starterCode: `from typing import List

def first_missing_positive(nums: List[int]) -> int:
    """
    Find smallest missing positive integer.
    
    Args:
        nums: Array of integers
        
    Returns:
        Smallest missing positive integer
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 2, 0]],
        expected: 3,
      },
      {
        input: [[3, 4, -1, 1]],
        expected: 2,
      },
      {
        input: [[7, 8, 9, 11, 12]],
        expected: 1,
      },
    ],
    solution: `from typing import List

def first_missing_positive(nums: List[int]) -> int:
    """
    In-place cyclic sort.
    Time: O(n), Space: O(1)
    """
    n = len(nums)
    
    # Place each positive number at its correct position
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            # Swap nums[i] with nums[nums[i] - 1]
            correct_idx = nums[i] - 1
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
    
    # Find first missing positive
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    
    # All positions correct, return n + 1
    return n + 1
`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/first-missing-positive/',
    youtubeUrl: 'https://www.youtube.com/watch?v=8g78yfzMlao',
  },
  // HARD - Subarrays with K Different Integers
  {
    id: 'subarrays-k-different',
    title: 'Subarrays with K Different Integers',
    difficulty: 'Hard',
    topic: 'Arrays & Hashing',
    order: 20,
    description: `Given an integer array \`nums\` and an integer \`k\`, return the number of **good subarrays** of \`nums\`.

A **good array** is an array where the number of different integers in that array is exactly \`k\`.

A **subarray** is a contiguous part of an array.`,
    examples: [
      {
        input: 'nums = [1,2,1,2,3], k = 2',
        output: '7',
        explanation:
          'Subarrays formed with exactly 2 different integers: [1,2], [2,1], [1,2], [2,3], [1,2,1], [2,1,2], [1,2,1,2].',
      },
      {
        input: 'nums = [1,2,1,3,4], k = 3',
        output: '3',
        explanation: '[1,2,1,3], [2,1,3], [1,3,4].',
      },
    ],
    constraints: [
      '1 <= nums.length <= 2 * 10^4',
      '1 <= nums[i], k <= nums.length',
    ],
    hints: [
      'Exactly k different = (at most k) - (at most k-1)',
      'Use sliding window with hash map',
      'Count frequency of elements in current window',
    ],
    starterCode: `from typing import List

def subarrays_with_k_distinct(nums: List[int], k: int) -> int:
    """
    Count subarrays with exactly k different integers.
    
    Args:
        nums: Array of integers
        k: Number of different integers required
        
    Returns:
        Number of subarrays with exactly k different integers
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 2, 1, 2, 3], 2],
        expected: 7,
      },
      {
        input: [[1, 2, 1, 3, 4], 3],
        expected: 3,
      },
    ],
    solution: `from typing import List

def subarrays_with_k_distinct(nums: List[int], k: int) -> int:
    """
    Exactly k = at_most(k) - at_most(k-1).
    Time: O(n), Space: O(k)
    """
    def at_most_k(k: int) -> int:
        """Count subarrays with at most k different integers"""
        count = 0
        freq = {}
        left = 0
        
        for right in range(len(nums)):
            # Add right element
            freq[nums[right]] = freq.get(nums[right], 0) + 1
            
            # Shrink window if more than k different
            while len(freq) > k:
                freq[nums[left]] -= 1
                if freq[nums[left]] == 0:
                    del freq[nums[left]]
                left += 1
            
            # All subarrays ending at right
            count += right - left + 1
        
        return count
    
    return at_most_k(k) - at_most_k(k - 1)
`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(k)',
    leetcodeUrl:
      'https://leetcode.com/problems/subarrays-with-k-different-integers/',
    youtubeUrl: 'https://www.youtube.com/watch?v=CBSeilNZePg',
  },
  // HARD - Longest Substring with At Most K Distinct Characters
  {
    id: 'longest-substring-k-distinct',
    title: 'Longest Substring with At Most K Distinct Characters',
    difficulty: 'Hard',
    topic: 'Arrays & Hashing',
    order: 21,
    description: `Given a string \`s\` and an integer \`k\`, return the length of the longest substring of \`s\` that contains at most \`k\` distinct characters.`,
    examples: [
      {
        input: 's = "eceba", k = 2',
        output: '3',
        explanation: 'The substring is "ece" with length 3.',
      },
      {
        input: 's = "aa", k = 1',
        output: '2',
        explanation: 'The substring is "aa" with length 2.',
      },
    ],
    constraints: ['1 <= s.length <= 5 * 10^4', '0 <= k <= 50'],
    hints: [
      'Use sliding window with hash map',
      'Track character frequencies',
      'Shrink window when distinct count exceeds k',
    ],
    starterCode: `def length_of_longest_substring_k_distinct(s: str, k: int) -> int:
    """
    Find longest substring with at most k distinct characters.
    
    Args:
        s: Input string
        k: Maximum number of distinct characters
        
    Returns:
        Length of longest valid substring
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['eceba', 2],
        expected: 3,
      },
      {
        input: ['aa', 1],
        expected: 2,
      },
    ],
    solution: `def length_of_longest_substring_k_distinct(s: str, k: int) -> int:
    """
    Sliding window with frequency map.
    Time: O(n), Space: O(k)
    """
    if k == 0:
        return 0
    
    max_length = 0
    freq = {}
    left = 0
    
    for right in range(len(s)):
        # Add right character
        freq[s[right]] = freq.get(s[right], 0) + 1
        
        # Shrink window if too many distinct characters
        while len(freq) > k:
            freq[s[left]] -= 1
            if freq[s[left]] == 0:
                del freq[s[left]]
            left += 1
        
        # Update max length
        max_length = max(max_length, right - left + 1)
    
    return max_length
`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(k)',
    leetcodeUrl:
      'https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/',
    youtubeUrl: 'https://www.youtube.com/watch?v=MK-NZ4hN7rs',
  },
  // HARD - Maximum Frequency Stack
  {
    id: 'max-frequency-stack',
    title: 'Maximum Frequency Stack',
    difficulty: 'Hard',
    topic: 'Arrays & Hashing',
    order: 22,
    description: `Design a stack-like data structure to push elements to the stack and pop the most frequent element from the stack.

Implement the \`FreqStack\` class:
- \`FreqStack()\` constructs an empty frequency stack.
- \`void push(int val)\` pushes an integer \`val\` onto the top of the stack.
- \`int pop()\` removes and returns the most frequent element in the stack. If there is a tie for the most frequent element, the element closest to the stack's top is removed and returned.`,
    examples: [
      {
        input:
          '["FreqStack","push","push","push","push","push","push","pop","pop","pop","pop"]\\n[[],[5],[7],[5],[7],[4],[5],[],[],[],[]]',
        output: '[null,null,null,null,null,null,null,5,7,5,4]',
        explanation:
          'FreqStack has [5,7,5,7,4,5]. Pop returns 5 (highest freq), then 7 (highest freq of remaining), then 5, then 4.',
      },
    ],
    constraints: [
      '0 <= val <= 10^9',
      'At most 2 * 10^4 calls will be made to push and pop',
    ],
    hints: [
      'Use a hash map to track frequency of each value',
      'Use a hash map of stacks grouped by frequency',
      'Track maximum frequency',
    ],
    starterCode: `class FreqStack:
    """
    Stack that always pops the most frequent element.
    """
    
    def __init__(self):
        """Initialize the frequency stack."""
        pass
    
    def push(self, val: int) -> None:
        """Push value to the stack."""
        pass
    
    def pop(self) -> int:
        """Pop the most frequent element."""
        pass
`,
    testCases: [
      {
        input: [
          [
            'push',
            'push',
            'push',
            'push',
            'push',
            'push',
            'pop',
            'pop',
            'pop',
            'pop',
          ],
          [[5], [7], [5], [7], [4], [5], [], [], [], []],
        ],
        expected: [null, null, null, null, null, null, 5, 7, 5, 4],
      },
    ],
    solution: `from collections import defaultdict

class FreqStack:
    """
    Frequency-based stack.
    Time: O(1) for push and pop
    Space: O(n)
    """
    
    def __init__(self):
        self.freq = {}  # value -> frequency
        self.group = defaultdict(list)  # frequency -> stack of values
        self.max_freq = 0
    
    def push(self, val: int) -> None:
        # Update frequency
        self.freq[val] = self.freq.get(val, 0) + 1
        f = self.freq[val]
        
        # Update max frequency
        self.max_freq = max(self.max_freq, f)
        
        # Add to frequency group
        self.group[f].append(val)
    
    def pop(self) -> int:
        # Pop from highest frequency group
        val = self.group[self.max_freq].pop()
        
        # Update frequency
        self.freq[val] -= 1
        
        # Update max frequency if needed
        if not self.group[self.max_freq]:
            self.max_freq -= 1
        
        return val
`,
    timeComplexity: 'O(1) for push and pop',
    spaceComplexity: 'O(n)',
    leetcodeUrl: 'https://leetcode.com/problems/maximum-frequency-stack/',
    youtubeUrl: 'https://www.youtube.com/watch?v=Z6idIicFDOE',
  },
  // HARD - Max Points on a Line
  {
    id: 'max-points-on-line',
    title: 'Max Points on a Line',
    difficulty: 'Hard',
    topic: 'Arrays & Hashing',
    order: 23,
    description: `Given an array of \`points\` where \`points[i] = [xi, yi]\` represents a point on the **X-Y** plane, return the maximum number of points that lie on the same straight line.`,
    examples: [
      {
        input: 'points = [[1,1],[2,2],[3,3]]',
        output: '3',
      },
      {
        input: 'points = [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]',
        output: '4',
      },
    ],
    constraints: [
      '1 <= points.length <= 300',
      'points[i].length == 2',
      '-10^4 <= xi, yi <= 10^4',
      'All the points are unique',
    ],
    hints: [
      'For each point, calculate slopes to all other points',
      'Use hash map to count points with same slope',
      'Handle vertical lines and duplicate points',
      'Use GCD to normalize slopes',
    ],
    starterCode: `from typing import List

def max_points(points: List[List[int]]) -> int:
    """
    Find maximum points on the same line.
    
    Args:
        points: Array of [x, y] coordinates
        
    Returns:
        Maximum number of points on same line
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [
            [1, 1],
            [2, 2],
            [3, 3],
          ],
        ],
        expected: 3,
      },
      {
        input: [
          [
            [1, 1],
            [3, 2],
            [5, 3],
            [4, 1],
            [2, 3],
            [1, 4],
          ],
        ],
        expected: 4,
      },
    ],
    solution: `from typing import List
from math import gcd
from collections import defaultdict

def max_points(points: List[List[int]]) -> int:
    """
    For each point, count slopes to all other points.
    Time: O(n^2), Space: O(n)
    """
    if len(points) <= 2:
        return len(points)
    
    max_count = 0
    
    for i in range(len(points)):
        slopes = defaultdict(int)
        
        for j in range(i + 1, len(points)):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            
            # Normalize slope using GCD
            g = gcd(dx, dy)
            slope = (dx // g, dy // g)
            
            slopes[slope] += 1
            max_count = max(max_count, slopes[slope])
    
    return max_count + 1  # +1 for the starting point
`,
    timeComplexity: 'O(n^2)',
    spaceComplexity: 'O(n)',
    leetcodeUrl: 'https://leetcode.com/problems/max-points-on-a-line/',
    youtubeUrl: 'https://www.youtube.com/watch?v=C1OxnJRpm7o',
  },
];
