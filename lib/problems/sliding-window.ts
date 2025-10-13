import { Problem } from '../types';

export const slidingWindowProblems: Problem[] = [
  {
    id: 'best-time-to-buy-sell-stock',
    title: 'Best Time to Buy and Sell Stock',
    difficulty: 'Easy',
    description: `You are given an array \`prices\` where \`prices[i]\` is the price of a given stock on the \`i\`th day.

You want to maximize your profit by choosing a **single day** to buy one stock and choosing a **different day in the future** to sell that stock.

Return **the maximum profit** you can achieve from this transaction. If you cannot achieve any profit, return \`0\`.


**Approach:**
This is a sliding window variant. Track the minimum price seen so far (buy price) and calculate profit at each day. The "window" conceptually represents the buying and selling days, expanding to find the maximum profit.`,
    examples: [
      {
        input: 'prices = [7,1,5,3,6,4]',
        output: '5',
        explanation:
          'Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.',
      },
      {
        input: 'prices = [7,6,4,3,1]',
        output: '0',
        explanation:
          'In this case, no transactions are done and the max profit = 0.',
      },
    ],
    constraints: ['1 <= prices.length <= 10^5', '0 <= prices[i] <= 10^4'],
    hints: [
      'Track the minimum price encountered so far',
      'For each price, calculate the profit if we sell at that price',
      'Keep track of the maximum profit seen',
      'You only need one pass through the array - O(N) time',
    ],
    starterCode: `from typing import List

def max_profit(prices: List[int]) -> int:
    """
    Find the maximum profit from buying and selling stock once.
    
    Args:
        prices: Array of stock prices
        
    Returns:
        Maximum profit possible
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[7, 1, 5, 3, 6, 4]],
        expected: 5,
      },
      {
        input: [[7, 6, 4, 3, 1]],
        expected: 0,
      },
      {
        input: [[2, 4, 1]],
        expected: 2,
      },
      {
        input: [[3, 2, 6, 5, 0, 3]],
        expected: 4,
      },
    ],
    solution: `from typing import List

def max_profit(prices: List[int]) -> int:
    """
    Sliding window approach: track min price and max profit.
    Time: O(N), Space: O(1)
    """
    if not prices:
        return 0
    
    min_price = float('inf')
    max_profit = 0
    
    for price in prices:
        # Update minimum price (best day to buy)
        min_price = min(min_price, price)
        
        # Calculate profit if we sell today
        profit = price - min_price
        
        # Update maximum profit
        max_profit = max(max_profit, profit)
    
    return max_profit


# Alternative: More explicit window tracking
def max_profit_window(prices: List[int]) -> int:
    left = 0  # Buy day
    max_profit = 0
    
    for right in range(1, len(prices)):  # Sell day
        # If price decreased, move buy day forward
        if prices[right] < prices[left]:
            left = right
        else:
            # Calculate profit for this window
            profit = prices[right] - prices[left]
            max_profit = max(max_profit, profit)
    
    return max_profit`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',

    leetcodeUrl:
      'https://leetcode.com/problems/best-time-to-buy-and-sell-stock/',
    youtubeUrl: 'https://www.youtube.com/watch?v=1pkOgXD63yU',
    order: 1,
    topic: 'Sliding Window',
    leetcodeUrl:
      'https://leetcode.com/problems/best-time-to-buy-and-sell-stock/',
    youtubeUrl: 'https://www.youtube.com/watch?v=1pkOgXD63yU',
  },
  {
    id: 'longest-substring-without-repeating',
    title: 'Longest Substring Without Repeating Characters',
    difficulty: 'Medium',
    description: `Given a string \`s\`, find the length of the **longest substring** without repeating characters.


**Approach:**
Use a variable-size sliding window. Expand the window by moving the right pointer to add new characters. When a duplicate is found, shrink the window from the left until the duplicate is removed. Track the maximum window size seen.`,
    examples: [
      {
        input: 's = "abcabcbb"',
        output: '3',
        explanation: 'The answer is "abc", with the length of 3.',
      },
      {
        input: 's = "bbbbb"',
        output: '1',
        explanation: 'The answer is "b", with the length of 1.',
      },
      {
        input: 's = "pwwkew"',
        output: '3',
        explanation:
          'The answer is "wke", with the length of 3. Note that "pwke" is a subsequence and not a substring.',
      },
    ],
    constraints: [
      '0 <= s.length <= 5 * 10^4',
      's consists of English letters, digits, symbols and spaces',
    ],
    hints: [
      'Use a sliding window with a hash set to track characters in the current window',
      'When you encounter a duplicate, remove characters from the left until the duplicate is gone',
      'Track the maximum window size throughout the process',
      'Alternative: Use a hash map to store the last seen index of each character',
    ],
    starterCode: `def length_of_longest_substring(s: str) -> int:
    """
    Find the length of the longest substring without repeating characters.
    
    Args:
        s: Input string
        
    Returns:
        Length of longest substring without repeating characters
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['abcabcbb'],
        expected: 3,
      },
      {
        input: ['bbbbb'],
        expected: 1,
      },
      {
        input: ['pwwkew'],
        expected: 3,
      },
      {
        input: [''],
        expected: 0,
      },
      {
        input: ['au'],
        expected: 2,
      },
      {
        input: ['dvdf'],
        expected: 3,
      },
    ],
    solution: `def length_of_longest_substring(s: str) -> int:
    """
    Sliding window with hash set.
    Time: O(N), Space: O(min(N, M)) where M is charset size
    """
    char_set = set()
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # Remove characters from left until no duplicate
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        
        # Add current character
        char_set.add(s[right])
        
        # Update max length
        max_length = max(max_length, right - left + 1)
    
    return max_length


# Alternative: Hash map with last seen index (more efficient)
def length_of_longest_substring_optimized(s: str) -> int:
    """
    Optimized approach using hash map to store last seen index.
    Allows skipping ahead instead of incrementing left one by one.
    """
    char_index = {}
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # If character seen before and within current window
        if s[right] in char_index and char_index[s[right]] >= left:
            # Move left pointer past the previous occurrence
            left = char_index[s[right]] + 1
        
        # Update last seen index
        char_index[s[right]] = right
        
        # Update max length
        max_length = max(max_length, right - left + 1)
    
    return max_length`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(min(n, m)) where m is the character set size',

    leetcodeUrl:
      'https://leetcode.com/problems/longest-substring-without-repeating-characters/',
    youtubeUrl: 'https://www.youtube.com/watch?v=wiGpQwVHdE0',
    order: 2,
    topic: 'Sliding Window',
    leetcodeUrl:
      'https://leetcode.com/problems/longest-substring-without-repeating-characters/',
    youtubeUrl: 'https://www.youtube.com/watch?v=wiGpQwVHdE0',
  },
  {
    id: 'minimum-window-substring',
    title: 'Minimum Window Substring',
    difficulty: 'Hard',
    description: `Given two strings \`s\` and \`t\` of lengths \`m\` and \`n\` respectively, return **the minimum window substring** of \`s\` such that every character in \`t\` (including duplicates) is included in the window. If there is no such substring, return the empty string \`""\`.

The testcases will be generated such that the answer is **unique**.


**Approach:**
Use a variable-size sliding window. Expand the window until it contains all characters of \`t\`, then shrink from the left to find the minimum valid window. Use two hash maps to track required characters and current window characters.`,
    examples: [
      {
        input: 's = "ADOBECODEBANC", t = "ABC"',
        output: '"BANC"',
        explanation:
          "The minimum window substring \"BANC\" includes 'A', 'B', and 'C' from string t.",
      },
      {
        input: 's = "a", t = "a"',
        output: '"a"',
        explanation: 'The entire string s is the minimum window.',
      },
      {
        input: 's = "a", t = "aa"',
        output: '""',
        explanation:
          "Both 'a's from t must be included in the window. Since the largest window of s only has one 'a', return empty string.",
      },
    ],
    constraints: [
      'm == s.length',
      'n == t.length',
      '1 <= m, n <= 10^5',
      's and t consist of uppercase and lowercase English letters',
    ],
    hints: [
      'Use two hash maps: one for required characters (from t), one for current window',
      'Expand window by moving right pointer until all characters are included',
      'Shrink window from left while still valid, updating minimum at each step',
      'Track how many required characters have been satisfied',
      'Use Counter from collections for easier frequency comparison',
    ],
    starterCode: `def min_window(s: str, t: str) -> str:
    """
    Find the minimum window substring containing all characters of t.
    
    Args:
        s: Source string
        t: Target string with required characters
        
    Returns:
        Minimum window substring, or empty string if none exists
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['ADOBECODEBANC', 'ABC'],
        expected: 'BANC',
      },
      {
        input: ['a', 'a'],
        expected: 'a',
      },
      {
        input: ['a', 'aa'],
        expected: '',
      },
      {
        input: ['ab', 'b'],
        expected: 'b',
      },
      {
        input: ['abc', 'cba'],
        expected: 'abc',
      },
    ],
    solution: `def min_window(s: str, t: str) -> str:
    """
    Sliding window with two frequency maps.
    Time: O(M + N), Space: O(M + N)
    """
    if not s or not t or len(s) < len(t):
        return ""
    
    from collections import Counter
    
    # Count required characters
    required = Counter(t)
    required_count = len(required)  # Number of unique chars
    
    # Track current window
    window = {}
    formed = 0  # Number of chars with desired frequency
    
    # Result: (window_length, left, right)
    result = float('inf'), 0, 0
    left = 0
    
    for right in range(len(s)):
        # Add character to window
        char = s[right]
        window[char] = window.get(char, 0) + 1
        
        # Check if this character's frequency matches requirement
        if char in required and window[char] == required[char]:
            formed += 1
        
        # Try to shrink window while still valid
        while formed == required_count and left <= right:
            # Update result if this window is smaller
            if right - left + 1 < result[0]:
                result = (right - left + 1, left, right)
            
            # Remove leftmost character
            char = s[left]
            window[char] -= 1
            if char in required and window[char] < required[char]:
                formed -= 1
            
            left += 1
    
    # Return the minimum window, or empty string if none found
    return "" if result[0] == float('inf') else s[result[1]:result[2] + 1]


# Alternative: More explicit tracking
def min_window_alt(s: str, t: str) -> str:
    if not s or not t:
        return ""
    
    from collections import defaultdict
    
    # Build frequency map for t
    target_freq = defaultdict(int)
    for char in t:
        target_freq[char] += 1
    
    required = len(target_freq)
    formed = 0
    
    window_freq = defaultdict(int)
    left = 0
    min_len = float('inf')
    min_left = 0
    
    for right in range(len(s)):
        char = s[right]
        window_freq[char] += 1
        
        # Check if frequency matches for this character
        if char in target_freq and window_freq[char] == target_freq[char]:
            formed += 1
        
        # Shrink window
        while formed == required:
            # Update minimum window
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_left = left
            
            # Remove from window
            left_char = s[left]
            window_freq[left_char] -= 1
            if left_char in target_freq and window_freq[left_char] < target_freq[left_char]:
                formed -= 1
            
            left += 1
    
    return "" if min_len == float('inf') else s[min_left:min_left + min_len]`,
    timeComplexity: 'O(m + n) where m = len(s), n = len(t)',
    spaceComplexity: 'O(m + n)',

    leetcodeUrl: 'https://leetcode.com/problems/minimum-window-substring/',
    youtubeUrl: 'https://www.youtube.com/watch?v=jSto0O4AJbM',
    order: 3,
    topic: 'Sliding Window',
    leetcodeUrl: 'https://leetcode.com/problems/minimum-window-substring/',
    youtubeUrl: 'https://www.youtube.com/watch?v=jSto0O4AJbM',
  },
  // EASY - Maximum Average Subarray I
  {
    id: 'maximum-average-subarray-i',
    title: 'Maximum Average Subarray I',
    difficulty: 'Easy',
    topic: 'Sliding Window',
    order: 4,
    description: `You are given an integer array \`nums\` consisting of \`n\` elements, and an integer \`k\`.

Find a contiguous subarray whose **length is equal to** \`k\` that has the maximum average value and return this value.`,
    examples: [
      {
        input: 'nums = [1,12,-5,-6,50,3], k = 4',
        output: '12.75000',
        explanation:
          'Maximum average is (12 - 5 - 6 + 50) / 4 = 51 / 4 = 12.75',
      },
      {
        input: 'nums = [5], k = 1',
        output: '5.00000',
      },
    ],
    constraints: [
      'n == nums.length',
      '1 <= k <= n <= 10^5',
      '-10^4 <= nums[i] <= 10^4',
    ],
    hints: [
      'Use a sliding window of size k',
      'Calculate sum of first k elements',
      'Slide window: subtract left, add right',
      'Track maximum sum',
    ],
    starterCode: `from typing import List

def find_max_average(nums: List[int], k: int) -> float:
    """
    Find maximum average of contiguous subarray of size k.
    
    Args:
        nums: Array of integers
        k: Subarray size
        
    Returns:
        Maximum average value
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 12, -5, -6, 50, 3], 4],
        expected: 12.75,
      },
      {
        input: [[5], 1],
        expected: 5.0,
      },
      {
        input: [[0, 1, 1, 3, 3], 4],
        expected: 2.0,
      },
    ],
    solution: `from typing import List

def find_max_average(nums: List[int], k: int) -> float:
    """
    Fixed-size sliding window.
    Time: O(n), Space: O(1)
    """
    # Calculate sum of first k elements
    window_sum = sum(nums[:k])
    max_sum = window_sum
    
    # Slide the window
    for i in range(k, len(nums)):
        window_sum = window_sum - nums[i - k] + nums[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum / k
`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/maximum-average-subarray-i/',
    youtubeUrl: 'https://www.youtube.com/watch?v=R8fKH2HFLbY',
  },
  // EASY - Contains Duplicate II
  {
    id: 'contains-duplicate-ii',
    title: 'Contains Duplicate II',
    difficulty: 'Easy',
    topic: 'Sliding Window',
    order: 5,
    description: `Given an integer array \`nums\` and an integer \`k\`, return \`true\` if there are two **distinct indices** \`i\` and \`j\` in the array such that \`nums[i] == nums[j]\` and \`abs(i - j) <= k\`.`,
    examples: [
      {
        input: 'nums = [1,2,3,1], k = 3',
        output: 'true',
      },
      {
        input: 'nums = [1,0,1,1], k = 1',
        output: 'true',
      },
      {
        input: 'nums = [1,2,3,1,2,3], k = 2',
        output: 'false',
      },
    ],
    constraints: [
      '1 <= nums.length <= 10^5',
      '-10^9 <= nums[i] <= 10^9',
      '0 <= k <= 10^5',
    ],
    hints: [
      'Use a sliding window of size k',
      'Maintain a set of elements in the window',
      'Check if current element exists in the set',
    ],
    starterCode: `from typing import List

def contains_nearby_duplicate(nums: List[int], k: int) -> bool:
    """
    Check if duplicate exists within distance k.
    
    Args:
        nums: Array of integers
        k: Maximum distance
        
    Returns:
        True if duplicate exists within distance k
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 2, 3, 1], 3],
        expected: true,
      },
      {
        input: [[1, 0, 1, 1], 1],
        expected: true,
      },
      {
        input: [[1, 2, 3, 1, 2, 3], 2],
        expected: false,
      },
    ],
    solution: `from typing import List

def contains_nearby_duplicate(nums: List[int], k: int) -> bool:
    """
    Sliding window with set.
    Time: O(n), Space: O(min(n, k))
    """
    window = set()
    
    for i, num in enumerate(nums):
        # Check if duplicate in window
        if num in window:
            return True
        
        # Add to window
        window.add(num)
        
        # Remove element outside window
        if len(window) > k:
            window.remove(nums[i - k])
    
    return False

# Alternative: Hash map approach
def contains_nearby_duplicate_map(nums: List[int], k: int) -> bool:
    """
    Hash map to track last index.
    Time: O(n), Space: O(n)
    """
    last_index = {}
    
    for i, num in enumerate(nums):
        if num in last_index and i - last_index[num] <= k:
            return True
        last_index[num] = i
    
    return False
`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(min(n, k))',
    leetcodeUrl: 'https://leetcode.com/problems/contains-duplicate-ii/',
    youtubeUrl: 'https://www.youtube.com/watch?v=ypn0aZ0nrL4',
  },
  // EASY - Minimum Difference Between Highest and Lowest of K Scores
  {
    id: 'min-difference-k-scores',
    title: 'Minimum Difference Between Highest and Lowest of K Scores',
    difficulty: 'Easy',
    topic: 'Sliding Window',
    order: 6,
    description: `You are given a **0-indexed** integer array \`nums\`, where \`nums[i]\` represents the score of the \`ith\` student. You are also given an integer \`k\`.

Pick the scores of any \`k\` students from the array so that the **difference** between the **highest** and the **lowest** of the \`k\` scores is **minimized**.

Return the **minimum** possible difference.`,
    examples: [
      {
        input: 'nums = [90], k = 1',
        output: '0',
        explanation:
          'There is one way to pick score(s) of one student: [90]. The difference is 90 - 90 = 0.',
      },
      {
        input: 'nums = [9,4,1,7], k = 2',
        output: '2',
        explanation:
          'Pick scores 4 and 1. The difference is 4 - 1 = 2. (Can also pick 7 and 9, difference is 2).',
      },
    ],
    constraints: ['1 <= k <= nums.length <= 1000', '0 <= nums[i] <= 10^5'],
    hints: [
      'Sort the array first',
      'Use a sliding window of size k on sorted array',
      'Find minimum difference between window endpoints',
    ],
    starterCode: `from typing import List

def minimum_difference(nums: List[int], k: int) -> int:
    """
    Find minimum difference for k students.
    
    Args:
        nums: Array of scores
        k: Number of students to pick
        
    Returns:
        Minimum possible difference
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[90], 1],
        expected: 0,
      },
      {
        input: [[9, 4, 1, 7], 2],
        expected: 2,
      },
      {
        input: [[41900, 69441, 94407, 37498], 4],
        expected: 56509,
      },
    ],
    solution: `from typing import List

def minimum_difference(nums: List[int], k: int) -> int:
    """
    Sort + sliding window.
    Time: O(n log n), Space: O(1)
    """
    if k == 1:
        return 0
    
    # Sort the scores
    nums.sort()
    
    # Try all windows of size k
    min_diff = float('inf')
    for i in range(len(nums) - k + 1):
        # Difference is max - min in sorted window
        diff = nums[i + k - 1] - nums[i]
        min_diff = min(min_diff, diff)
    
    return min_diff
`,
    timeComplexity: 'O(n log n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl:
      'https://leetcode.com/problems/minimum-difference-between-highest-and-lowest-of-k-scores/',
    youtubeUrl: 'https://www.youtube.com/watch?v=kLHXyGCNzBA',
  },
  // EASY - Defanging an IP Address (String manipulation with sliding window concept)
  {
    id: 'max-consecutive-ones',
    title: 'Max Consecutive Ones',
    difficulty: 'Easy',
    topic: 'Sliding Window',
    order: 7,
    description: `Given a binary array \`nums\`, return the maximum number of consecutive \`1\`'s in the array.`,
    examples: [
      {
        input: 'nums = [1,1,0,1,1,1]',
        output: '3',
        explanation:
          'The first two digits or the last three digits are consecutive 1s. The maximum number of consecutive 1s is 3.',
      },
      {
        input: 'nums = [1,0,1,1,0,1]',
        output: '2',
      },
    ],
    constraints: ['1 <= nums.length <= 10^5', 'nums[i] is either 0 or 1'],
    hints: [
      'Track current consecutive count',
      'Reset when you see 0',
      'Track maximum count seen',
    ],
    starterCode: `from typing import List

def find_max_consecutive_ones(nums: List[int]) -> int:
    """
    Find maximum consecutive 1s.
    
    Args:
        nums: Binary array
        
    Returns:
        Maximum number of consecutive 1s
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 1, 0, 1, 1, 1]],
        expected: 3,
      },
      {
        input: [[1, 0, 1, 1, 0, 1]],
        expected: 2,
      },
      {
        input: [[0, 0]],
        expected: 0,
      },
    ],
    solution: `from typing import List

def find_max_consecutive_ones(nums: List[int]) -> int:
    """
    Single pass to count consecutive 1s.
    Time: O(n), Space: O(1)
    """
    max_count = 0
    current_count = 0
    
    for num in nums:
        if num == 1:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0
    
    return max_count
`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/max-consecutive-ones/',
    youtubeUrl: 'https://www.youtube.com/watch?v=2hBPrR8vx1I',
  },
  // EASY - Longest Nice Substring
  {
    id: 'longest-nice-substring',
    title: 'Longest Nice Substring',
    difficulty: 'Easy',
    topic: 'Sliding Window',
    order: 8,
    description: `A string \`s\` is **nice** if, for every letter of the alphabet that \`s\` contains, it appears **both** in uppercase and lowercase. For example, \`"abABB"\` is nice because \`'A'\` and \`'a'\` appear, and \`'B'\` and \`'b'\` appear. However, \`"abA"\` is not because \`'b'\` appears, but \`'B'\` does not.

Given a string \`s\`, return the longest **substring** of \`s\` that is **nice**. If there are multiple, return the substring of the **earliest** occurrence. If there are none, return an empty string.`,
    examples: [
      {
        input: 's = "YazaAay"',
        output: '"aAa"',
        explanation:
          '"aAa" is a nice string because \'A/a\' and \'Y/y\' are both present, but "aAa" is longer.',
      },
      {
        input: 's = "Bb"',
        output: '"Bb"',
        explanation: "\"Bb\" is a nice string because both 'B' and 'b' appear.",
      },
      {
        input: 's = "c"',
        output: '""',
        explanation: 'There are no nice substrings.',
      },
    ],
    constraints: [
      '1 <= s.length <= 100',
      's consists of uppercase and lowercase English letters',
    ],
    hints: [
      'Check all substrings',
      "For each substring, verify if it's nice",
      'A string is nice if for every char, both cases exist',
    ],
    starterCode: `def longest_nice_substring(s: str) -> str:
    """
    Find longest nice substring.
    
    Args:
        s: Input string
        
    Returns:
        Longest nice substring
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['YazaAay'],
        expected: 'aAa',
      },
      {
        input: ['Bb'],
        expected: 'Bb',
      },
      {
        input: ['c'],
        expected: '',
      },
    ],
    solution: `def longest_nice_substring(s: str) -> str:
    """
    Check all substrings for nice property.
    Time: O(n^2), Space: O(n)
    """
    def is_nice(substring: str) -> bool:
        """Check if substring is nice"""
        char_set = set(substring)
        for char in char_set:
            if char.swapcase() not in char_set:
                return False
        return True
    
    longest = ""
    n = len(s)
    
    # Try all substrings
    for i in range(n):
        for j in range(i + 1, n + 1):
            substring = s[i:j]
            if is_nice(substring) and len(substring) > len(longest):
                longest = substring
    
    return longest

# Alternative: Divide and conquer
def longest_nice_substring_dc(s: str) -> str:
    """
    Divide and conquer approach.
    Time: O(n^2) worst case, Space: O(n)
    """
    if len(s) < 2:
        return ""
    
    char_set = set(s)
    
    for i, char in enumerate(s):
        if char.swapcase() not in char_set:
            # Split at this position
            left = longest_nice_substring_dc(s[:i])
            right = longest_nice_substring_dc(s[i + 1:])
            return left if len(left) >= len(right) else right
    
    # Entire string is nice
    return s
`,
    timeComplexity: 'O(n^2)',
    spaceComplexity: 'O(n)',
    leetcodeUrl: 'https://leetcode.com/problems/longest-nice-substring/',
    youtubeUrl: 'https://www.youtube.com/watch?v=fS2Rz0_JVVE',
  },

  // EASY - Find All Anagrams in a String
  {
    id: 'find-all-anagrams',
    title: 'Find All Anagrams in a String',
    difficulty: 'Easy',
    topic: 'Sliding Window',
    description: `Given two strings \`s\` and \`p\`, return an array of all the start indices of \`p\`'s anagrams in \`s\`. You may return the answer in any order.

An **Anagram** is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.`,
    examples: [
      {
        input: 's = "cbaebabacd", p = "abc"',
        output: '[0,6]',
        explanation:
          'The substring with start index = 0 is "cba", which is an anagram of "abc". The substring with start index = 6 is "bac", which is an anagram of "abc".',
      },
      {
        input: 's = "abab", p = "ab"',
        output: '[0,1,2]',
      },
    ],
    constraints: [
      '1 <= s.length, p.length <= 3 * 10^4',
      's and p consist of lowercase English letters',
    ],
    hints: [
      'Use a sliding window of size len(p)',
      'Compare character frequencies',
    ],
    starterCode: `from typing import List

def find_anagrams(s: str, p: str) -> List[int]:
    """
    Find all start indices of anagrams of p in s.
    
    Args:
        s: String to search in
        p: Pattern string
        
    Returns:
        List of start indices
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['cbaebabacd', 'abc'],
        expected: [0, 6],
      },
      {
        input: ['abab', 'ab'],
        expected: [0, 1, 2],
      },
      {
        input: ['baa', 'aa'],
        expected: [1],
      },
    ],
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/find-all-anagrams-in-a-string/',
    youtubeUrl: 'https://www.youtube.com/watch?v=G8xtZy0fDKg',
  },

  // EASY - Maximum Sum of Subarray of Size K
  {
    id: 'max-sum-subarray-size-k',
    title: 'Maximum Sum of Subarray of Size K',
    difficulty: 'Easy',
    topic: 'Sliding Window',
    description: `Given an array of integers \`nums\` and an integer \`k\`, find the maximum sum of a contiguous subarray of size \`k\`.`,
    examples: [
      {
        input: 'nums = [2,1,5,1,3,2], k = 3',
        output: '9',
        explanation: 'Subarray [5,1,3] has the maximum sum 9.',
      },
      {
        input: 'nums = [2,3,4,1,5], k = 2',
        output: '7',
        explanation: 'Subarray [3,4] has the maximum sum 7.',
      },
    ],
    constraints: [
      '1 <= nums.length <= 10^5',
      '1 <= k <= nums.length',
      '-10^4 <= nums[i] <= 10^4',
    ],
    hints: [
      'Use sliding window technique',
      'Subtract the left element and add the right element',
    ],
    starterCode: `from typing import List

def max_sum_subarray(nums: List[int], k: int) -> int:
    """
    Find maximum sum of subarray of size k.
    
    Args:
        nums: Integer array
        k: Subarray size
        
    Returns:
        Maximum sum
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[2, 1, 5, 1, 3, 2], 3],
        expected: 9,
      },
      {
        input: [[2, 3, 4, 1, 5], 2],
        expected: 7,
      },
      {
        input: [[1, 4, 2, 10, 23, 3, 1, 0, 20], 4],
        expected: 39,
      },
    ],
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl:
      'https://leetcode.com/problems/maximum-sum-of-subarray-of-size-k/',
    youtubeUrl: 'https://www.youtube.com/watch?v=KtpqeN0Goro',
  },

  // EASY - Minimum Size Subarray Sum
  {
    id: 'minimum-size-subarray-sum',
    title: 'Minimum Size Subarray Sum',
    difficulty: 'Easy',
    topic: 'Sliding Window',
    description: `Given an array of positive integers \`nums\` and a positive integer \`target\`, return the minimal length of a subarray whose sum is greater than or equal to \`target\`. If there is no such subarray, return \`0\` instead.`,
    examples: [
      {
        input: 'target = 7, nums = [2,3,1,2,4,3]',
        output: '2',
        explanation:
          'The subarray [4,3] has the minimal length under the problem constraint.',
      },
      {
        input: 'target = 4, nums = [1,4,4]',
        output: '1',
      },
    ],
    constraints: [
      '1 <= target <= 10^9',
      '1 <= nums.length <= 10^5',
      '1 <= nums[i] <= 10^4',
    ],
    hints: [
      'Use variable-size sliding window',
      'Expand window by adding right, contract by removing left',
    ],
    starterCode: `from typing import List

def min_subarray_len(target: int, nums: List[int]) -> int:
    """
    Find minimum length of subarray with sum >= target.
    
    Args:
        target: Target sum
        nums: Array of positive integers
        
    Returns:
        Minimum length, or 0 if not possible
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [7, [2, 3, 1, 2, 4, 3]],
        expected: 2,
      },
      {
        input: [4, [1, 4, 4]],
        expected: 1,
      },
      {
        input: [11, [1, 1, 1, 1, 1, 1, 1, 1]],
        expected: 0,
      },
    ],
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/minimum-size-subarray-sum/',
    youtubeUrl: 'https://www.youtube.com/watch?v=aYqYMIqZx5s',
  },

  // MEDIUM - Permutation in String
  {
    id: 'permutation-in-string',
    title: 'Permutation in String',
    difficulty: 'Medium',
    topic: 'Sliding Window',
    description: `Given two strings \`s1\` and \`s2\`, return \`true\` if \`s2\` contains a permutation of \`s1\`, or \`false\` otherwise.

In other words, return \`true\` if one of \`s1\`'s permutations is the substring of \`s2\`.`,
    examples: [
      {
        input: 's1 = "ab", s2 = "eidbaooo"',
        output: 'true',
        explanation: 's2 contains one permutation of s1 ("ba").',
      },
      {
        input: 's1 = "ab", s2 = "eidboaoo"',
        output: 'false',
      },
    ],
    constraints: [
      '1 <= s1.length, s2.length <= 10^4',
      's1 and s2 consist of lowercase English letters',
    ],
    hints: [
      'Use a sliding window of size len(s1)',
      'Compare character frequencies',
    ],
    starterCode: `def check_inclusion(s1: str, s2: str) -> bool:
    """
    Check if s2 contains a permutation of s1.
    
    Args:
        s1: Pattern string
        s2: String to search in
        
    Returns:
        True if s2 contains permutation of s1
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['ab', 'eidbaooo'],
        expected: true,
      },
      {
        input: ['ab', 'eidboaoo'],
        expected: false,
      },
      {
        input: ['adc', 'dcda'],
        expected: true,
      },
    ],
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/permutation-in-string/',
    youtubeUrl: 'https://www.youtube.com/watch?v=UbyhOgBN834',
  },

  // MEDIUM - Longest Repeating Character Replacement
  {
    id: 'longest-repeating-character-replacement',
    title: 'Longest Repeating Character Replacement',
    difficulty: 'Medium',
    topic: 'Sliding Window',
    description: `You are given a string \`s\` and an integer \`k\`. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most \`k\` times.

Return the length of the longest substring containing the same letter you can get after performing the above operations.`,
    examples: [
      {
        input: 's = "ABAB", k = 2',
        output: '4',
        explanation: "Replace the two A's with two B's or vice versa.",
      },
      {
        input: 's = "AABABBA", k = 1',
        output: '4',
        explanation:
          'Replace the one A in the middle with B and form "AABBBBA". The substring "BBBB" has the longest repeating letters, which is 4.',
      },
    ],
    constraints: [
      '1 <= s.length <= 10^5',
      's consists of only uppercase English letters',
      '0 <= k <= s.length',
    ],
    hints: [
      'Use sliding window with variable size',
      'Track the most frequent character in the window',
      'If window_size - max_freq > k, shrink window',
    ],
    starterCode: `def character_replacement(s: str, k: int) -> int:
    """
    Find longest substring with same letter after k replacements.
    
    Args:
        s: String of uppercase letters
        k: Maximum number of replacements allowed
        
    Returns:
        Length of longest substring
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['ABAB', 2],
        expected: 4,
      },
      {
        input: ['AABABBA', 1],
        expected: 4,
      },
      {
        input: ['AAAA', 0],
        expected: 4,
      },
    ],
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl:
      'https://leetcode.com/problems/longest-repeating-character-replacement/',
    youtubeUrl: 'https://www.youtube.com/watch?v=gqXU1UyA8pk',
  },

  // MEDIUM - Frequency of the Most Frequent Element
  {
    id: 'frequency-most-frequent-element',
    title: 'Frequency of the Most Frequent Element',
    difficulty: 'Medium',
    topic: 'Sliding Window',
    description: `The **frequency** of an element is the number of times it occurs in an array.

You are given an integer array \`nums\` and an integer \`k\`. In one operation, you can choose an index of \`nums\` and increment the element at that index by \`1\`.

Return the **maximum possible frequency** of an element after performing **at most** \`k\` operations.`,
    examples: [
      {
        input: 'nums = [1,2,4], k = 5',
        output: '3',
        explanation:
          'Increment the first element three times and the second element two times to make nums = [4,4,4]. 4 has a frequency of 3.',
      },
      {
        input: 'nums = [1,4,8,13], k = 5',
        output: '2',
        explanation:
          'There are multiple optimal solutions: Increment the first element three times to make nums = [4,4,8,13]. 4 has a frequency of 2.',
      },
    ],
    constraints: [
      '1 <= nums.length <= 10^5',
      '1 <= nums[i] <= 10^5',
      '1 <= k <= 10^5',
    ],
    hints: [
      'Sort the array first',
      'Use sliding window',
      'For a window to be valid, sum of increments needed should be <= k',
    ],
    starterCode: `from typing import List

def max_frequency(nums: List[int], k: int) -> int:
    """
    Find maximum frequency after at most k increments.
    
    Args:
        nums: Integer array
        k: Maximum number of increments
        
    Returns:
        Maximum possible frequency
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 2, 4], 5],
        expected: 3,
      },
      {
        input: [[1, 4, 8, 13], 5],
        expected: 2,
      },
      {
        input: [[3, 9, 6], 2],
        expected: 1,
      },
    ],
    timeComplexity: 'O(n log n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl:
      'https://leetcode.com/problems/frequency-of-the-most-frequent-element/',
    youtubeUrl: 'https://www.youtube.com/watch?v=vgBrQ0NM5vE',
  },
];
