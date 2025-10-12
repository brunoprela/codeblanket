import { Problem } from '../types';

export const slidingWindowProblems: Problem[] = [
  {
    id: 'best-time-to-buy-sell-stock',
    title: 'Best Time to Buy and Sell Stock',
    difficulty: 'Easy',
    description: `You are given an array \`prices\` where \`prices[i]\` is the price of a given stock on the \`i\`th day.

You want to maximize your profit by choosing a **single day** to buy one stock and choosing a **different day in the future** to sell that stock.

Return **the maximum profit** you can achieve from this transaction. If you cannot achieve any profit, return \`0\`.

**LeetCode:** [121. Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)
**YouTube:** [NeetCode - Best Time to Buy and Sell Stock](https://www.youtube.com/watch?v=1pkOgXD63yU)

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

**LeetCode:** [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
**YouTube:** [NeetCode - Longest Substring Without Repeating Characters](https://www.youtube.com/watch?v=wiGpQwVHdE0)

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

**LeetCode:** [76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)
**YouTube:** [NeetCode - Minimum Window Substring](https://www.youtube.com/watch?v=jSto0O4AJbM)

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
    order: 3,
    topic: 'Sliding Window',
    leetcodeUrl: 'https://leetcode.com/problems/minimum-window-substring/',
    youtubeUrl: 'https://www.youtube.com/watch?v=jSto0O4AJbM',
  },
];
