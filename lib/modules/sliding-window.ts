import { Module } from '@/lib/types';

export const slidingWindowModule: Module = {
  id: 'sliding-window',
  title: 'Sliding Window',
  description:
    'Master the sliding window technique for optimizing substring, subarray, and sequence problems.',
  icon: '🪟',
  timeComplexity: 'O(N) for most problems',
  spaceComplexity: 'O(1) to O(K)',
  sections: [
    {
      id: 'introduction',
      title: 'Introduction to Sliding Window',
      content: `The **sliding window** technique is an optimization pattern that transforms nested loops (O(N²) or O(N*K)) into single-pass algorithms (O(N)). It's used to solve problems involving **contiguous sequences** in arrays or strings.

**Core Concept:**
Instead of recalculating results from scratch for each subarray, we maintain a "window" that slides through the data, incrementally updating our result by:
1. **Adding** new elements as the window expands/moves right
2. **Removing** old elements as the window contracts/moves left

**Real-World Analogies:**
- **Netflix viewing window**: As you scroll, new thumbnails appear on the right while old ones disappear on the left
- **Train window view**: The scenery changes continuously as the train moves
- **Reading comprehension**: Your attention spans a few sentences at a time, sliding as you read

**When to Use Sliding Window:**
- Finding subarrays/substrings with specific properties
- Problems involving **"contiguous"** or **"consecutive"** elements
- Keywords: "longest", "shortest", "maximum", "minimum" with constraints
- Optimizing from O(N²) to O(N)

**Two Main Types:**

**1. Fixed-Size Window**
- Window size is constant (k elements)
- Always move right pointer, adjust left to maintain size
- Example: Maximum sum of k consecutive elements

**2. Variable-Size Window**
- Window size changes based on conditions
- Expand window when condition not met
- Shrink window when condition met
- Example: Longest substring without repeating characters`,
    },
    {
      id: 'patterns',
      title: 'Sliding Window Patterns',
      content: `**Pattern 1: Fixed-Size Window**

**Problem:** Find maximum sum of k consecutive elements.

**Visualization:**
\`\`\`
Array: [2, 1, 5, 1, 3, 2], k = 3

Window 1: [2, 1, 5] → sum = 8
Window 2:    [1, 5, 1] → sum = 7
Window 3:       [5, 1, 3] → sum = 9 ← maximum
Window 4:          [1, 3, 2] → sum = 6
\`\`\`

**Naive O(N*K):**
\`\`\`python
for i in range(len(arr) - k + 1):
    current_sum = sum(arr[i:i+k])  # Recalculate every time
    max_sum = max(max_sum, current_sum)
\`\`\`

**Optimized O(N) with Sliding Window:**
\`\`\`python
# Initial window
window_sum = sum(arr[:k])
max_sum = window_sum

# Slide the window
for i in range(k, len(arr)):
    window_sum += arr[i]      # Add new element
    window_sum -= arr[i - k]  # Remove old element
    max_sum = max(max_sum, window_sum)
\`\`\`

---

**Pattern 2: Variable-Size Window (Longest/Maximum)**

**Problem:** Longest substring without repeating characters.

**Visualization:**
\`\`\`
String: "abcabcbb"

Step 1: "a"       → valid, expand. Longest = 1
Step 2: "ab"      → valid, expand. Longest = 2
Step 3: "abc"     → valid, expand. Longest = 3
Step 4: "abca"    → invalid ('a' repeats), shrink
        "bca"     → valid, expand. Longest = 3
Step 5: "bcab"    → invalid ('b' repeats), shrink
        "cab"     → valid, expand. Longest = 3
...
\`\`\`

**Template:**
\`\`\`python
left = 0
max_length = 0
window_data = {}  # Track window state

for right in range(len(arr)):
    # Add arr[right] to window
    window_data[arr[right]] = window_data.get(arr[right], 0) + 1
    
    # Shrink window while condition violated
    while condition_violated(window_data):
        window_data[arr[left]] -= 1
        if window_data[arr[left]] == 0:
            del window_data[arr[left]]
        left += 1
    
    # Update result with current window
    max_length = max(max_length, right - left + 1)

return max_length
\`\`\`

---

**Pattern 3: Variable-Size Window (Shortest/Minimum)**

**Problem:** Minimum window substring containing all characters.

**Visualization:**
\`\`\`
String: "ADOBECODEBANC", Target: "ABC"

Expand until valid:
"ADOBEC" → contains A, B, C → valid! Length = 6

Shrink while still valid:
"DOBEC"  → no A, invalid → stop
"ADOBEC" → shortest so far = 6

Continue:
"ODEBANC" → contains A, B, C → valid! Length = 7
"BANC"    → contains A, B, C → valid! Length = 4 ← best!
\`\`\`

**Template:**
\`\`\`python
left = 0
min_length = float('inf')
required = {}  # Characters we need
window = {}    # Characters we have

for right in range(len(arr)):
    # Expand window
    window[arr[right]] = window.get(arr[right], 0) + 1
    
    # Shrink window while condition met
    while condition_met(window, required):
        min_length = min(min_length, right - left + 1)
        window[arr[left]] -= 1
        left += 1

return min_length
\`\`\`

---

**Pattern 4: Sliding Window with Auxiliary Data Structure**

Use hash maps, sets, or counters to track window state:

**Hash Map (Frequency Count):**
\`\`\`python
from collections import defaultdict

window = defaultdict(int)
for char in s[left:right+1]:
    window[char] += 1
\`\`\`

**Set (Unique Elements):**
\`\`\`python
window = set()
for i in range(left, right + 1):
    window.add(arr[i])
\`\`\`

**Counter (Efficient Frequency):**
\`\`\`python
from collections import Counter

window = Counter(s[left:right+1])
\`\`\``,
      codeExample: `from typing import List

def max_sum_fixed_window(arr: List[int], k: int) -> int:
    """
    Fixed-size sliding window: maximum sum of k consecutive elements.
    """
    if len(arr) < k:
        return 0
    
    # Calculate sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    # Slide the window
    for i in range(k, len(arr)):
        window_sum += arr[i]      # Add new element
        window_sum -= arr[i - k]  # Remove old element
        max_sum = max(max_sum, window_sum)
    
    return max_sum


def longest_substring_without_repeating(s: str) -> int:
    """
    Variable-size window: longest substring without repeating characters.
    """
    char_index = {}  # Track last seen index of each character
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # If character seen before and in current window
        if s[right] in char_index and char_index[s[right]] >= left:
            left = char_index[s[right]] + 1  # Move left past duplicate
        
        char_index[s[right]] = right
        max_length = max(max_length, right - left + 1)
    
    return max_length


def min_window_substring(s: str, t: str) -> str:
    """
    Minimum window substring containing all characters of t.
    """
    if not s or not t:
        return ""
    
    # Count characters we need
    from collections import Counter
    required = Counter(t)
    required_count = len(required)
    
    # Track characters in current window
    window = {}
    formed = 0  # Number of unique chars in window with desired frequency
    
    left = 0
    min_len = float('inf')
    result = (0, 0)
    
    for right in range(len(s)):
        # Add character to window
        char = s[right]
        window[char] = window.get(char, 0) + 1
        
        # Check if frequency matches required
        if char in required and window[char] == required[char]:
            formed += 1
        
        # Shrink window while valid
        while formed == required_count and left <= right:
            # Update result if this window is smaller
            if right - left + 1 < min_len:
                min_len = right - left + 1
                result = (left, right)
            
            # Remove leftmost character
            char = s[left]
            window[char] -= 1
            if char in required and window[char] < required[char]:
                formed -= 1
            left += 1
    
    return s[result[0]:result[1] + 1] if min_len != float('inf') else ""`,
    },
    {
      id: 'complexity',
      title: 'Complexity Analysis',
      content: `**Sliding Window Complexity:**

**Time Complexity:**
- **Fixed-size window**: O(N) where N is array length
  - Initial window: O(K) to calculate first sum
  - Sliding: O(N - K) windows, O(1) per window
  - Total: O(K + N - K) = O(N)

- **Variable-size window**: O(N)
  - Right pointer moves N times: O(N)
  - Left pointer moves at most N times: O(N)
  - Total: O(N + N) = O(N)
  - **Key insight**: Each element is visited at most twice (once by right, once by left)

**Space Complexity:**
- **Without auxiliary structure**: O(1)
  - Only variables (pointers, sums, counters)

- **With hash map/set**: O(K)
  - K = size of window or character set
  - Example: At most 26 characters for lowercase English letters → O(26) = O(1)
  - Example: Window of size K → O(K)

**Comparison with Brute Force:**

| Approach | Time | Space | Notes |
|----------|------|-------|-------|
| Brute Force | O(N²) or O(N*K) | O(1) | Check every subarray |
| Sliding Window | O(N) | O(1) to O(K) | Optimal for contiguous problems |

**Example: Maximum Sum of Size K**
- Brute Force: O(N*K) - recalculate sum for each window
- Sliding Window: O(N) - reuse previous sum, add/remove one element

**Example: Longest Substring Without Repeating**
- Brute Force: O(N³) - check all substrings for duplicates
- Sliding Window: O(N) - single pass with hash set

**Why O(N) for Variable Window?**
\`\`\`python
left = 0
for right in range(len(arr)):  # N iterations
    # ... add arr[right]
    
    while condition:  # How many times does this run?
        # ... remove arr[left]
        left += 1
\`\`\`

**Key Insight:** Although there's a nested while loop, left can only move from 0 to N-1 throughout the entire algorithm. So:
- Outer loop: N iterations
- Inner loop: N iterations TOTAL (not per outer iteration)
- **Total: O(N + N) = O(N)**`,
    },
    {
      id: 'templates',
      title: 'Code Templates',
      content: `**Template 1: Fixed-Size Window**
\`\`\`python
def fixed_window(arr: List[int], k: int) -> int:
    """
    Generic fixed-size sliding window.
    Adjust the logic for your specific problem.
    """
    if len(arr) < k:
        return 0  # or appropriate default
    
    # Initialize window with first k elements
    window_sum = sum(arr[:k])  # or other initialization
    result = window_sum
    
    # Slide the window
    for i in range(k, len(arr)):
        # Add new element on the right
        window_sum += arr[i]
        
        # Remove old element on the left
        window_sum -= arr[i - k]
        
        # Update result
        result = max(result, window_sum)  # or min, etc.
    
    return result
\`\`\`

**Template 2: Variable Window - Find Maximum/Longest**
\`\`\`python
def variable_window_max(arr: List[int]) -> int:
    """
    Variable-size window to find maximum/longest.
    Expand when invalid, track max when valid.
    """
    left = 0
    max_length = 0
    window_state = {}  # Hash map/set to track window
    
    for right in range(len(arr)):
        # Add arr[right] to window
        # Update window_state
        
        # Shrink window while condition violated
        while window_violates_condition(window_state):
            # Remove arr[left] from window
            # Update window_state
            left += 1
        
        # Update result (window is now valid)
        max_length = max(max_length, right - left + 1)
    
    return max_length
\`\`\`

**Template 3: Variable Window - Find Minimum/Shortest**
\`\`\`python
def variable_window_min(arr: List[int], target) -> int:
    """
    Variable-size window to find minimum/shortest.
    Expand until valid, then shrink to find minimum.
    """
    left = 0
    min_length = float('inf')
    window_state = {}
    
    for right in range(len(arr)):
        # Add arr[right] to window
        # Update window_state
        
        # Shrink window while condition MET
        while window_meets_condition(window_state, target):
            # Update result (current window is valid)
            min_length = min(min_length, right - left + 1)
            
            # Remove arr[left] from window
            # Update window_state
            left += 1
    
    return min_length if min_length != float('inf') else 0
\`\`\`

**Template 4: Sliding Window with Frequency Map**
\`\`\`python
from collections import defaultdict

def sliding_window_with_freq(s: str) -> int:
    """
    Sliding window tracking character frequencies.
    """
    left = 0
    max_length = 0
    freq = defaultdict(int)
    
    for right in range(len(s)):
        # Add character to window
        freq[s[right]] += 1
        
        # Shrink if condition violated
        while condition_violated(freq):
            freq[s[left]] -= 1
            if freq[s[left]] == 0:
                del freq[s[left]]  # Clean up
            left += 1
        
        max_length = max(max_length, right - left + 1)
    
    return max_length
\`\`\`

**Template 5: Sliding Window with Two Pointers (Distinct Elements)**
\`\`\`python
def sliding_window_set(arr: List[int], k: int) -> int:
    """
    Track unique elements using a set.
    """
    left = 0
    window = set()
    
    for right in range(len(arr)):
        # Add to window, handle duplicates
        while arr[right] in window:
            window.remove(arr[left])
            left += 1
        
        window.add(arr[right])
        
        # Process window
        if len(window) == k:
            # Found a valid window
            pass
    
    return result
\`\`\``,
    },
    {
      id: 'advanced',
      title: 'Advanced Techniques',
      content: `**Technique 1: Sliding Window Maximum (Monotonic Deque)**

Use a deque to track the maximum in each window efficiently:

\`\`\`python
from collections import deque

def sliding_window_maximum(nums: List[int], k: int) -> List[int]:
    """
    Find maximum in each window of size k.
    Time: O(N), Space: O(K)
    """
    result = []
    dq = deque()  # Store indices of useful elements
    
    for i in range(len(nums)):
        # Remove indices outside current window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Remove smaller elements (they'll never be max)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # Add to result once window is full
        if i >= k - 1:
            result.append(nums[dq[0]])  # Front is the maximum
    
    return result
\`\`\`

**Why this works:**
- Deque maintains indices in decreasing order of values
- Front always has the maximum
- Remove elements that can't be maximum (smaller + earlier)

---

**Technique 2: Sliding Window with Multiple Conditions**

Track multiple constraints simultaneously:

\`\`\`python
def longest_substring_k_distinct(s: str, k: int) -> int:
    """
    Longest substring with at most k distinct characters.
    """
    left = 0
    max_length = 0
    char_count = {}
    
    for right in range(len(s)):
        char_count[s[right]] = char_count.get(s[right], 0) + 1
        
        # Shrink if too many distinct characters
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1
        
        max_length = max(max_length, right - left + 1)
    
    return max_length
\`\`\`

---

**Technique 3: Caterpillar/Two-Pointer Variant**

Sometimes the window doesn't always move forward:

\`\`\`python
def subarray_sum_equals_k(nums: List[int], k: int) -> int:
    """
    Count subarrays with sum equal to k.
    Note: This uses prefix sum, not pure sliding window.
    """
    from collections import defaultdict
    
    count = 0
    prefix_sum = 0
    sum_count = defaultdict(int)
    sum_count[0] = 1  # Empty prefix
    
    for num in nums:
        prefix_sum += num
        # Check if (prefix_sum - k) exists
        count += sum_count[prefix_sum - k]
        sum_count[prefix_sum] += 1
    
    return count
\`\`\`

---

**Technique 4: Sliding Window on Two Arrays**

\`\`\`python
def find_anagram_indices(s: str, p: str) -> List[int]:
    """
    Find all starting indices of p's anagrams in s.
    """
    from collections import Counter
    
    result = []
    p_count = Counter(p)
    window_count = Counter()
    
    for i in range(len(s)):
        # Add character to window
        window_count[s[i]] += 1
        
        # Remove character outside window
        if i >= len(p):
            if window_count[s[i - len(p)]] == 1:
                del window_count[s[i - len(p)]]
            else:
                window_count[s[i - len(p)]] -= 1
        
        # Check if window is an anagram
        if window_count == p_count:
            result.append(i - len(p) + 1)
    
    return result
\`\`\``,
    },
    {
      id: 'common-pitfalls',
      title: 'Common Pitfalls',
      content: `**Pitfall 1: Off-by-One Errors in Window Size**

❌ **Wrong:**
\`\`\`python
# Window size is right - left, missing the +1
max_length = max(max_length, right - left)
\`\`\`

✅ **Correct:**
\`\`\`python
# Window size is right - left + 1 (inclusive)
max_length = max(max_length, right - left + 1)
\`\`\`

---

**Pitfall 2: Not Cleaning Up Hash Map**

❌ **Wrong:**
\`\`\`python
freq[s[left]] -= 1
left += 1
# Leaves 0 counts in map, affecting len(freq)
\`\`\`

✅ **Correct:**
\`\`\`python
freq[s[left]] -= 1
if freq[s[left]] == 0:
    del freq[s[left]]  # Clean up to maintain accurate distinct count
left += 1
\`\`\`

---

**Pitfall 3: Forgetting to Initialize First Window**

❌ **Wrong (Fixed Window):**
\`\`\`python
for i in range(len(arr)):  # Starts from 0, recalculating
    window_sum = sum(arr[i:i+k])
\`\`\`

✅ **Correct:**
\`\`\`python
window_sum = sum(arr[:k])  # Calculate first window once
for i in range(k, len(arr)):  # Start sliding from index k
    window_sum += arr[i] - arr[i-k]
\`\`\`

---

**Pitfall 4: Moving Both Pointers in Same Iteration**

❌ **Wrong:**
\`\`\`python
for right in range(len(arr)):
    # Add arr[right]
    while invalid:
        left += 1
    right += 1  # Don't manually increment right!
\`\`\`

✅ **Correct:**
\`\`\`python
for right in range(len(arr)):  # for loop handles right
    # Add arr[right]
    while invalid:
        left += 1  # Only manually move left
\`\`\`

---

**Pitfall 5: Using Wrong Condition for Min vs Max Window**

**For Maximum/Longest** (want largest valid window):
\`\`\`python
while window_is_INVALID:  # Shrink when invalid
    left += 1
max_length = max(max_length, right - left + 1)  # Update outside while
\`\`\`

**For Minimum/Shortest** (want smallest valid window):
\`\`\`python
while window_is_VALID:  # Keep shrinking while still valid
    min_length = min(min_length, right - left + 1)  # Update inside while
    left += 1
\`\`\``,
    },
    {
      id: 'interview',
      title: 'Interview Strategy',
      content: `**Recognition Signals:**

**Use Sliding Window when you see:**
- "Contiguous" subarray/substring
- "Consecutive" elements
- "Window" explicitly mentioned
- "Longest"/"Shortest"/"Maximum"/"Minimum" with constraints
- "At most K" or "At least K" distinct/same elements
- Array or string traversal problems
- Can optimize from O(N²) to O(N)

---

**Problem-Solving Steps:**

**Step 1: Identify Window Type**
- **Fixed size?** → Use Template 1 (add right, remove left)
- **Variable size?** → Use Template 2 or 3 (adjust left based on condition)

**Step 2: Determine Objective**
- **Maximum/Longest?** → Shrink when invalid, update outside while loop
- **Minimum/Shortest?** → Shrink while valid, update inside while loop
- **Count/Existence?** → Check condition at each step

**Step 3: Choose Data Structure**
- **Need frequencies?** → Hash map or Counter
- **Need uniqueness?** → Set
- **Need order/maximum?** → Deque (monotonic queue)
- **Simple sum/count?** → Variables only

**Step 4: Define Validity Condition**
What makes a window valid or invalid?
- "No repeating characters" → Set size equals window size
- "At most K distinct" → len(freq_map) <= K
- "Sum equals target" → current_sum == target
- "Contains all of T" → All chars in T are in window with sufficient count

**Step 5: Handle Edge Cases**
- Empty input
- K > length of array
- All elements same
- Single element array

---

**Interview Communication:**

1. **Identify pattern:** "This is a sliding window problem because we're looking for contiguous elements."

2. **Choose approach:** "I'll use a variable-size window with a hash set to track distinct characters."

3. **Explain invariant:** "The window will always contain at most K distinct characters."

4. **Walk through example:**
   \`\`\`
   s = "eceba", k = 2
   "e"     → 1 distinct, valid
   "ec"    → 2 distinct, valid, length = 2
   "ece"   → 2 distinct, valid, length = 3 ← max
   "eceb"  → 3 distinct, invalid → shrink to "ceb"
   \`\`\`

5. **Discuss complexity:** "Time O(N) since each element is visited at most twice. Space O(K) for the hash map."

---

**Common Follow-ups:**

**Q: Can you solve it with constant space?**
- If character set is limited (e.g., 26 letters), use array instead of hash map: O(1) space

**Q: What if we need to track the actual substring/subarray?**
- Store indices: \`result = (left, right)\`
- Return: \`s[result[0]:result[1]+1]\`

**Q: How would you modify this for "at least K"?**
- Reverse the condition: shrink while count >= K

**Q: Can this be parallelized?**
- Sliding window is inherently sequential, but can divide array into chunks for approximate solutions

---

**Practice Plan:**

1. **Fixed Window (Day 1-2):**
   - Maximum Sum Subarray of Size K
   - Maximum Average Subarray

2. **Variable Window - Maximum (Day 3-4):**
   - Longest Substring Without Repeating Characters
   - Longest Substring with At Most K Distinct Characters

3. **Variable Window - Minimum (Day 5-6):**
   - Minimum Window Substring
   - Minimum Size Subarray Sum

4. **Advanced (Day 7):**
   - Sliding Window Maximum
   - Permutation in String
   - Find All Anagrams

5. **Resources:**
   - LeetCode Sliding Window tag
   - Practice until you can identify the pattern instantly`,
    },
  ],
  keyTakeaways: [
    'Sliding window optimizes O(N²) brute force to O(N) for contiguous sequence problems',
    'Fixed-size window: maintain window size K by adding right, removing left at position (i-K)',
    'Variable-size window: expand right to add elements, shrink left when condition violated',
    'For maximum/longest: shrink when invalid, update result when valid',
    'For minimum/shortest: shrink while still valid, update result during shrinking',
    'Use hash map to track frequencies, set for uniqueness, deque for maximum/minimum',
    'Time complexity is O(N) because each element visited at most twice (once by each pointer)',
    'Window size formula: right - left + 1 (inclusive of both endpoints)',
  ],
  relatedProblems: [
    'best-time-to-buy-sell-stock',
    'longest-substring-without-repeating',
    'minimum-window-substring',
  ],
};
