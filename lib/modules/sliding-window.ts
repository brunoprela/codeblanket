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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain the core concept of the sliding window technique. Why is it called a "window" and what makes it "slide"?',
          sampleAnswer:
            'The sliding window technique uses two pointers to define a contiguous subarray or substring - that is the "window". We start with both pointers at the beginning, then expand the window by moving the right pointer to include more elements. When certain conditions are met, we shrink the window by moving the left pointer. This creates a sliding motion - the window slides across the array or string. It is powerful because instead of checking all possible subarrays O(n²), we maintain one window and adjust it as we go in O(n) time. The key is that we process each element at most twice - once when entering the window, once when leaving.',
          keyPoints: [
            'Two pointers define a contiguous subarray/substring',
            'Expand: move right pointer',
            'Shrink: move left pointer',
            'Sliding motion across data',
            'O(n) vs O(n²) for all subarrays',
          ],
        },
        {
          id: 'q2',
          question:
            'Compare fixed-size vs variable-size windows. When would you use each approach?',
          sampleAnswer:
            'Fixed-size windows have a predetermined size k - like max sum of k consecutive elements. I expand until reaching size k, then slide by adding right and removing left simultaneously. This is straightforward and great when the problem specifies exact window size. Variable-size windows adjust based on conditions - like longest substring without repeating characters. I expand right to include elements while condition holds, then shrink left when condition breaks. This is more complex but handles optimization problems where we seek maximum or minimum window satisfying constraints. Fixed-size: problem gives exact size. Variable-size: we optimize to find best size.',
          keyPoints: [
            'Fixed: predetermined size k',
            'Fixed: slide by add right, remove left',
            'Variable: adjust based on conditions',
            'Variable: expand and shrink as needed',
            'Fixed: exact size given, Variable: optimize size',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk me through a simple example of sliding window solving a problem in O(n) that would be O(n²) with brute force.',
          sampleAnswer:
            'Take max sum of k=3 consecutive elements in [2, 1, 5, 1, 3, 2]. Brute force computes sum of every 3-element window: [2,1,5]=8, [1,5,1]=7, [5,1,3]=9, [1,3,2]=6. That is 4 windows each taking 3 adds, total 12 operations. Sliding window: compute first window [2,1,5]=8. Then slide: remove 2, add 1 → [1,5,1]=7. Remove 1, add 3 → [5,1,3]=9. Remove 5, add 2 → [1,3,2]=6. Each slide is 2 operations, total 3+6=9 operations. For large n, brute force is O(n×k) vs sliding window O(n). We reuse previous sum instead of recalculating from scratch each time.',
          keyPoints: [
            'Brute force: recalculate each window',
            'Sliding window: reuse previous calculation',
            'Remove leaving element, add entering element',
            'Brute force: O(n×k), Sliding window: O(n)',
            'Efficiency from incremental updates',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Describe the shrinkable window pattern. When do you expand and when do you shrink?',
          sampleAnswer:
            'In the shrinkable window pattern, I expand by moving right pointer to include more elements, making the window larger. I shrink by moving left pointer when the window violates some condition. For longest substring without repeating characters, I expand right to add characters, but when I hit a duplicate, I shrink left until removing the previous occurrence of that character. The key pattern: expand right in the outer loop unconditionally, shrink left in an inner while loop when condition breaks. This maintains the invariant that the window always satisfies our constraints. The answer is typically the maximum window size seen during expansion.',
          keyPoints: [
            'Expand right: add elements unconditionally',
            'Shrink left: while condition violated',
            'Outer loop: right pointer',
            'Inner while: left pointer',
            'Answer: maximum window size',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the non-shrinkable window pattern and how it differs from shrinkable. When would you use each?',
          sampleAnswer:
            'Non-shrinkable window maintains maximum window size once achieved and never shrinks below it. Instead of a while loop to shrink, I use an if statement to move left just once. This keeps window size at least as large as the best seen so far. For example, in longest substring with at most k distinct characters, once I have found a window of size 5, I never let it shrink below 5 - I just slide it forward looking for larger valid windows. Use non-shrinkable when you want maximum window satisfying constraints and do not need to track all valid windows. Use shrinkable when you need minimum window or need to process all valid windows.',
          keyPoints: [
            'Maintains maximum window size achieved',
            'If statement instead of while to move left',
            'Never shrinks below best size',
            'Slides forward at fixed size',
            'Use for maximum window problems',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk through the minimum window substring problem. How does the sliding window with character counting work?',
          sampleAnswer:
            'For minimum window containing all characters of a target string, I use two hash maps: one for target character counts and one for current window counts. I expand right, adding characters to window count. When window contains all target characters (use a counter to check), I shrink left to minimize window while maintaining validity. For each valid window, I track the minimum. The trick is efficiently checking validity - I maintain a "formed" counter that tracks how many unique characters have reached their required count. When formed equals number of unique chars in target, window is valid. This avoids checking all characters each time.',
          keyPoints: [
            'Two hash maps: target counts and window counts',
            'Expand right: add to window',
            'Shrink left: while window valid',
            '"formed" counter for efficient validity check',
            'Track minimum valid window',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain why sliding window is O(n) even though there is a nested loop. How does amortized analysis apply here?',
          sampleAnswer:
            'Sliding window looks like O(n²) with nested loops - outer loop moves right pointer n times, inner loop moves left pointer. But amortized analysis shows it is O(n). The key insight: left pointer moves at most n times total across the entire algorithm, not n times per right pointer iteration. Each element enters the window once (right pointer) and leaves at most once (left pointer), giving 2n operations total. So outer loop contributes O(n), inner loop across all iterations contributes O(n), total is O(2n) = O(n). The inner loop iterations are amortized - distributed across all outer iterations, not concentrated.',
          keyPoints: [
            'Looks like nested loops: O(n²)?',
            'Left pointer moves n times TOTAL',
            'Each element enters once, leaves once',
            '2n operations total: O(n)',
            'Amortized: inner loop cost distributed',
          ],
        },
        {
          id: 'q2',
          question:
            'Compare the space complexity of different sliding window problems. When do you need extra space and when can you solve in O(1)?',
          sampleAnswer:
            'Space complexity depends on what you track in the window. For simple sum or count, O(1) space - just variables for left, right, sum. For character or element frequency, O(k) space where k is alphabet size or unique elements - need hash map to count. For subarray problems tracking indices or elements, potentially O(n) if storing all elements in window. Fixed-size numeric windows can be O(1). Variable-size windows with character constraints need O(k) for hash map. The question to ask: what information must I maintain about window contents? Minimal tracking enables O(1), frequency tracking needs O(k).',
          keyPoints: [
            'Simple sum/count: O(1) space',
            'Character frequency: O(k) for hash map',
            'Tracking elements: O(n) worst case',
            'Fixed numeric windows: O(1)',
            'Depends on what you track in window',
          ],
        },
        {
          id: 'q3',
          question:
            'Why is it critical to process the right pointer before checking conditions? Walk me through what happens if you do it wrong.',
          sampleAnswer:
            'Processing right pointer first means adding the element to the window before checking if conditions break. This is correct because we want to include the element, then determine if we need to shrink. If I check conditions before processing right, I am checking the window without the new element, which is the previous window state. For example, in longest substring without repeating, if I check for duplicate before adding current character, I miss that current character might be the duplicate. Then I add it anyway, leaving duplicates in window. Correct order: add to window, update state, check conditions, shrink if needed. This maintains the invariant that we process each position exactly once.',
          keyPoints: [
            'Add element first, then check conditions',
            'Checking before adding tests previous window',
            'Wrong order: might miss violations',
            'Correct: add, update, check, shrink',
            'Maintains invariant: each position processed once',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Walk me through the fixed-size window template. What is the key pattern for initialization and sliding?',
          sampleAnswer:
            'For fixed-size windows of size k, the template is: first, build the initial window by iterating 0 to k-1 and computing the initial sum or state. Second, start sliding from index k: for each new position, add the element entering at right and subtract the element leaving at left. Update the answer if current window is better. The key pattern: one loop to build initial window, one loop starting at k to slide. Each slide is constant time: subtract left, add right, check answer. This avoids recalculating the entire window each time. Common for problems like max sum of k elements, average of k elements, or any fixed-size aggregate.',
          keyPoints: [
            'Build initial window: loop 0 to k-1',
            'Slide from index k onward',
            'Each slide: subtract left, add right',
            'Update answer each iteration',
            'Avoid recalculating entire window',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the variable-size shrinkable window template. What goes in the outer loop vs the inner while loop?',
          sampleAnswer:
            'The shrinkable template has right pointer in outer for loop, left pointer in inner while loop. Outer loop: for right in range(n), add arr[right] to window, update state. Inner while loop: while condition violated, remove arr[left] from window, increment left. After inner while, check and update answer using current window. The key structure: unconditionally expand right, conditionally shrink left while needed. This ensures we explore all possible windows and the inner while maintains validity. Used for maximum window problems where we want largest valid window, like longest substring without repeating characters or max consecutive ones after k flips.',
          keyPoints: [
            'Outer for: right pointer, expand',
            'Inner while: left pointer, shrink',
            'Add right unconditionally',
            'Shrink left while condition breaks',
            'Update answer after shrinking',
          ],
        },
        {
          id: 'q3',
          question:
            'Compare the shrinkable vs non-shrinkable templates. How does changing while to if affect the solution?',
          sampleAnswer:
            'Shrinkable uses "while condition violated" to move left - shrinks as much as needed to restore validity. Non-shrinkable uses "if condition violated" to move left once - maintains max window size achieved. For example, longest substring with k distinct: shrinkable shrinks fully when exceeding k distinct, exploring all valid windows. Non-shrinkable moves left just once when exceeding k, keeping window size at maximum found so far and just sliding forward. Shrinkable finds exact maximum and tracks all valid windows. Non-shrinkable is optimization that works when we only need final maximum size and can skip smaller valid windows. While gives thorough exploration, if gives efficient maximum tracking.',
          keyPoints: [
            'While: shrink fully, restore validity',
            'If: move once, maintain max size',
            'Shrinkable: explore all valid windows',
            'Non-shrinkable: track maximum, skip smaller',
            'While for thorough, if for optimization',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain the monotonic deque technique for sliding window maximum. Why use a deque instead of just tracking the max?',
          sampleAnswer:
            'For sliding window maximum, just tracking current max fails when the max element leaves the window - we do not know the next maximum. Monotonic deque solves this by maintaining indices in decreasing order of their values. When a new element enters, we remove elements from the back that are smaller than it (they can never be maximum while the new element is in window). When elements leave the window, we remove from front if their index is outside window. The front always has the maximum for current window. This gives O(n) total time because each element is added and removed from deque at most once. Deque enables both ends operations in O(1).',
          keyPoints: [
            'Need to track next maximum when current leaves',
            'Deque maintains indices in decreasing value order',
            'Remove smaller elements from back',
            'Remove out-of-window indices from front',
            'O(n) total: each element in/out once',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through sliding window with multiple conditions. How do you track vowels and consonants simultaneously?',
          sampleAnswer:
            'For multiple conditions like k vowels and m consonants in window, I track both counts separately using variables or a hash map. As I expand right, I check if the new character is vowel or consonant and update respective count. When shrinking left, I decrement the appropriate count. The window is valid when both conditions are satisfied simultaneously. I check "if vowels == k and consonants == m" to know validity. The key is maintaining independent counters for each condition and checking all conditions together. This extends to any number of simultaneous constraints - just add more counters and check all conditions in your validity check.',
          keyPoints: [
            'Track each condition with separate counter',
            'Update appropriate counter on add/remove',
            'Check all conditions together for validity',
            'Independent counters for each constraint',
            'Extends to any number of conditions',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe the pattern for finding all anagrams using sliding window. What makes this more efficient than checking each substring?',
          sampleAnswer:
            'For finding all anagrams of pattern in string, I use fixed-size window of length equal to pattern length. I maintain two frequency maps: one for pattern (built once), one for current window (updated as window slides). I slide through the string, adding the entering character and removing the leaving character from window map. After each slide, I compare window map with pattern map - if equal, found an anagram. This is O(n) because each comparison is O(26) for alphabet size, constant time. Brute force would generate and sort each substring: O(n × m log m) where m is pattern length. Sliding window avoids repeated sorting by maintaining frequency incrementally.',
          keyPoints: [
            'Fixed window size = pattern length',
            'Two frequency maps: pattern and window',
            'Slide: update window map incrementally',
            'Compare maps: O(26) constant time',
            'O(n) vs O(n × m log m) brute force',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain the off-by-one error with window size calculation. Why is it right - left + 1 and not right - left?',
          sampleAnswer:
            'Window size is right - left + 1 because indices are inclusive. If left = 2 and right = 5, the window contains indices 2, 3, 4, 5 which is 4 elements. If I just do right - left, I get 5 - 2 = 3, missing one element. The +1 accounts for the element at the left index itself. Another way to think: if left equals right, window has one element, so size should be 1, not 0. Common mistake is writing right - left and wondering why all answers are off by one. This is similar to calculating array length from indices: end - start + 1 when both ends inclusive.',
          keyPoints: [
            'Indices are inclusive: need +1',
            'Example: left=2, right=5 → 4 elements',
            'right - left gives 3, wrong',
            'right - left + 1 gives 4, correct',
            'When left equals right: size is 1, not 0',
          ],
        },
        {
          id: 'q2',
          question:
            'What is the difference between checking conditions before vs after processing right pointer? Why does order matter?',
          sampleAnswer:
            'Processing right pointer means adding arr[right] to window state. I should do this before checking conditions because I want to check the window that includes the new element. If I check conditions first, I am testing the old window without the new element, which is wrong. For example, in longest substring without repeating, if I check for duplicates before adding current character, I miss detecting if current character itself is the duplicate. The correct sequence: add arr[right] to window, update state, check if conditions violated, shrink left if needed. This ensures each element is properly included in the window before evaluation.',
          keyPoints: [
            'Process right: add to window first',
            'Then check conditions on updated window',
            'Checking before: tests old window',
            'Wrong order: miss detecting violations',
            'Correct: add, update, check, shrink',
          ],
        },
        {
          id: 'q3',
          question:
            'Compare the logic for maximum vs minimum window problems. Where do you update the answer and why?',
          sampleAnswer:
            'For maximum window problems, I shrink while window is invalid, then update answer outside the while loop with right - left + 1. This captures the maximum valid window at each position. For minimum window problems, I shrink while window is still valid, updating answer inside the while loop before each shrink. This captures progressively smaller valid windows. The key difference: maximum wants largest valid so update after restoring validity. Minimum wants smallest valid so update while shrinking valid window. If I use wrong placement, I either miss the optimal or include invalid windows. The while condition and update placement must match the optimization goal.',
          keyPoints: [
            'Maximum: shrink while invalid, update outside while',
            'Minimum: shrink while valid, update inside while',
            'Maximum: largest valid window',
            'Minimum: smallest valid window',
            'Condition and update placement must match goal',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'How do you recognize in an interview that a problem needs sliding window? What keywords or patterns signal this?',
          sampleAnswer:
            'Several signals tell me sliding window. First, keywords: "contiguous", "subarray", "substring", "consecutive" - these scream window. Second, optimization terms: "longest", "shortest", "maximum", "minimum" with constraints - we are seeking optimal window. Third, constraints like "at most k", "at least k" distinct elements - these are window validity conditions. Fourth, if brute force would check all subarrays O(n²) - sliding window likely optimizes to O(n). Fifth, two-pointer vibes with sequential processing. The key question: am I looking for something in a contiguous sequence that can be computed incrementally? If yes, sliding window is probably the answer.',
          keyPoints: [
            'Keywords: contiguous, subarray, substring, consecutive',
            'Optimization: longest, shortest, maximum, minimum',
            'Constraints: at most k, at least k',
            'Brute force: O(n²) checking all subarrays',
            'Incremental computation on contiguous sequence',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through explaining a sliding window solution in an interview, from problem recognition to complexity analysis.',
          sampleAnswer:
            'First, I identify: "This is a longest substring problem with constraints, so I am thinking sliding window". Then I explain window type: "Variable-size window - I will expand right to include characters and shrink left when constraints break". I describe the state: "I will track character frequency using a hash map". Then the algorithm: "Expand right each iteration, update frequency. When I have duplicates, shrink left until removing the duplicate. Track maximum window size." I code while explaining each part. After coding, I trace an example: "For \'abcabcbb\', right moves to \'abc\' len 3, hits duplicate \'a\'...". Finally complexity: "O(n) time as each element enters and leaves once, O(k) space for hash map where k is alphabet size." Communication throughout is key.',
          keyPoints: [
            'Identify problem type and pattern',
            'Explain window type and expansion/shrinking logic',
            'Describe state tracking',
            'Explain algorithm step-by-step while coding',
            'Trace example',
            'State time and space complexity',
          ],
        },
        {
          id: 'q3',
          question:
            'What are the most common mistakes in sliding window problems and how do you avoid them?',
          sampleAnswer:
            'First mistake: off-by-one in window size calculation - use right - left + 1, not right - left. I verify with simple case: if left equals right, size is 1. Second: processing order - must add arr[right] before checking conditions, or we check old window. Third: update answer placement - outside while for maximum, inside while for minimum. I remember: shrink invalid for max, shrink valid for min. Fourth: forgetting to update state when moving left - must remove arr[left] from tracking. Fifth: manually incrementing right in for loop - let the for loop handle it. Sixth: not initializing properly - hash map, variables. I avoid these by using templates and testing edge cases early.',
          keyPoints: [
            'Off-by-one: use right - left + 1',
            'Order: add right before checking',
            'Update placement: depends on max vs min',
            'State: update when moving left',
            'Do not manually increment right in for loop',
            'Use templates and test edge cases',
          ],
        },
      ],
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
