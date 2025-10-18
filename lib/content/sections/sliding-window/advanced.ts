/**
 * Advanced Techniques Section
 */

export const advancedSection = {
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
};
