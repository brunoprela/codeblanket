/**
 * Sliding Window Patterns Section
 */

export const patternsSection = {
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
};
