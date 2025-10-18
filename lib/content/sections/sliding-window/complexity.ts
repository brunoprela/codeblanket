/**
 * Complexity Analysis Section
 */

export const complexitySection = {
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
};
