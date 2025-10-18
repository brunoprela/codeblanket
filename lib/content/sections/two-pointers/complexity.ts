/**
 * Time & Space Complexity Analysis Section
 */

export const complexitySection = {
  id: 'complexity',
  title: 'Time & Space Complexity Analysis',
  content: `**Time Complexity: O(n)**

**Why Linear Time?**
- Each pointer moves through the array at most once
- In opposite direction pattern: combined movement covers array once
- In same direction pattern: fast pointer covers array once
- No nested loops required!

**Comparison with Brute Force:**

| Approach | Time | Space | Array Size 1000 |
|----------|------|-------|-----------------|
| Brute Force (nested loops) | O(n²) | O(1) | 1,000,000 ops |
| Two Pointers | O(n) | O(1) | 1,000 ops |

**The Difference is Massive:**
- For n = 1,000: 1000x faster
- For n = 10,000: 10,000x faster
- For n = 100,000: 100,000x faster

**Space Complexity: O(1)**
- Only use two pointer variables
- Modify array in-place when needed
- No additional data structures
- Constant extra memory regardless of input size

**Detailed Analysis by Pattern:**

**1. Opposite Direction:**
- Time: O(n) - each element visited once
- Space: O(1) - two pointer variables
- Pointers move total of n times combined

**2. Same Direction:**
- Time: O(n) - fast pointer goes through array once
- Space: O(1) - two pointer variables
- Slow pointer always behind fast pointer

**3. Sliding Window:**
- Time: O(n) - each element enters and leaves window once
- Space: O(1) for basic window, O(k) if storing window elements
- Amortized O(1) per element

**When Complexity Increases:**
- If operation at each step is expensive: O(n × k)
- If storing window elements: Space becomes O(window_size)
- If sorting is needed first: Time becomes O(n log n)

**Practical Performance:**
Beyond Big-O notation:
- Cache-friendly: sequential access pattern
- Low overhead: minimal extra variables
- Parallelizable: in some cases
- In-place: doesn't require extra memory`,
};
