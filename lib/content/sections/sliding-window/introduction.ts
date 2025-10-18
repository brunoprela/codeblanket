/**
 * Introduction to Sliding Window Section
 */

export const introductionSection = {
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
};
