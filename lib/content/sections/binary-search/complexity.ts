/**
 * Time & Space Complexity Analysis Section
 */

export const complexitySection = {
  id: 'complexity',
  title: 'Time & Space Complexity Analysis',
  content: `**Time Complexity: O(log n)**

**Why Logarithmic?**
- Start with n elements
- After 1 comparison: n/2 elements remain
- After 2 comparisons: n/4 elements remain
- After 3 comparisons: n/8 elements remain
- After k comparisons: n/2^k elements remain

When n/2^k = 1, we've found our answer: k = log₂(n)

**Concrete Examples:**
- **10 elements:** max 4 comparisons (2^4 = 16)
- **100 elements:** max 7 comparisons (2^7 = 128)
- **1,000 elements:** max 10 comparisons (2^10 = 1,024)
- **1,000,000 elements:** max 20 comparisons (2^20 = 1,048,576)
- **1,000,000,000 elements:** max 30 comparisons!

**Comparison with Linear Search:**

| Array Size | Linear Search | Binary Search | Speedup |
|------------|--------------|---------------|---------|
| 100        | 100          | 7             | 14x     |
| 10,000     | 10,000       | 14            | 714x    |
| 1,000,000  | 1,000,000    | 20            | 50,000x |

**Space Complexity: O(1)**
- Iterative version uses constant space (just a few variables)
- Recursive version uses O(log n) space for the call stack

**Best, Average, Worst Cases:**
- **Best Case:** O(1) - target is at the middle
- **Average Case:** O(log n) - typical scenario
- **Worst Case:** O(log n) - target at an end or not present

The consistency of performance is a major advantage!`,
};
