/**
 * Introduction to Fenwick Tree Section
 */

export const introductionSection = {
  id: 'introduction',
  title: 'Introduction to Fenwick Tree',
  content: `**Fenwick Tree** (also called **Binary Indexed Tree** or **BIT**) is a data structure that efficiently calculates prefix sums and handles updates.

**What it Does:**
- **Prefix Sum Query**: Sum of elements from index 0 to i
- **Range Sum Query**: Sum from index L to R (= prefix[R] - prefix[L-1])
- **Point Update**: Add value to element at index i

**Why Fenwick Tree:**
- Simpler than Segment Tree
- Same complexity: O(log N) for both operations
- Less memory: Just an array of size N+1
- Elegant implementation using bit manipulation

**When to Use:**
- Need prefix sums with updates
- Range sum queries
- Cumulative frequency (counting inversions)
- 2D range sums

**Compared to Alternatives:**

| Data Structure | Build | Query | Update | Space |
|----------------|-------|-------|--------|-------|
| Prefix Sum | O(N) | O(1) | O(N) | O(N) |
| Fenwick Tree | O(N log N) | O(log N) | O(log N) | O(N) |
| Segment Tree | O(N) | O(log N) | O(log N) | O(4N) |

**Key Advantage:**
- Simpler to code than Segment Tree
- Works for any operation with an **inverse** (addition/subtraction, XOR)
- Can't do min/max (no inverse operation)`,
};
