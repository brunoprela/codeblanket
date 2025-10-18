/**
 * Introduction to Segment Trees Section
 */

export const introductionSection = {
  id: 'introduction',
  title: 'Introduction to Segment Trees',
  content: `**Segment Tree** is a tree data structure for storing intervals (segments). It allows efficient querying and updating of array intervals.

**Problem it Solves:**
Given an array, perform these operations efficiently:
- **Range Query**: Find sum/min/max/gcd of elements in range [L, R]
- **Point Update**: Update a single element
- **Range Update**: Update all elements in a range (with lazy propagation)

**Why Segment Trees:**
- Range queries: O(log N) instead of O(N)
- Point updates: O(log N) instead of O(1) simple array
- Balance between query and update efficiency

**When to Use:**
- Multiple range queries on static or dynamic arrays
- Need both queries and updates
- Range sum, min, max, GCD, etc.
- Problems with intervals/segments

**Alternative Data Structures:**
- **Prefix Sum**: Fast queries (O(1)) but can't handle updates
- **Fenwick Tree**: Simpler, but limited to associative operations
- **Sqrt Decomposition**: Simpler but slower (O(√N))`,
};
