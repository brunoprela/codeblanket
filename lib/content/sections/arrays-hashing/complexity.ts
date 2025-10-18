/**
 * Time & Space Complexity Analysis Section
 */

export const complexitySection = {
  id: 'complexity',
  title: 'Time & Space Complexity Analysis',
  content: `**Array Operations:**

| Operation | Time | Notes |
|-----------|------|-------|
| Access by index | O(1) | Direct memory access |
| Search (unsorted) | O(n) | Must check each element |
| Search (sorted) | O(log n) | Binary search |
| Insert/Delete at end | O(1)* | *Amortized with dynamic arrays |
| Insert/Delete at start | O(n) | Requires shifting all elements |
| Insert/Delete in middle | O(n) | Requires shifting elements |

**Hash Table Operations:**

| Operation | Average | Worst | Notes |
|-----------|---------|-------|-------|
| Insert | O(1) | O(n) | Worst case with many collisions |
| Delete | O(1) | O(n) | Same as insert |
| Search | O(1) | O(n) | Usually much faster than arrays |
| Iteration | O(n) | O(n) | Must visit all elements |

**Space Complexity:**
- **Array:** O(n) for n elements
- **Hash Table:** O(n) plus overhead for hash structure
- **Trade-off:** Hash tables use more memory but provide faster operations

**When to Use Each:**

**Use Arrays when:**
- Need ordered/indexed access
- Memory is limited
- Elements are accessed by position
- Implementing other data structures

**Use Hash Tables when:**
- Need fast lookups by key
- Counting frequencies
- Detecting duplicates
- Grouping elements
- Caching results

**Optimization Example:**
Finding if array has duplicate:
- Brute force (nested loops): O(n²)
- Sorting then scanning: O(n log n)
- Using hash set: O(n) ✅ Best!`,
};
