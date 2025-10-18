/**
 * Complexity Analysis Section
 */

export const complexitySection = {
  id: 'complexity',
  title: 'Complexity Analysis',
  content: `**Time Complexity Patterns:**

| Operation | Unsorted | Sorted | Notes |
|-----------|----------|--------|-------|
| Merge Intervals | O(n log n) | O(n) | Sorting dominates |
| Insert Interval | O(n log n) | O(n) | Linear if sorted |
| Overlap Check | O(n²) | O(n log n) | Much better sorted |
| Find Free Time | O(n log n) | O(n) | Needs sorting |
| Interval Intersection | O(n²) | O(n+m) | Two pointers |

**Key Insight:** Sorting intervals costs O(n log n) but enables O(n) operations.

---

**Space Complexity:**

**Result Storage:**
- Merging: O(n) worst case (no merges)
- Insertion: O(n)
- Intersection: O(min(n, m))

**Auxiliary Space:**
- Sorting: O(log n) for quicksort
- Sweep line: O(n) for events array

---

**Common Complexities:**

**Merge Intervals:**
- Time: O(n log n) for sort + O(n) for merge = **O(n log n)**
- Space: O(n) for result

**Insert Interval (sorted input):**
- Time: **O(n)**
- Space: O(n)

**Meeting Rooms II:**
- Time: O(n log n) for sorting
- Space: O(n) for events

**Interval Intersection:**
- Time: **O(n + m)** with two pointers
- Space: O(min(n, m)) for result

---

**Optimization Techniques:**

**1. In-place Modification**
If allowed to modify input, can save space:
\`\`\`python
def merge_inplace(intervals):
    intervals.sort()
    write = 0
    for read in range(1, len(intervals)):
        if intervals[read][0] <= intervals[write][1]:
            intervals[write][1] = max(
                intervals[write][1],
                intervals[read][1]
            )
        else:
            write += 1
            intervals[write] = intervals[read]
    return intervals[:write + 1]
\`\`\`

**2. Early Termination**
For insertion, stop once past new interval.

**3. Custom Sorting**
Sort by end time for scheduling problems.`,
};
