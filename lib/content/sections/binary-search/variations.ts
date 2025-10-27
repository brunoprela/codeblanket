/**
 * Advanced Variations & Applications Section
 */

export const variationsSection = {
  id: 'variations',
  title: 'Advanced Variations & Applications',
  content: `Binary search is incredibly versatile. Once you master the basics, you can apply it to many problems that don't look like traditional search!

**1. Search in Rotated Sorted Array**
**Problem:** Array was sorted, then rotated. Find target.
**Example:** [4,5,6,7,0,1,2], target = 0

**Key Insight:** At least one half is always sorted. Check which half is sorted, then decide where to search.

**2. Find Peak Element**
**Problem:** Find any local maximum in an unsorted array.
**Key Insight:** If mid < mid+1, peak must be on the right. Binary search on the gradient!

**3. Search in 2D Matrix**
**Problem:** Matrix sorted row-wise and column-wise.
**Key Insight:** Treat as 1D array: \`mid = mid // cols, mid % cols\`

**4. Find Minimum in Rotated Sorted Array**
**Problem:** Find the smallest element after rotation.
**Key Insight:** Minimum is at the rotation point. Compare with rightmost element.

**5. Square Root / nth Root**
**Problem:** Find floor (sqrt (x)) without using sqrt function.
**Key Insight:** Binary search on the answer space [0, x].

**6. First Bad Version**
**Problem:** Find first failing version in sequence.
**Key Insight:** Find first True in array of [False, False, ..., True, True].

**Problem-Solving Framework:**1. **Identify if binary search applies:**
   - Is there a sorted order? (explicit or implicit)
   - Can you check a condition in O(1) or O(log n)?
   - Is the answer monotonic? (if x works, all smaller/larger x work too)

2. **Define search space:**
   - What are the minimum and maximum possible answers?
   - What type: indices, values, or abstract space?

3. **Write the condition:**
   - What makes \`mid\` a valid/invalid answer?
   - How do you decide to go left or right?

4. **Handle edge cases:**
   - Empty input
   - Single element
   - All same elements
   - Target at boundaries

**Interview Tip:**
Binary search problems in interviews often hide the "sorted" aspect. Look for:
- "Find first/last..."
- "Minimum/maximum..."
- "At least/at most..."
- Problems with monotonic properties`,
};
