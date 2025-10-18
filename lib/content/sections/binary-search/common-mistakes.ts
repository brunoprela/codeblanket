/**
 * Common Pitfalls & How to Avoid Them Section
 */

export const commonmistakesSection = {
  id: 'common-mistakes',
  title: 'Common Pitfalls & How to Avoid Them',
  content: `**1. Integer Overflow (Critical in Some Languages)**

❌ **Wrong:**
\`\`\`python
mid = (left + right) // 2  # Can overflow in Java/C++
\`\`\`

✅ **Correct:**
\`\`\`python
mid = left + (right - left) // 2  # Safe from overflow
\`\`\`

**Why:** In languages with fixed integer sizes, \`left + right\` can exceed max integer value.

**2. Incorrect Loop Condition**

❌ **Wrong:**
\`\`\`python
while left < right:  # Will miss single element case
\`\`\`

✅ **Correct:**
\`\`\`python
while left <= right:  # Handles all cases correctly
\`\`\`

**Why:** When \`left == right\`, we still need to check that element.

**3. Off-by-One Errors in Pointer Updates**

❌ **Wrong:**
\`\`\`python
left = mid  # Can cause infinite loop!
right = mid  # Can cause infinite loop!
\`\`\`

✅ **Correct:**
\`\`\`python
left = mid + 1  # Properly excludes mid
right = mid - 1  # Properly excludes mid
\`\`\`

**Why:** Using \`mid\` directly can create infinite loops when the search space reduces to 2 elements.

**4. Forgetting to Check if Array is Sorted**

Always verify the precondition! If the array isn't sorted, binary search will give incorrect results.

**5. Using Binary Search on Unsorted Data**

Binary search ONLY works on sorted data. For unsorted data:
- Sort first (O(n log n)), then search
- Or use linear search (O(n))
- Or use hash table (O(1) average lookup)

**6. Returning Wrong Value**

Make sure you return:
- The index (not the value) when found
- -1 or appropriate sentinel when not found
- The correct boundary for "find first/last" variants

**Debugging Tips:**
- Print \`left\`, \`mid\`, \`right\` in each iteration
- Verify the search space is shrinking
- Check boundary conditions: empty array, single element, target at ends
- Test with duplicates if applicable`,
};
