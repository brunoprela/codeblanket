/**
 * Problem-Solving Strategy & Interview Tips Section
 */

export const problemsolvingSection = {
  id: 'problem-solving',
  title: 'Problem-Solving Strategy & Interview Tips',
  content: `**Step-by-Step Approach:**

**1. Clarify Requirements (30 seconds)**
- Ask: "Is the array sorted?" (Critical!)
- Ask: "Can there be duplicates?"
- Ask: "What should I return if not found?"
- Ask: "What\'s the expected size?" (helps choose algorithm)

**2. Explain the Approach (1-2 minutes)**
- State: "I'll use binary search because the array is sorted"
- Explain time complexity: "O(log n) instead of O(n) linear search"
- Mention edge cases you'll handle

**3. Code (5-7 minutes)**
- Start with the template
- Clearly label left, right, mid
- Add comments at decision points
- Don't rush! Accuracy > Speed

**4. Test Your Code (2-3 minutes)**
- **Test case 1:** Target found in middle
- **Test case 2:** Target at boundaries (first/last element)
- **Test case 3:** Target not in array
- **Test case 4:** Single element array
- **Test case 5:** Empty array

**5. Analyze Complexity (30 seconds)**
- Time: O(log n) - explain why
- Space: O(1) iterative, O(log n) recursive

**Common Interview Follow-ups:**
1. "What if there are duplicates?" → Find first/last occurrence
2. "What if it's rotated?" → Modified binary search
3. "Can you do it recursively?" → Show recursive version
4. "What about 2D array?" → Treat as 1D or use 2D binary search

**Optimization Tips:**
- For very large arrays, consider cache-friendly modifications
- For repeated searches, consider preprocessing
- For range queries, consider segment trees or other data structures

**Red Flags to Avoid:**
❌ Not checking if array is sorted
❌ Infinite loops from wrong pointer updates
❌ Forgetting edge cases
❌ Integer overflow in mid calculation
❌ Wrong return value (returning value instead of index)

**How to Practice:**
1. Master the basic template first
2. Solve "find first/last occurrence" problems
3. Try rotated array problems
4. Tackle abstract binary search problems
5. Time yourself - aim for 10-15 minutes per problem

**Remember:**
- Binary search is about **eliminating possibilities**
- The search space **must shrink** every iteration
- When in doubt, **trace through with a small example**
- **Practice** until the template becomes second nature`,
};
