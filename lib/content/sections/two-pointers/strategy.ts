/**
 * Problem-Solving Strategy & Interview Tips Section
 */

export const strategySection = {
  id: 'strategy',
  title: 'Problem-Solving Strategy & Interview Tips',
  content: `**Recognition Patterns:**

Ask yourself these questions to identify two pointer problems:

**1. "Do I need to find pairs/triplets?"**
- If yes and sorted/sortable → Opposite direction pointers

**2. "Do I need to modify array in-place?"**
- Removing elements → Same direction pointers
- Partitioning → Fast/slow pointers

**3. "Am I looking at subarrays/substrings?"**
- Fixed size window → Sliding window with fixed gap
- Variable size window → Expanding/shrinking window

**4. "Is there a two-pass O(n²) solution?"**
- Often can be optimized to O(n) with two pointers

**Step-by-Step Approach:**

**Step 1: Clarify (30 seconds)**
- "Is the input sorted?" (Crucial!)
- "Can I modify the input?"
- "What should I return?"
- "Are there duplicates?"

**Step 2: Choose Pattern (30 seconds)**
- Sorted + pairs? → Opposite direction
- Modify in-place? → Same direction  
- Window problem? → Sliding window

**Step 3: Explain Approach (1-2 minutes)**
- State which pattern you're using
- Explain why it's better than brute force
- Mention time/space complexity

**Step 4: Code (5-7 minutes)**
- Initialize pointers correctly
- Write clear loop condition
- Handle pointer movement logic
- Consider edge cases

**Step 5: Test (2-3 minutes)**
Test these cases:
- Empty/single element input
- All elements same
- Target at boundaries
- No valid answer
- Multiple valid answers

**Common Mistakes to Avoid:**

**1. Wrong Initialization**
❌ \`left, right = 0, len (nums)\` // right should be len (nums) - 1
✅ \`left, right = 0, len (nums) - 1\`

**2. Infinite Loops**
❌ Forgetting to move pointers
✅ Always ensure at least one pointer moves each iteration

**3. Off-by-One Errors**
❌ \`while left <= right\` when should be \`left < right\`
✅ Think carefully about when pointers can be equal

**4. Not Handling Duplicates**
- For 3Sum, need to skip duplicate values
- For unique pairs, need extra checks

**5. Wrong Pointer Movement**
- In container problem, move the shorter height
- In partition, be careful which pointer to move

**Interview Follow-Ups:**

**Q: "Can you do it without extra space?"**
A: Two pointers is already O(1) space - emphasize this advantage

**Q: "What if array is not sorted?"**
A: "I can sort it first in O(n log n), still better than O(n²)"

**Q: "Can you extend to 4Sum?"**
A: "Yes, fix two elements and use two pointers for other two"

**Q: "What about cycle detection in linked list?"**
A: "Use fast and slow pointers - Floyd\'s algorithm"

**Practice Roadmap:**

**Week 1: Master Basics**
1. Two Sum (sorted array)
2. Remove duplicates
3. Valid palindrome
4. Container with most water

**Week 2: Intermediate**
5. 3Sum
6. Trapping rain water
7. Longest substring without repeating chars
8. Minimum window substring

**Week 3: Advanced**
9. 4Sum
10. Linked list cycle detection
11. Dutch national flag
12. Merge sorted arrays

**Time Management:**
- Spend 2-3 minutes planning before coding
- Should complete easy problems in 10-15 minutes
- Medium problems in 15-25 minutes
- Don't rush - correct code beats fast wrong code!`,
};
