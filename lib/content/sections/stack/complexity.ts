/**
 * Complexity Analysis Section
 */

export const complexitySection = {
  id: 'complexity',
  title: 'Complexity Analysis',
  content: `**Stack Operations:**

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| push() | O(1) | - |
| pop() | O(1) | - |
| peek() | O(1) | - |
| isEmpty() | O(1) | - |
| Full traversal | O(N) | - |

**Space Complexity:**
- Stack itself: **O(N)** where N is the number of elements
- Auxiliary space for stack operations: **O(1)**

**Common Problem Complexities:**

**Valid Parentheses:**
- Time: O(N) - single pass through string
- Space: O(N) - worst case all opening brackets

**Min Stack:**
- Time: O(1) for all operations (push, pop, getMin)
- Space: O(N) - need to maintain min values

**Next Greater Element:**
- Time: O(N) - each element pushed/popped once
- Space: O(N) - stack can contain all elements

**Largest Rectangle in Histogram:**
- Time: O(N) - each bar pushed/popped once
- Space: O(N) - stack stores indices

**Key Insight:**
Stacks enable O(N) solutions to problems that would otherwise require O(N²) nested loops.`,
};
