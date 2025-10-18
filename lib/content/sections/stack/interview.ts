/**
 * Interview Strategy Section
 */

export const interviewSection = {
  id: 'interview',
  title: 'Interview Strategy',
  content: `**Recognition Signals:**

**Use Stack when you see:**
- "Valid parentheses" or "balanced brackets"
- "Next greater/smaller element"
- "Recent" or "most recent" operations
- "Backtrack" or "undo"
- "Nested" structures
- "Last-In-First-Out" behavior
- "Monotonic" increasing/decreasing
- Parsing/evaluating expressions

---

**Problem-Solving Steps:**

**Step 1: Identify the Pattern**
- Matching pairs? → Use stack for validation
- Finding next greater/smaller? → Monotonic stack
- Need to track min/max? → Dual stack approach
- Expression evaluation? → Two stacks (operands + operators)

**Step 2: Choose Stack Content**
- Store values directly? (e.g., parentheses validation)
- Store indices? (e.g., next greater element - need positions)
- Store tuples? (e.g., (value, min) for MinStack)

**Step 3: Define Loop Invariant**
What property does your stack maintain at each step?
- "Stack contains unmatched opening brackets"
- "Stack is monotonically decreasing"
- "Stack[i] is minimum from 0 to i"

**Step 4: Handle Edge Cases**
- Empty input
- Single element
- All same elements
- Already sorted (increasing/decreasing)
- Stack becomes empty mid-iteration

---

**Interview Communication:**

1. **State the approach:** "I'll use a stack to track unmatched opening brackets."
2. **Explain the invariant:** "The stack will always contain brackets that haven't found their closing pair yet."
3. **Walk through example:** Show 2-3 steps of stack operations
4. **Discuss complexity:** Time O(N), Space O(N) for stack
5. **Mention optimizations:** "We could use array instead of list for slight speed improvement."

---

**Common Follow-ups:**

**Q: Can you solve it without a stack?**
- Some problems can use indices/counters (e.g., simple parentheses counting)
- Monotonic stack problems usually need the stack

**Q: What if input is too large for memory?**
- Process in chunks if possible
- Use streaming/online algorithms
- Discuss space-time tradeoffs

**Q: How would you handle this recursively?**
- Stack problems often have recursive equivalents
- Call stack acts as implicit stack
- Discuss pros/cons (readability vs. stack overflow risk)

---

**Practice Plan:**

1. **Basics (Day 1-2):**
   - Valid Parentheses
   - Implement Stack (Min, Max variants)

2. **Monotonic Stack (Day 3-5):**
   - Next Greater Element
   - Daily Temperatures
   - Largest Rectangle in Histogram

3. **Advanced (Day 6-7):**
   - Basic Calculator
   - Decode String
   - Trapping Rain Water

4. **Resources:**
   - LeetCode Stack tag (50+ problems)
   - Practice daily until patterns become automatic`,
};
