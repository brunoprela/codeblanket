/**
 * Common Pitfalls Section
 */

export const commonpitfallsSection = {
  id: 'common-pitfalls',
  title: 'Common Pitfalls',
  content: `**Pitfall 1: Not Checking Empty Stack**

❌ **Wrong:**
\`\`\`python
def pop_without_check(stack):
    return stack.pop()  # IndexError if stack is empty
\`\`\`

✅ **Correct:**
\`\`\`python
def pop_with_check(stack):
    if not stack:
        return None  # or raise custom exception
    return stack.pop()
\`\`\`

---

**Pitfall 2: Forgetting to Pop in Matching Problems**

❌ **Wrong:**
\`\`\`python
def valid_parentheses_wrong(s: str) -> bool:
    stack = []
    for char in s:
        if char in '({[':
            stack.append(char)
        else:
            if not stack:  # Forgot to check match!
                return False
    return len(stack) == 0
\`\`\`

✅ **Correct:**
\`\`\`python
def valid_parentheses_correct(s: str) -> bool:
    stack = []
    pairs = {'(': ')', '{': '}', '[': ']'}
    for char in s:
        if char in pairs:
            stack.append(char)
        else:
            if not stack or pairs[stack.pop()] != char:  # Check match!
                return False
    return len(stack) == 0
\`\`\`

---

**Pitfall 3: Monotonic Stack - Wrong Comparison Direction**

❌ **Wrong (Next Greater):**
\`\`\`python
while stack and nums[stack[-1]] > nums[i]:  # Should be <
    stack.pop()
\`\`\`

✅ **Correct:**
\`\`\`python
while stack and nums[stack[-1]] < nums[i]:  # < for next GREATER
    idx = stack.pop()
    result[idx] = nums[i]
\`\`\`

**Memory Aid:**
- **Next Greater** → pop **smaller** (use **<**)
- **Next Smaller** → pop **greater** (use **>**)

---

**Pitfall 4: Off-by-One in Rectangle Problems**

When computing areas with stacks, be careful with index boundaries:
\`\`\`python
# Common mistake: forgetting to add sentinel values
heights = [2, 1, 5, 6, 2, 3]
heights = [0] + heights + [0]  # Add sentinels for easier computation
\`\`\``,
};
