/**
 * Interview Strategy Section
 */

export const interviewSection = {
  id: 'interview',
  title: 'Interview Strategy',
  content: `**Recognition Signals:**

**Use Linked List techniques when you see:**
- "Linked list" explicitly mentioned
- "Node" with "next" pointer
- Problems about "reversing", "merging", "detecting cycles"
- "In-place" with O(1) space requirement
- "Find middle", "kth from end"
- "Reorder" or "rearrange" nodes

---

**Problem-Solving Steps:**

**Step 1: Clarify the Problem**
- Singly or doubly linked?
- Can we modify in-place?
- What's the format of input/output?
- Any dummy nodes or sentinel values?

**Step 2: Identify the Pattern**
- **Two pointers?** → Fast & slow, cycle detection, middle finding
- **Reversal?** → Iterative or recursive reverse
- **Merging?** → Merge sorted lists
- **Reordering?** → Find pattern, possibly reverse part
- **Cycle?** → Floyd's algorithm
- **Kth from end?** → Runner technique

**Step 3: Handle Edge Cases**
- Empty list (\`head == None\`)
- Single node (\`head.next == None\`)
- Two nodes
- All same values
- Cycle vs. no cycle

**Step 4: Choose Approach**
- **Iterative** (usually O(1) space)
- **Recursive** (simpler but O(N) space)
- **With/without dummy node**

**Step 5: Trace Through Example**
Draw the list and pointers at each step:
\`\`\`
Initial: [1] → [2] → [3] → None
Step 1:  None ← [1]  [2] → [3] → None
Step 2:  None ← [1] ← [2]  [3] → None
Final:   None ← [1] ← [2] ← [3]
\`\`\`

---

**Interview Communication:**

1. **Clarify:**
   - "Is this a singly or doubly linked list?"
   - "Can I modify the list in place?"

2. **Explain approach:**
   - "I'll use two pointers: fast moves 2 steps, slow moves 1 step."
   - "When fast reaches the end, slow will be at the middle."

3. **Discuss alternatives:**
   - "Recursively would be cleaner but uses O(N) space."
   - "Iteratively is O(1) space but slightly more complex."

4. **Handle edge cases:**
   - "For empty list, I'll return None immediately."
   - "For single node, no work needed, return head."

5. **Complexity analysis:**
   - "Time: O(N) since we traverse once."
   - "Space: O(1) using only pointers, no extra data structures."

---

**Common Follow-ups:**

**Q: Can you do it in O(1) space?**
- Avoid hash sets/maps
- Use pointer manipulation instead

**Q: Can you do it without modifying the input?**
- Create new nodes
- Or note that you're modifying in-place

**Q: What if there are multiple cycles?**
- Typically only one cycle in problems
- But clarify assumptions

**Q: Can you do it recursively?**
- Show both iterative and recursive
- Discuss tradeoffs

---

**Practice Plan:**

1. **Basics (Day 1-2):**
   - Reverse Linked List (iterative & recursive)
   - Find Middle
   - Delete Node

2. **Two Pointers (Day 3-4):**
   - Linked List Cycle
   - Cycle Start
   - Remove Nth From End

3. **Merging/Rearranging (Day 5-6):**
   - Merge Two Lists
   - Merge K Lists
   - Reorder List

4. **Advanced (Day 7):**
   - Palindrome Check
   - Copy List with Random Pointer
   - Reverse in K Groups

5. **Resources:**
   - LeetCode Linked List tag (100+ problems)
   - Draw diagrams for every problem
   - Practice until pointer manipulation is intuitive`,
};
