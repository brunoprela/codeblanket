/**
 * Interview Strategy Section
 */

export const interviewSection = {
  id: 'interview',
  title: 'Interview Strategy',
  content: `**Recognition Signals:**

**Use Sliding Window when you see:**
- "Contiguous" subarray/substring
- "Consecutive" elements
- "Window" explicitly mentioned
- "Longest"/"Shortest"/"Maximum"/"Minimum" with constraints
- "At most K" or "At least K" distinct/same elements
- Array or string traversal problems
- Can optimize from O(N²) to O(N)

---

**Problem-Solving Steps:**

**Step 1: Identify Window Type**
- **Fixed size?** → Use Template 1 (add right, remove left)
- **Variable size?** → Use Template 2 or 3 (adjust left based on condition)

**Step 2: Determine Objective**
- **Maximum/Longest?** → Shrink when invalid, update outside while loop
- **Minimum/Shortest?** → Shrink while valid, update inside while loop
- **Count/Existence?** → Check condition at each step

**Step 3: Choose Data Structure**
- **Need frequencies?** → Hash map or Counter
- **Need uniqueness?** → Set
- **Need order/maximum?** → Deque (monotonic queue)
- **Simple sum/count?** → Variables only

**Step 4: Define Validity Condition**
What makes a window valid or invalid?
- "No repeating characters" → Set size equals window size
- "At most K distinct" → len(freq_map) <= K
- "Sum equals target" → current_sum == target
- "Contains all of T" → All chars in T are in window with sufficient count

**Step 5: Handle Edge Cases**
- Empty input
- K > length of array
- All elements same
- Single element array

---

**Interview Communication:**

1. **Identify pattern:** "This is a sliding window problem because we're looking for contiguous elements."

2. **Choose approach:** "I'll use a variable-size window with a hash set to track distinct characters."

3. **Explain invariant:** "The window will always contain at most K distinct characters."

4. **Walk through example:**
   \`\`\`
   s = "eceba", k = 2
   "e"     → 1 distinct, valid
   "ec"    → 2 distinct, valid, length = 2
   "ece"   → 2 distinct, valid, length = 3 ← max
   "eceb"  → 3 distinct, invalid → shrink to "ceb"
   \`\`\`

5. **Discuss complexity:** "Time O(N) since each element is visited at most twice. Space O(K) for the hash map."

---

**Common Follow-ups:**

**Q: Can you solve it with constant space?**
- If character set is limited (e.g., 26 letters), use array instead of hash map: O(1) space

**Q: What if we need to track the actual substring/subarray?**
- Store indices: \`result = (left, right)\`
- Return: \`s[result[0]:result[1]+1]\`

**Q: How would you modify this for "at least K"?**
- Reverse the condition: shrink while count >= K

**Q: Can this be parallelized?**
- Sliding window is inherently sequential, but can divide array into chunks for approximate solutions

---

**Practice Plan:**

1. **Fixed Window (Day 1-2):**
   - Maximum Sum Subarray of Size K
   - Maximum Average Subarray

2. **Variable Window - Maximum (Day 3-4):**
   - Longest Substring Without Repeating Characters
   - Longest Substring with At Most K Distinct Characters

3. **Variable Window - Minimum (Day 5-6):**
   - Minimum Window Substring
   - Minimum Size Subarray Sum

4. **Advanced (Day 7):**
   - Sliding Window Maximum
   - Permutation in String
   - Find All Anagrams

5. **Resources:**
   - LeetCode Sliding Window tag
   - Practice until you can identify the pattern instantly`,
};
