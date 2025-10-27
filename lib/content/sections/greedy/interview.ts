/**
 * Interview Strategy Section
 */

export const interviewSection = {
  id: 'interview',
  title: 'Interview Strategy',
  content: `**Recognition Signals:**

**Use Greedy when you see:**
- "Maximum", "minimum", "optimal"
- Scheduling, intervals
- "Can reach", "jump to"
- Sorted input or easy to sort
- Local choice seems to work
- No need to reconsider choices

---

**Problem-Solving Steps:**

**Step 1: Identify Greedy Potential (2 min)**
- Does local optimal lead to global?
- Can I make choice without looking back?
- Is there clear "best" choice each step?

**Step 2: Define Greedy Choice (3 min)**
- What\'s the greedy criterion?
  - Earliest finish?
  - Highest ratio?
  - Largest/smallest?
- How to select at each step?

**Step 3: Sort if Needed (1 min)**
- What sorting key?
- Ascending or descending?

**Step 4: Prove Correctness (5 min)**
- Exchange argument
- Stays ahead
- Can you find counterexample?

**Step 5: Implement (8 min)**

**Step 6: Test Edge Cases (3 min)**
- Empty input
- Single element
- All same
- Already sorted

---

**Interview Communication:**

**Example: Jump Game**

*Interviewer: Can you reach last index?*

**You:**1. **Clarify:**
   - "Is array always valid (no negatives)?"
   - "Can values be 0?"

2. **Approach:**
   - "I'll use greedy - track maximum reachable index."
   - "At each position, update max reach."
   - "If current position > max reach, can't continue."

3. **Why Greedy:**
   - "Making locally optimal choice (max reach) is globally optimal."
   - "Never need to backtrack."

4. **Complexity:**
   - "Time: O(n) - single pass."
   - "Space: O(1) - only track one variable."

---

**Common Mistakes:**

**1. Assuming Greedy Works**
Always verify or prove!

**2. Wrong Sorting Key**
Activity selection: sort by END, not start.

**3. Not Considering Counterexamples**
Test greedy on small examples first.

**4. Confusing with DP**
If greedy fails, switch to DP.

---

**Red Flags (Greedy Won't Work):**

- "Longest path" (try DP)
- "All possible ways" (try DP)
- "Can't sort" (greedy harder)
- "Need to reconsider" (not greedy)

---

**Practice Progression:**

**Week 1: Basics**
- Jump Game
- Best Time to Buy/Sell Stock
- Maximum Subarray

**Week 2: Intervals**
- Meeting Rooms II
- Non-overlapping Intervals
- Minimum Arrows

**Week 3: Advanced**
- Gas Station
- Jump Game II
- Task Scheduler`,
};
