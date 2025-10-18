/**
 * Interview Strategy Section
 */

export const interviewSection = {
  id: 'interview',
  title: 'Interview Strategy',
  content: `**Recognizing DP Problems:**

**Strong Indicators:**
1. "Find maximum/minimum" → Optimization
2. "Count ways to..." → Counting
3. "Is it possible to..." → Decision
4. Naive recursion too slow (exponential)
5. Overlapping subproblems
6. Optimal substructure

**Problem Types:**
- Fibonacci-like (stairs, tribonacci)
- Grid traversal (paths, cost)
- String problems (LCS, edit distance)
- Subset/subsequence problems
- Stock trading problems

---

**Problem-Solving Process:**

**Step 1: Understand & Clarify (2 min)**
- What are we optimizing?
- What are constraints?
- Can we do brute force?

**Step 2: Identify if DP (1 min)**
- Optimal substructure?
- Overlapping subproblems?
- Choices at each step?

**Step 3: Define State (3 min)**
- What does \`dp[i]\` represent?
- 1D or 2D needed?
- Write it down clearly!

**Step 4: Find Recurrence (5 min)**
- How to compute \`dp[i]\` from previous?
- Try small examples
- Draw the DP table

**Step 5: Identify Base Cases (2 min)**
- Smallest subproblems
- Edge cases

**Step 6: Implement (10 min)**
- Start with clear, simple solution
- Test with examples
- Debug

**Step 7: Optimize (5 min)**
- Can space be reduced?
- Any redundant computations?

---

**Communication Tips:**

**Example: Coin Change**

*Interviewer: Find minimum coins to make amount using given coins.*

**You:**
1. **Clarify:** "Can I use each coin unlimited times? Can amount be 0? Are coins always positive?"

2. **Identify:** "This looks like an optimization problem with optimal substructure. If I can make amount \`i\`, I can make amount \`i + coin\` with one more coin."

3. **State:** "I'll use \`dp[i]\` to represent the minimum coins needed to make amount \`i\`."

4. **Recurrence:** "For each amount, I'll try using each coin. If I use coin \`c\`, then \`dp[i] = dp[i-c] + 1\`. I'll take the minimum across all coins."

5. **Base case:** "\`dp[0] = 0\` because we need 0 coins for amount 0."

6. **Implementation:** (write code)

7. **Complexity:** "Time is O(amount * coins), space is O(amount)."

---

**Common Mistakes:**

**1. Wrong State Definition**
Be precise! "Maximum profit" vs "Maximum profit ending at i"

**2. Missing Base Cases**
Always handle n=0, n=1, empty input

**3. Wrong Iteration Order**
Ensure dependencies computed first

**4. Off-by-One Errors**
Carefully handle indices (0-based vs 1-based)

**5. Not Considering All Choices**
At each state, consider all valid transitions

---

**Practice Progression:**

**Week 1: Fundamentals**
- Climbing Stairs
- House Robber
- Min Cost Climbing Stairs

**Week 2: 1D DP**
- Longest Increasing Subsequence
- Decode Ways
- Word Break

**Week 3: 2D DP**
- Unique Paths
- Longest Common Subsequence
- Edit Distance

**Week 4: Advanced**
- Coin Change (Unbounded Knapsack)
- Partition Equal Subset Sum (0/1 Knapsack)
- Best Time to Buy/Sell Stock (all variations)

**Resources:**
- LeetCode DP tag (400+ problems)
- Start with Easy, progress to Hard
- Group by pattern (knapsack, LCS, grid)`,
};
