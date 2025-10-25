/**
 * The 5-Step DP Framework Section
 */

export const stepsSection = {
  id: 'steps',
  title: 'The 5-Step DP Framework',
  content: `**Step 1: Define the DP State**

What does \`dp[i]\` or \`dp[i][j]\` represent?

Examples:
- \`dp[i]\`: Maximum profit using first i items
- \`dp[i][j]\`: Minimum cost to reach cell (i, j)
- \`dp[i]\`: Number of ways to make amount i

**Be precise!** Clear state definition is crucial.

---

**Step 2: Identify the Recurrence Relation**

How to compute \`dp[i]\` from previous states?

Examples:
- Fibonacci: \`dp[i] = dp[i-1] + dp[i-2]\`
- Climbing Stairs: \`dp[i] = dp[i-1] + dp[i-2]\`
- Coin Change: \`dp[i] = min (dp[i-c] + 1 for c in coins)\`

**This is the heart of DP!**

---

**Step 3: Set Base Cases**

What are the smallest subproblems you can solve directly?

Examples:
- Fibonacci: \`dp[0] = 0, dp[1] = 1\`
- Climbing Stairs: \`dp[0] = 1, dp[1] = 1\`
- Coin Change: \`dp[0] = 0\` (0 coins for amount 0)

---

**Step 4: Determine Iteration Order**

What order to fill the DP table?

- **1D**: Usually left to right (i = 0 to n)
- **2D**: Usually row by row, or column by column
- **Key**: Ensure dependencies are computed first

---

**Step 5: Compute Final Answer**

Where is your answer stored?

- Often \`dp[n]\` for 1D problems
- Or \`dp[m][n]\` for 2D problems
- Sometimes requires combining multiple states

---

**Example: Climbing Stairs**

*You can climb 1 or 2 steps at a time. How many ways to reach step n?*

**Step 1: State**
\`dp[i]\` = number of ways to reach step i

**Step 2: Recurrence**
\`dp[i] = dp[i-1] + dp[i-2]\`
- From step i-1, take 1 step
- From step i-2, take 2 steps

**Step 3: Base Cases**
\`dp[0] = 1\` (one way: don't move)
\`dp[1] = 1\` (one way: one step)

**Step 4: Iteration**
For i from 2 to n

**Step 5: Answer**
\`dp[n]\`

\`\`\`python
def climb_stairs (n):
    if n <= 1:
        return 1
    
    dp = [0] * (n + 1)
    dp[0], dp[1] = 1, 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]
\`\`\``,
};
