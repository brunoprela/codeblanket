/**
 * Space Optimization Section
 */

export const optimizationSection = {
  id: 'optimization',
  title: 'Space Optimization',
  content: `**Reducing Space Complexity**

Many DP solutions can be optimized from O(n²) or O(n) to O(1) space.

---

**Technique 1: Rolling Array**

If \`dp[i]\` only depends on \`dp[i-1]\` and \`dp[i-2]\`:

**Before: O(n) space**
\`\`\`python
dp = [0] * (n + 1)
dp[0], dp[1] = 0, 1
for i in range(2, n + 1):
    dp[i] = dp[i-1] + dp[i-2]
return dp[n]
\`\`\`

**After: O(1) space**
\`\`\`python
prev2, prev1 = 0, 1
for i in range(2, n + 1):
    curr = prev1 + prev2
    prev2, prev1 = prev1, curr
return prev1
\`\`\`

---

**Technique 2: In-Place Modification**

If allowed to modify input:

**Before: O(m*n) space**
\`\`\`python
dp = [[0] * n for _ in range (m)]
for i in range (m):
    for j in range (n):
        dp[i][j] = grid[i][j] + min (dp[i-1][j], dp[i][j-1])
\`\`\`

**After: O(1) space**
\`\`\`python
# Modify grid in-place
for i in range (m):
    for j in range (n):
        if i == 0 and j == 0:
            continue
        grid[i][j] += min(
            grid[i-1][j] if i > 0 else float('inf'),
            grid[i][j-1] if j > 0 else float('inf')
        )
return grid[-1][-1]
\`\`\`

---

**Technique 3: Two Rows for 2D DP**

If \`dp[i][j]\` only depends on row i-1:

**Before: O(m*n) space**
\`\`\`python
dp = [[0] * n for _ in range (m)]
\`\`\`

**After: O(n) space**
\`\`\`python
prev_row = [0] * n
curr_row = [0] * n

for i in range (m):
    for j in range (n):
        curr_row[j] = compute (prev_row, curr_row, i, j)
    prev_row, curr_row = curr_row, prev_row
\`\`\`

---

**When to Optimize:**

**Do optimize when:**
- Space is critical constraint
- Problem explicitly asks for space optimization
- Interview follow-up question
- Easy to implement (like Fibonacci)

**Don't optimize when:**
- Reduces code clarity significantly
- Time is more important
- Need to trace back solution path
- Premature optimization

**General Rule:**1. First solve correctly with clear DP table
2. Then optimize space if needed
3. Test thoroughly after optimization`,
};
