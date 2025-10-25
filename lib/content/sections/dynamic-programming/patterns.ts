/**
 * Common DP Patterns Section
 */

export const patternsSection = {
  id: 'patterns',
  title: 'Common DP Patterns',
  content: `**Pattern 1: 0/1 Knapsack**

Given items with weights and values, and a capacity, maximize value without exceeding capacity.

**State**: \`dp[i][w]\` = max value using first i items with capacity w

**Recurrence**:
\`\`\`python
# For each item, choose to include it or not
if weight[i] <= w:
    dp[i][w] = max(
        dp[i-1][w],                    # Don't take
        dp[i-1][w-weight[i]] + value[i] # Take
    )
else:
    dp[i][w] = dp[i-1][w]  # Can't take (too heavy)
\`\`\`

**Applications**: Subset sum, partition equal subset sum, target sum

---

**Pattern 2: Unbounded Knapsack**

Can use each item unlimited times.

**Recurrence**:
\`\`\`python
if weight[i] <= w:
    dp[i][w] = max(
        dp[i-1][w],                # Don't take
        dp[i][w-weight[i]] + value[i]  # Take (can reuse)
    )
\`\`\`

**Applications**: Coin change, min cost climbing stairs

---

**Pattern 3: Longest Common Subsequence (LCS)**

Find longest subsequence common to two strings.

**State**: \`dp[i][j]\` = LCS length of text1[0:i] and text2[0:j]

**Recurrence**:
\`\`\`python
if text1[i-1] == text2[j-1]:
    dp[i][j] = dp[i-1][j-1] + 1
else:
    dp[i][j] = max (dp[i-1][j], dp[i][j-1])
\`\`\`

**Applications**: Edit distance, shortest common supersequence

---

**Pattern 4: Palindrome Problems**

Check if string/substring is palindrome.

**State**: \`dp[i][j]\` = is substring s[i:j+1] a palindrome?

**Recurrence**:
\`\`\`python
if s[i] == s[j]:
    dp[i][j] = (i+1 > j-1) or dp[i+1][j-1]
else:
    dp[i][j] = False
\`\`\`

**Applications**: Longest palindromic substring, palindrome partitioning

---

**Pattern 5: Grid Path Problems**

Count paths or find min/max cost in a grid.

**State**: \`dp[i][j]\` = count/cost to reach cell (i, j)

**Recurrence** (top-left to bottom-right):
\`\`\`python
dp[i][j] = grid[i][j] + min(
    dp[i-1][j],  # From above
    dp[i][j-1]   # From left
)
\`\`\`

**Applications**: Unique paths, minimum path sum, dungeon game

---

**Pattern 6: Decision Making**

Make optimal decisions at each step.

**State**: \`dp[i]\` = best outcome after i decisions

**Example - House Robber**:
\`\`\`python
dp[i] = max(
    dp[i-1],           # Skip house i
    dp[i-2] + nums[i]  # Rob house i
)
\`\`\`

**Applications**: Best time to buy/sell stock, delete and earn

---

**Pattern 7: State Machine DP**

Track different states (e.g., holding stock, cooldown).

**Example - Stock with cooldown**:
\`\`\`python
# Three states: hold stock, sold, rest
hold[i] = max (hold[i-1], rest[i-1] - price[i])
sold[i] = hold[i-1] + price[i]
rest[i] = max (rest[i-1], sold[i-1])
\`\`\``,
};
