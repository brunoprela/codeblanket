/**
 * Complexity Analysis Section
 */

export const complexitySection = {
  id: 'complexity',
  title: 'Complexity Analysis',
  content: `**Time Complexity Patterns:**

**1D DP (single loop):**
\`\`\`python
for i in range(n):
    dp[i] = some_function(dp[i-1], dp[i-2])
\`\`\`
**Time**: O(n)

---

**2D DP (nested loops):**
\`\`\`python
for i in range(m):
    for j in range(n):
        dp[i][j] = some_function(dp[i-1][j], dp[i][j-1])
\`\`\`
**Time**: O(m * n)

---

**DP with Inner Loop:**
\`\`\`python
for i in range(n):
    for j in range(i):  # Inner depends on outer
        dp[i] = max(dp[i], dp[j] + something)
\`\`\`
**Time**: O(n²)

Examples: Longest Increasing Subsequence (naive), Word Break

---

**DP with Coin/Item Loop:**
\`\`\`python
for amount in range(target + 1):
    for coin in coins:
        dp[amount] = min(dp[amount], dp[amount - coin] + 1)
\`\`\`
**Time**: O(amount * len(coins))

Examples: Coin Change, Combination Sum

---

**Space Complexity Patterns:**

**Original Problem → DP Space:**
- 1 variable (n) → O(n) array
- 2 variables (m, n) → O(m * n) 2D array
- String length n → O(n) or O(n²)
- Two strings (m, n) → O(m * n)

**After Optimization:**
- O(n) → O(1) (rolling variables)
- O(m * n) → O(n) (single row)
- O(n²) → O(n) (single row/column)

---

**Comparison: Recursion vs DP:**

**Fibonacci Example:**

| Approach | Time | Space | Notes |
|----------|------|-------|-------|
| Naive Recursion | O(2ⁿ) | O(n) | Exponential! |
| Memoization | O(n) | O(n) | Top-down |
| Tabulation | O(n) | O(n) | Bottom-up |
| Optimized | O(n) | O(1) | Two variables |

**Dramatic improvement!** DP reduces exponential to linear.

---

**Common Complexities:**

**O(n)**: Climbing stairs, house robber, min cost climbing
**O(n²)**: LIS (naive), palindrome substrings, best time to buy/sell (with k)
**O(m * n)**: LCS, edit distance, unique paths, minimum path sum
**O(n * target)**: Coin change, partition equal subset, target sum
**O(n * k)**: Paint house, best time to buy/sell with k transactions`,
};
