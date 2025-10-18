/**
 * Introduction to Dynamic Programming Section
 */

export const introductionSection = {
  id: 'introduction',
  title: 'Introduction to Dynamic Programming',
  content: `**Dynamic Programming (DP)** is an optimization technique that solves complex problems by breaking them down into simpler **overlapping subproblems** and storing their solutions to avoid redundant computation.

**Core Principles:**

**1. Optimal Substructure**
The optimal solution can be constructed from optimal solutions of subproblems.

Example: Fibonacci
- \`fib(5)\` = \`fib(4)\` + \`fib(3)\`
- Optimal solution built from optimal subproblems

**2. Overlapping Subproblems**
The same subproblems are solved multiple times.

Example: Computing \`fib(5)\` naively:
\`\`\`
fib(5)
├─ fib(4)
│  ├─ fib(3)
│  │  ├─ fib(2)  ← Computed multiple times
│  │  └─ fib(1)
│  └─ fib(2)     ← Repeated
└─ fib(3)        ← Repeated
   ├─ fib(2)
   └─ fib(1)
\`\`\`

Without DP: O(2ⁿ) - exponential!
With DP: O(n) - linear!

---

**Two Main Approaches:**

**1. Top-Down (Memoization)**
- Start with original problem
- Recursively break down
- Cache results in memo
- Natural to write, easier to understand

\`\`\`python
def fib(n, memo={}):
    if n <= 1:
        return n
    if n not in memo:
        memo[n] = fib(n-1, memo) + fib(n-2, memo)
    return memo[n]
\`\`\`

**2. Bottom-Up (Tabulation)**
- Start with smallest subproblems
- Build up to original problem
- Store in table/array
- More space-efficient, iterative

\`\`\`python
def fib(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
\`\`\`

---

**When to Use DP:**

**Recognition Signals:**
- "Maximize/minimize" something
- "Count number of ways"
- "Is it possible to..."
- Problem has optimal substructure
- Naive recursion is too slow
- Making choices at each step

**Classic DP Problem Categories:**
- **1D DP**: Fibonacci, climbing stairs, house robber
- **2D DP**: Knapsack, longest common subsequence, edit distance
- **String DP**: Palindromes, pattern matching
- **Path DP**: Grid paths, triangle, dungeon game
- **Decision DP**: Buy/sell stock, jump game`,
};
