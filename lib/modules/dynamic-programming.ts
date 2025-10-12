import { Module } from '@/lib/types';

export const dynamicProgrammingModule: Module = {
  id: 'dynamic-programming',
  title: 'Dynamic Programming',
  description:
    'Master the art of breaking problems into overlapping subproblems and building optimal solutions.',
  icon: '🧩',
  timeComplexity: 'Varies (often O(n²) or O(n*m))',
  spaceComplexity: 'O(n) to O(n²) typically',
  sections: [
    {
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
    },
    {
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
- Coin Change: \`dp[i] = min(dp[i-c] + 1 for c in coins)\`

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
def climb_stairs(n):
    if n <= 1:
        return 1
    
    dp = [0] * (n + 1)
    dp[0], dp[1] = 1, 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]
\`\`\``,
    },
    {
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
    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
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
hold[i] = max(hold[i-1], rest[i-1] - price[i])
sold[i] = hold[i-1] + price[i]
rest[i] = max(rest[i-1], sold[i-1])
\`\`\``,
    },
    {
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
dp = [[0] * n for _ in range(m)]
for i in range(m):
    for j in range(n):
        dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
\`\`\`

**After: O(1) space**
\`\`\`python
# Modify grid in-place
for i in range(m):
    for j in range(n):
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
dp = [[0] * n for _ in range(m)]
\`\`\`

**After: O(n) space**
\`\`\`python
prev_row = [0] * n
curr_row = [0] * n

for i in range(m):
    for j in range(n):
        curr_row[j] = compute(prev_row, curr_row, i, j)
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

**General Rule:**
1. First solve correctly with clear DP table
2. Then optimize space if needed
3. Test thoroughly after optimization`,
    },
    {
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
    },
    {
      id: 'templates',
      title: 'Code Templates',
      content: `**Template 1: 1D DP (Bottom-Up)**
\`\`\`python
def solve(n):
    # Initialize DP array
    dp = [0] * (n + 1)
    
    # Base cases
    dp[0] = base_case_0
    dp[1] = base_case_1
    
    # Fill DP table
    for i in range(2, n + 1):
        dp[i] = compute_from_previous(dp, i)
    
    # Return answer
    return dp[n]
\`\`\`

---

**Template 2: 1D DP (Top-Down Memoization)**
\`\`\`python
def solve(n, memo={}):
    # Base cases
    if n <= 1:
        return base_case
    
    # Check memo
    if n in memo:
        return memo[n]
    
    # Compute and store
    memo[n] = compute_recursive(n, memo)
    return memo[n]
\`\`\`

---

**Template 3: 2D DP**
\`\`\`python
def solve(m, n):
    # Initialize 2D DP table
    dp = [[0] * n for _ in range(m)]
    
    # Base cases
    dp[0][0] = base_case
    
    # Fill first row/column if needed
    for i in range(m):
        dp[i][0] = init_value
    for j in range(n):
        dp[0][j] = init_value
    
    # Fill DP table
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = compute_from_neighbors(dp, i, j)
    
    # Return answer
    return dp[m-1][n-1]
\`\`\`

---

**Template 4: String DP**
\`\`\`python
def solve(s):
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    
    # Base case: single character
    for i in range(n):
        dp[i][i] = True
    
    # Check substrings of increasing length
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j]:
                dp[i][j] = (length == 2) or dp[i+1][j-1]
            else:
                dp[i][j] = other_logic(dp, i, j)
    
    return dp[0][n-1]
\`\`\`

---

**Template 5: 0/1 Knapsack**
\`\`\`python
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Don't take item i-1
            dp[i][w] = dp[i-1][w]
            
            # Take item i-1 (if fits)
            if weights[i-1] <= w:
                dp[i][w] = max(
                    dp[i][w],
                    dp[i-1][w-weights[i-1]] + values[i-1]
                )
    
    return dp[n][capacity]
\`\`\`

---

**Template 6: Space-Optimized 1D**
\`\`\`python
def solve(n):
    # Only keep last 2 values
    prev2, prev1 = base_0, base_1
    
    for i in range(2, n + 1):
        curr = compute(prev1, prev2)
        prev2, prev1 = prev1, curr
    
    return prev1
\`\`\``,
    },
    {
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
    },
  ],
  keyTakeaways: [
    'DP solves problems with optimal substructure and overlapping subproblems',
    'Two approaches: Top-down (memoization) and Bottom-up (tabulation)',
    '5-step framework: Define state, find recurrence, base cases, iteration order, compute answer',
    'Common patterns: Knapsack, LCS, palindromes, grid paths, decision making',
    'Typical complexity: O(n) for 1D, O(m*n) for 2D DP problems',
    'Space optimization: Often reduce O(n) to O(1) or O(m*n) to O(n)',
    'DP dramatically reduces exponential time to polynomial (2ⁿ → n²)',
    'Clear state definition is crucial - be precise about what dp[i] represents',
  ],
  relatedProblems: ['climbing-stairs', 'house-robber', 'coin-change'],
};
