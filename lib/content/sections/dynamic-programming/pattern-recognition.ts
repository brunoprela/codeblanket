/**
 * DP Pattern Recognition & Decision Guide Section
 */

export const patternrecognitionSection = {
  id: 'pattern-recognition',
  title: 'DP Pattern Recognition & Decision Guide',
  content: `## 🎯 DP Pattern Recognition Decision Tree

\`\`\`
START: Is this a DP problem?

├─ Ask: "Can I break this into smaller similar subproblems?"
│  └─ If NO → Not DP, consider greedy/divide-and-conquer
│
├─ Ask: "Do subproblems overlap (same calculations repeated)?"
│  └─ If NO → Use divide-and-conquer or recursion
│
├─ Ask: "Does the problem ask for optimal/count/possible?"
│  ├─ "Find maximum/minimum/longest/shortest" → OPTIMIZATION DP
│  ├─ "Count number of ways" → COUNTING DP
│  └─ "Is it possible to..." → EXISTENCE DP
│
└─ YES, this is DP! Now identify the pattern:

   ├─ LINEAR SEQUENCE (1D DP)
   │  ├─ Each element: take or skip → House Robber pattern
   │  ├─ Build up to target → Coin Change pattern
   │  ├─ Climbing steps → Fibonacci pattern
   │  └─ Break into substrings → Word Break pattern
   │
   ├─ TWO SEQUENCES (2D DP)
   │  ├─ Match characters → LCS/Edit Distance pattern
   │  ├─ Transform one to other → Edit Distance pattern
   │  └─ Merge sequences → Interleaving String pattern
   │
   ├─ GRID/MATRIX
   │  ├─ Top-left to bottom-right paths → Grid Path pattern
   │  ├─ Square submatrices → Maximal Square pattern
   │  └─ Area/sum in rectangle → 2D Prefix Sum pattern
   │
   ├─ KNAPSACK FAMILY
   │  ├─ Weight limit, maximize value → 0/1 Knapsack
   │  ├─ Subset with target sum → Subset Sum pattern
   │  ├─ Unlimited items → Unbounded Knapsack
   │  └─ Partition into groups → Partition Equal Subset
   │
   ├─ STRING PATTERNS
   │  ├─ Palindrome questions → Expand Around Center
   │  ├─ Subsequences → LCS pattern
   │  ├─ Pattern matching → Regex DP pattern
   │  └─ Partition into palindromes → Palindrome Partitioning
   │
   ├─ DECISION MAKING
   │  ├─ Buy/sell with limits → Stock Trading pattern
   │  ├─ Multiple states → State Machine DP
   │  └─ With cooldown/fee → Extended State DP
   │
   └─ GAME THEORY
      ├─ Minimax scenarios → Game DP pattern
      └─ Predict winner → Optimal Strategy DP
\`\`\`

---

## 📊 DP Pattern Comparison Table

| Pattern | State Definition | Recurrence | Complexity | Example Problems |
|---------|------------------|------------|------------|------------------|
| **Fibonacci** | dp[i] = answer for i | dp[i] = dp[i-1] + dp[i-2] | O(n), O(1) | Climbing Stairs, Decode Ways |
| **House Robber** | dp[i] = max money robbing 0..i | dp[i] = max (dp[i-1], dp[i-2]+nums[i]) | O(n), O(1) | House Robber, Delete and Earn |
| **Coin Change** | dp[i] = ways/min coins for amount i | dp[i] = min (dp[i-coin]+1) for each coin | O(n*m), O(n) | Coin Change, Perfect Squares |
| **Knapsack 0/1** | dp[i][w] = max value with i items, weight w | dp[i][w] = max (dp[i-1][w], dp[i-1][w-wt[i]]+val[i]) | O(n*W), O(W) | Partition Equal Subset, Target Sum |
| **LCS** | dp[i][j] = LCS length of s1[0..i], s2[0..j] | dp[i][j] = dp[i-1][j-1]+1 if match else max (dp[i-1][j], dp[i][j-1]) | O(m*n), O(n) | Longest Common Subsequence, Edit Distance |
| **Grid Paths** | dp[i][j] = ways/min cost to reach (i,j) | dp[i][j] = dp[i-1][j] + dp[i][j-1] | O(m*n), O(n) | Unique Paths, Minimum Path Sum |
| **Palindrome** | dp[i][j] = is s[i..j] palindrome | dp[i][j] = s[i]==s[j] && dp[i+1][j-1] | O(n²), O(n²) | Longest Palindromic Substring, Palindrome Partitioning |

---

## 🚨 Common Mistakes & How to Avoid Them

### Mistake 1: Unclear State Definition
**Problem:** Not being precise about what dp[i] represents.

\`\`\`python
# ❌ BAD: Vague state
# "dp[i] = something about the first i elements"

# ✅ GOOD: Precise state
# "dp[i] = maximum sum of non-adjacent elements in nums[0..i]"
# "dp[i][j] = minimum edit distance between s1[0..i] and s2[0..j]"
\`\`\`

**Why it matters:** Clear state definition drives correct recurrence relation.

### Mistake 2: Wrong Base Cases
**Problem:** Not handling empty inputs or edge cases.

\`\`\`python
# ❌ BAD: Missing base case
def coinChange (coins, amount):
    dp = [float('inf')] * (amount + 1)
    # Forgot dp[0] = 0!
    for i in range(1, amount + 1):
        for coin in coins:
            if i >= coin:
                dp[i] = min (dp[i], dp[i-coin] + 1)
    return dp[amount]

# ✅ GOOD: Proper base case
def coinChange (coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # Base case: 0 coins needed for amount 0
    for i in range(1, amount + 1):
        for coin in coins:
            if i >= coin:
                dp[i] = min (dp[i], dp[i-coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
\`\`\`

### Mistake 3: Wrong Iteration Order
**Problem:** Computing dp[i] before its dependencies are ready.

\`\`\`python
# ❌ BAD: Wrong order for LCS
def longestCommonSubsequence (text1, text2):
    m, n = len (text1), len (text2)
    dp = [[0] * (n + 1) for _ in range (m + 1)]
    
    # WRONG: Going backwards when we need forward
    for i in range (m, 0, -1):  # ❌
        for j in range (n, 0, -1):  # ❌
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1  # Depends on smaller indices!
            else:
                dp[i][j] = max (dp[i-1][j], dp[i][j-1])
    return dp[m][n]

# ✅ GOOD: Forward iteration
def longestCommonSubsequence (text1, text2):
    m, n = len (text1), len (text2)
    dp = [[0] * (n + 1) for _ in range (m + 1)]
    
    for i in range(1, m + 1):  # ✅
        for j in range(1, n + 1):  # ✅
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max (dp[i-1][j], dp[i][j-1])
    return dp[m][n]
\`\`\`

### Mistake 4: Not Optimizing Space
**Problem:** Using 2D array when 1D would suffice.

\`\`\`python
# ❌ UNOPTIMIZED: O(n) space when O(1) possible
def rob (nums):
    if not nums:
        return 0
    n = len (nums)
    dp = [0] * n
    dp[0] = nums[0]
    if n > 1:
        dp[1] = max (nums[0], nums[1])
    for i in range(2, n):
        dp[i] = max (dp[i-1], dp[i-2] + nums[i])
    return dp[n-1]

# ✅ OPTIMIZED: O(1) space
def rob (nums):
    if not nums:
        return 0
    prev2, prev1 = 0, 0
    for num in nums:
        prev2, prev1 = prev1, max (prev1, prev2 + num)
    return prev1
\`\`\`

### Mistake 5: Using DP When Greedy Works
**Problem:** Over-complicating with DP when greedy is sufficient.

\`\`\`python
# ❌ OVERKILL: Using DP for jump game when greedy works
def canJump (nums):
    n = len (nums)
    dp = [False] * n
    dp[0] = True
    for i in range (n):
        if dp[i]:
            for j in range(1, nums[i] + 1):
                if i + j < n:
                    dp[i + j] = True
    return dp[n-1]

# ✅ BETTER: Greedy approach - O(n) time, O(1) space
def canJump (nums):
    max_reach = 0
    for i in range (len (nums)):
        if i > max_reach:
            return False
        max_reach = max (max_reach, i + nums[i])
    return True
\`\`\`

---

## 💡 Interview Tips for DP Problems

### Tip 1: Start with Brute Force Recursion
\`\`\`
✅ Good Approach:
1. Write naive recursive solution
2. Identify overlapping subproblems
3. Add memoization (top-down DP)
4. Convert to bottom-up if asked
5. Optimize space if possible

"Let me start with recursion to understand the problem structure,
then optimize with memoization..."
\`\`\`

### Tip 2: Draw the DP Table
\`\`\`
✅ Communication Strategy:
"Let me visualize the dp table for a small example..."

Example: LCS of "abc" and "def"
    ""  d  e  f
""   0  0  0  0
a    0  0  0  0
b    0  0  0  0
c    0  0  0  0

Walk through filling the table: "If characters match, we take diagonal + 1..."
\`\`\`

### Tip 3: Explain State Transitions
\`\`\`
✅ Clear Explanation:
"At each position, I have two choices:
1. Include current element → dp[i-1] + nums[i]
2. Exclude current element → dp[i-1]
I take the maximum of these two choices."
\`\`\`

### Tip 4: Recognize the Pattern Early
\`\`\`
Common question → Pattern mapping:

"Climbing stairs" → Fibonacci pattern
"Maximum subarray" → Kadane\'s algorithm (1D DP)
"Longest increasing subsequence" → LIS pattern
"Can partition to equal sum" → 0/1 Knapsack
"Minimum cost path" → Grid DP
"Transform string A to B" → Edit Distance
"Find all ways to..." → Counting DP
\`\`\`

### Tip 5: Discuss Time-Space Tradeoffs
\`\`\`
✅ Show Optimization Thinking:
"My initial solution is O(n²) time and O(n²) space.
I can optimize space to O(n) by keeping only the previous row,
since we only need the previous row to compute the current row.
For some problems, we can even achieve O(1) space by keeping
just the last two values."
\`\`\`

---

## 🎓 Problem-Solving Framework for DP

### Step 1: Identify It's a DP Problem
**Keywords to watch for:**
- Maximum/Minimum
- Longest/Shortest
- Count number of ways
- Is it possible
- Optimal strategy

**Validation:**
- Can break into subproblems?
- Do subproblems overlap?
- Can I optimize from brute force?

### Step 2: Define the State
**Ask yourself:**
- What varies in the problem?
- What do I need to know to make a decision?
- How many dimensions needed?

**Examples:**
\`\`\`
Fibonacci: dp[i] = fibonacci (i)
House Robber: dp[i] = max money robbing houses[0..i]
Knapsack: dp[i][w] = max value with first i items, capacity w
LCS: dp[i][j] = LCS length of s1[0..i] and s2[0..j]
\`\`\`

### Step 3: Find the Recurrence Relation
**Think about:**
- What decision am I making at each step?
- How do smaller subproblems combine?

**Common patterns:**
\`\`\`
Take/Skip: max (skip_it, take_it + prev)
Min/Max: min/max over all choices
Count ways: sum of ways from previous states
Path problems: sum of ways to reach current from previous cells
\`\`\`

### Step 4: Determine Base Cases
**Consider:**
- Empty input: dp[0] = ?
- Smallest valid input: dp[1] = ?
- Edge cases: boundaries, negatives, zero

### Step 5: Decide Iteration Order
**Rules:**
- Compute dependencies before dependents
- 1D: Usually left to right
- 2D: Usually top-left to bottom-right
- Sometimes reverse order (check recurrence!)

### Step 6: Implement and Optimize
**Progression:**
1. Recursive with memoization (top-down)
2. Iterative with table (bottom-up)
3. Space optimization (rolling array, variables)

---

## 🔍 When NOT to Use DP

### ❌ Don't use DP if:

**1. Greedy Works Better**
\`\`\`python
# Coin change with greedy coins (powers of 2: 1,2,4,8,16...)
# Greedy gives optimal solution
# DP would be overkill
\`\`\`

**2. No Overlapping Subproblems**
\`\`\`python
# Merge sort - divide and conquer, but subproblems don't overlap
# DP adds no benefit
\`\`\`

**3. Simple Math Formula Exists**
\`\`\`python
# Sum of 1 to n → use formula n*(n+1)/2
# Don't need DP when closed-form solution exists
\`\`\`

**4. Linear Scan Suffices**
\`\`\`python
# Find maximum element → just iterate
# DP would be unnecessary complexity
\`\`\`

**5. Problem Requires Backtracking**
\`\`\`python
# Generate all permutations → need backtracking
# DP can't enumerate solutions, only count/optimize
\`\`\`

---

## 📝 DP Pattern Cheatsheet

### Quick Pattern Matching:

**"Climbing stairs with n steps"** → Fibonacci
- State: dp[i] = ways to reach step i
- Recurrence: dp[i] = dp[i-1] + dp[i-2]

**"Rob houses, can't rob adjacent"** → House Robber
- State: dp[i] = max money from houses[0..i]
- Recurrence: dp[i] = max (dp[i-1], dp[i-2] + nums[i])

**"Minimum coins to make amount"** → Coin Change
- State: dp[i] = min coins for amount i
- Recurrence: dp[i] = min (dp[i-coin] + 1) for all coins

**"Can partition into equal sum"** → 0/1 Knapsack
- State: dp[i][sum] = can make sum with first i nums
- Recurrence: dp[i][sum] = dp[i-1][sum] || dp[i-1][sum-nums[i]]

**"Longest common subsequence"** → 2D String DP
- State: dp[i][j] = LCS of s1[0..i], s2[0..j]
- Recurrence: if match → dp[i-1][j-1]+1, else max (dp[i-1][j], dp[i][j-1])

**"Unique paths in grid"** → Grid DP
- State: dp[i][j] = paths to reach (i,j)
- Recurrence: dp[i][j] = dp[i-1][j] + dp[i][j-1]

**"Longest palindromic substring"** → Palindrome DP
- State: dp[i][j] = is s[i..j] palindrome
- Recurrence: dp[i][j] = (s[i]==s[j]) && dp[i+1][j-1]

**"Buy/sell stock with k transactions"** → State Machine DP
- State: dp[i][k][0/1] = max profit at day i, k transactions, holding/not holding
- Complex recurrence based on buy/sell/hold decisions`,
};
