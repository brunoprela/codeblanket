/**
 * Code Templates Section
 */

export const templatesSection = {
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
};
