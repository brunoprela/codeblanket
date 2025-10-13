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
            quiz: [
                {
                    id: 'q1',
                    question:
                        'Explain what dynamic programming is and how it differs from recursion. When do you use DP over plain recursion?',
                    sampleAnswer:
                        'Dynamic programming optimizes recursive solutions by storing results of subproblems to avoid recomputing them. Plain recursion recalculates same subproblems many times. For example, Fibonacci fib(5) calls fib(3) multiple times. DP stores fib(3) result once and reuses it. Use DP when problem has two properties: optimal substructure (solution built from optimal solutions of subproblems) and overlapping subproblems (same subproblems solved repeatedly). Without overlapping subproblems, DP provides no benefit over recursion. DP trades space for time: stores O(n) or O(n^2) results to avoid exponential redundant computation. The key insight: if you find yourself solving the same problem multiple times, memoize it.',
                    keyPoints: [
                        'DP stores subproblem results to avoid recomputation',
                        'Plain recursion: recalculates same subproblems',
                        'Needs: optimal substructure + overlapping subproblems',
                        'Trades space for time: O(n) space, avoids exponential time',
                        'If solving same problem multiple times → memoize',
                    ],
                },
                {
                    id: 'q2',
                    question:
                        'Compare top-down (memoization) vs bottom-up (tabulation) DP. When would you choose each?',
                    sampleAnswer:
                        'Top-down uses recursion with memoization: solve problem recursively, store results in cache, check cache before computing. Bottom-up uses iteration with table: solve smallest subproblems first, build up to final answer using table. Top-down is intuitive (matches recursive thinking), only computes needed subproblems, but has recursion overhead. Bottom-up is faster (no recursion), uses less space (can optimize to O(1) in some cases), but computes all subproblems even if not needed. Choose top-down for: complex dependencies, not all subproblems needed, easier to code. Choose bottom-up for: simple dependencies, need all subproblems anyway, want best performance. In interviews, start with top-down (easier), optimize to bottom-up if asked.',
                    keyPoints: [
                        'Top-down: recursion + memoization',
                        'Bottom-up: iteration + table',
                        'Top-down: intuitive, only needed subproblems',
                        'Bottom-up: faster, can optimize space',
                        'Interview: start top-down, optimize to bottom-up',
                    ],
                },
                {
                    id: 'q3',
                    question:
                        'Walk me through recognizing a DP problem. What keywords or patterns signal that DP might be needed?',
                    sampleAnswer:
                        'Several signals indicate DP. Keywords: "maximum", "minimum", "longest", "shortest", "count ways", "can you", "all possible". Asking for optimal value or count of solutions. Problem naturally recursive but naive recursion is too slow. You are making choices and want optimal outcome. Pattern examples: Fibonacci-like recurrence (current depends on previous), grid paths (combine paths from top and left), subsequence problems (include or skip element), knapsack-like decisions (take or leave item). If you write recursive solution and notice same parameters recurring many times, that is overlapping subproblems - use DP. Ask: can I break problem into smaller similar problems? Do subproblems overlap?',
                    keyPoints: [
                        'Keywords: max, min, longest, shortest, count ways',
                        'Optimal value or count of solutions',
                        'Recursive but too slow',
                        'Patterns: Fibonacci, grid paths, subsequence, knapsack',
                        'Same subproblems recurring → use DP',
                    ],
                },
            ],
            multipleChoice: [
                {
                    id: 'mc1',
                    question: 'What are the two key properties for Dynamic Programming?',
                    options: [
                        'Fast and simple',
                        'Optimal substructure and overlapping subproblems',
                        'Recursion and iteration',
                        'Space and time',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'DP requires: 1) Optimal substructure - optimal solution built from optimal subproblems, 2) Overlapping subproblems - same subproblems solved multiple times. Both must be present.',
                },
                {
                    id: 'mc2',
                    question: 'What is the difference between top-down and bottom-up DP?',
                    options: [
                        'No difference',
                        'Top-down: recursive with memoization (cache). Bottom-up: iterative with tabulation (table)',
                        'Top-down is always better',
                        'Random',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Top-down (memoization): start with original problem, recurse, cache results. Bottom-up (tabulation): start with base cases, iteratively build up. Top-down more intuitive, bottom-up more space-efficient.',
                },
                {
                    id: 'mc3',
                    question: 'Why does DP improve time complexity for Fibonacci from O(2^N) to O(N)?',
                    options: [
                        'Different algorithm',
                        'Caches subproblem results - each fib(i) computed once instead of exponentially many times',
                        'Uses more space',
                        'Random',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Naive recursion recomputes fib(2) many times = O(2^N). DP caches each fib(i) result. Each of N subproblems computed once = O(N). Trades space for time.',
                },
                {
                    id: 'mc4',
                    question: 'When should you use DP instead of greedy?',
                    options: [
                        'Always',
                        'When greedy doesn\'t give optimal (e.g., coin change with arbitrary denominations)',
                        'Never',
                        'Random',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Use DP when greedy fails to give optimal solution. Example: coin change [1,3,4] for amount 6. Greedy: 4+1+1=3 coins. DP optimal: 3+3=2 coins. DP tries all possibilities.',
                },
                {
                    id: 'mc5',
                    question: 'What is memoization?',
                    options: [
                        'Memory management',
                        'Caching results of expensive function calls to avoid recomputation',
                        'Writing notes',
                        'Random',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Memoization: cache function results in dictionary/map. When function called with same args, return cached result instead of recomputing. Top-down DP approach. Trade-off: space for time.',
                },
            ],
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
            quiz: [
                {
                    id: 'q1',
                    question:
                        'Walk me through the 5-step DP framework with a concrete example like Climbing Stairs.',
                    sampleAnswer:
                        'Climbing Stairs: reach top of n stairs, can climb 1 or 2 steps. Step 1 (State): dp[i] = ways to reach stair i. Step 2 (Recurrence): dp[i] = dp[i-1] + dp[i-2] because we can reach stair i from i-1 (one step) or i-2 (two steps). Step 3 (Base case): dp[0] = 1 (one way to stay at ground), dp[1] = 1 (one way to reach first stair). Step 4 (Order): iterate i from 2 to n because dp[i] depends on dp[i-1] and dp[i-2]. Step 5 (Answer): dp[n] is ways to reach top. This framework forces systematic thinking: define what you are computing, how to compute it, what are starting values, in what order, and where final answer is.',
                    keyPoints: [
                        'Step 1: dp[i] = ways to reach stair i',
                        'Step 2: dp[i] = dp[i-1] + dp[i-2]',
                        'Step 3: dp[0] = 1, dp[1] = 1',
                        'Step 4: iterate i from 2 to n',
                        'Step 5: answer is dp[n]',
                    ],
                },
                {
                    id: 'q2',
                    question:
                        'Explain how to derive a recurrence relation. What questions do you ask yourself?',
                    sampleAnswer:
                        'To derive recurrence, ask: how can I solve dp[i] using smaller subproblems? What choices do I have at position i? For Fibonacci, to get F(n), I add F(n-1) and F(n-2). For Coin Change, to make amount i, I try each coin and take minimum of (1 + ways to make i-coin). For House Robber, at house i, I either rob it (take value + dp[i-2]) or skip it (take dp[i-1]). Pattern: express current state as combination of previous states based on problem constraints. Draw small examples (n=3, n=4) and see pattern. The recurrence captures the decision or combination logic of the problem. This is the creative step - finding the relationship between subproblems.',
                    keyPoints: [
                        'Ask: how to solve dp[i] from smaller subproblems?',
                        'What choices at position i?',
                        'Express current state from previous states',
                        'Fibonacci: sum previous two',
                        'House Robber: max of (rob + skip previous, skip)',
                    ],
                },
                {
                    id: 'q3',
                    question:
                        'Describe base case initialization. What happens if you initialize incorrectly?',
                    sampleAnswer:
                        'Base cases are smallest subproblems that do not depend on others - directly solvable. For Fibonacci: F(0)=0, F(1)=1. For Climbing Stairs: ways[0]=1, ways[1]=1. For Coin Change: dp[0]=0 (zero coins for amount 0). Initialize wrong and entire solution fails because all other values build on base cases. Common mistake: forgetting edge cases like empty array or zero amount. For example, if you set ways[0]=0 for Climbing Stairs, all subsequent values will be wrong. Base cases seed the DP table - they must be correct. Test base cases independently before running full algorithm. They represent the termination condition of recursive formulation.',
                    keyPoints: [
                        'Base cases: smallest subproblems, directly solvable',
                        'Do not depend on other subproblems',
                        'Wrong base case → entire solution fails',
                        'All other values build on base cases',
                        'Test base cases independently',
                    ],
                },
            ],
            multipleChoice: [
                {
                    id: 'mc1',
                    question: 'What is the first step in the DP framework?',
                    options: [
                        'Write code',
                        'Define DP state - what does dp[i] represent?',
                        'Find base case',
                        'Random',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Step 1: Define state clearly. What does dp[i] or dp[i][j] mean? Example: dp[i] = min cost to reach stair i. Clear state definition guides rest of solution.',
                },
                {
                    id: 'mc2',
                    question: 'What is the recurrence relation in DP?',
                    options: [
                        'Random formula',
                        'Equation expressing current state in terms of previous states: dp[i] = f(dp[i-1], dp[i-2], ...)',
                        'Base case',
                        'Loop',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Recurrence relation: formula expressing dp[i] using previous states. Example: dp[i] = dp[i-1] + dp[i-2] for Fibonacci. Shows how to build solution from subproblems.',
                },
                {
                    id: 'mc3',
                    question: 'Why are base cases critical in DP?',
                    options: [
                        'Optional',
                        'Bootstrap the recursion/iteration - smallest subproblems with known answers',
                        'Random',
                        'For speed',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Base cases: smallest subproblems with direct answers. Example: fib(0)=0, fib(1)=1. Without base cases, recursion never terminates or table has no starting point.',
                },
                {
                    id: 'mc4',
                    question: 'What order should you compute DP table?',
                    options: [
                        'Random order',
                        'Ensure dependencies computed before current state - topological order',
                        'Reverse order',
                        'Any order',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Compute in order where dependencies available. For dp[i] = dp[i-1] + dp[i-2], compute i=0,1,2,... sequentially. For 2D, often row-by-row or column-by-column. Ensure required values already computed.',
                },
                {
                    id: 'mc5',
                    question: 'What is the final step in DP?',
                    options: [
                        'Print everything',
                        'Return/extract answer from DP table (often dp[n] or max/min of certain cells)',
                        'Optimize space',
                        'Random',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Final step: extract answer from DP table. Often dp[n], dp[n][m], or max/min of specific cells depending on problem. Example: longest subsequence = dp[n-1], max profit = dp[n-1][k].',
                },
            ],
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
            quiz: [
                {
                    id: 'q1',
                    question:
                        'Explain the 0/1 Knapsack pattern. Why is it called 0/1 and how does the 2D DP table work?',
                    sampleAnswer:
                        'Called 0/1 because for each item, you have binary choice: include (1) or exclude (0) it. Cannot take partial items or multiple copies. The 2D table dp[i][w] means maximum value using first i items with weight limit w. Recurrence: dp[i][w] = max(skip item i which gives dp[i-1][w], or take item i which gives value[i] + dp[i-1][w-weight[i]]). Take option only valid if weight[i] <= w. Build table row by row: for each item, for each weight, decide take or skip. Final answer dp[n][W] is max value using all n items with weight limit W. This pattern extends to many problems: subset sum, partition, target sum - all involve include/exclude decisions.',
                    keyPoints: [
                        '0/1: binary choice (include or exclude)',
                        'dp[i][w]: max value, first i items, weight limit w',
                        'Recurrence: max(skip, take if weight allows)',
                        'Build row by row: for each item, each weight',
                        'Extends to: subset sum, partition, target sum',
                    ],
                },
                {
                    id: 'q2',
                    question:
                        'Describe the Longest Common Subsequence pattern. How do you fill the 2D table?',
                    sampleAnswer:
                        'LCS finds longest subsequence common to two strings. Not substring - elements need not be consecutive but must maintain order. Table dp[i][j] is LCS length of s1[0..i-1] and s2[0..j-1]. If s1[i-1] equals s2[j-1], characters match: dp[i][j] = 1 + dp[i-1][j-1]. If no match, take max of excluding either character: dp[i][j] = max(dp[i-1][j], dp[i][j-1]). Base case: dp[0][j] = dp[i][0] = 0 (empty string has LCS 0). Fill table row by row, left to right. For "abcde" and "ace", dp[5][3] = 3 (subsequence "ace"). This pattern extends to edit distance, diff algorithms, sequence alignment.',
                    keyPoints: [
                        'LCS: longest common subsequence (maintain order)',
                        'dp[i][j]: LCS of s1[0..i-1] and s2[0..j-1]',
                        'Match: 1 + dp[i-1][j-1]',
                        'No match: max(dp[i-1][j], dp[i][j-1])',
                        'Extends to: edit distance, diff, alignment',
                    ],
                },
                {
                    id: 'q3',
                    question:
                        'Walk me through the State Machine DP pattern with Buy/Sell Stock. Why do we need multiple states?',
                    sampleAnswer:
                        'Stock problems have constraints on transitions between states: can only sell after buying, might have cooldown, might limit transactions. Model as state machine with DP. For cooldown problem: three states per day - holding stock (just bought or continuing to hold), sold today (just sold), resting (waiting, cooldown or never bought). Each state depends on previous day states based on valid transitions. Holding today = max(continue holding, buy from rest). Sold today = sell from holding. Rest today = max(continue rest, cooldown after sold). Multiple states capture the constraints: cannot buy immediately after selling due to cooldown. Final answer is max of sold and rest on last day (cannot be holding). This pattern handles complex state transitions in sequential decision problems.',
                    keyPoints: [
                        'Model constraints as state machine',
                        'Example: holding, sold, resting states',
                        'Each state: DP based on valid prev transitions',
                        'Captures complex constraints (cooldown)',
                        'Answer: max of valid final states',
                    ],
                },
            ],
            multipleChoice: [
                {
                    id: 'mc1',
                    question: 'What is the 1D DP pattern?',
                    options: [
                        'Random',
                        'dp[i] depends on previous indices like dp[i-1], dp[i-2] - linear sequence problems',
                        'Always 2D',
                        'No pattern',
                    ],
                    correctAnswer: 1,
                    explanation:
                        '1D DP: dp[i] represents state at position i. Depends on earlier indices. Examples: Fibonacci, climbing stairs, house robber. Recurrence: dp[i] = f(dp[i-1], dp[i-2], ...).',
                },
                {
                    id: 'mc2',
                    question: 'What is the 2D DP pattern?',
                    options: [
                        'Matrix multiplication',
                        'dp[i][j] represents state with two dimensions - grid paths, LCS, edit distance',
                        'Random',
                        'Never used',
                    ],
                    correctAnswer: 1,
                    explanation:
                        '2D DP: dp[i][j] for problems with two sequences/dimensions. Examples: longest common subsequence, edit distance, grid paths. Often dp[i][j] = f(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]).',
                },
                {
                    id: 'mc3',
                    question: 'What is the knapsack pattern?',
                    options: [
                        'Packing algorithm',
                        'Choose/skip items with capacity constraint - dp[i][w] = max value with i items, capacity w',
                        'Random',
                        'Sorting',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Knapsack: dp[i][w] = maximum value using first i items with weight limit w. For each item: take it (dp[i-1][w-weight]+value) or skip (dp[i-1][w]). Choose max.',
                },
                {
                    id: 'mc4',
                    question: 'What is the subsequence pattern?',
                    options: [
                        'Sequential processing',
                        'Find optimal subsequence - LIS, LCS - often compare/match elements at i and j',
                        'Random',
                        'Substring',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Subsequence: elements in order but not necessarily contiguous. LIS: dp[i] = longest increasing ending at i. LCS: dp[i][j] = longest common of s1[0..i] and s2[0..j]. Match or skip patterns.',
                },
                {
                    id: 'mc5',
                    question: 'What is the partition pattern?',
                    options: [
                        'Divide array',
                        'Split into groups optimally - partition equal subset sum, palindrome partitioning',
                        'Random',
                        'Sorting',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Partition: split into optimal groups. Example: partition equal subset sum - dp[i][s] = can partition first i elements into sum s. Palindrome partition: dp[i] = min cuts for s[0..i].',
                },
            ],
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
            quiz: [
                {
                    id: 'q1',
                    question:
                        'Explain the space optimization technique. When can you reduce from O(n) to O(1)?',
                    sampleAnswer:
                        'Space can be optimized when each dp[i] only depends on fixed number of previous states. For Fibonacci, dp[i] = dp[i-1] + dp[i-2] only needs last two values, not entire array. Use two variables instead of array: prev1 (dp[i-1]), prev2 (dp[i-2]). Each iteration: compute current = prev1 + prev2, then update prev2 = prev1, prev1 = current. This reduces O(n) space to O(1). Works for: Climbing Stairs, House Robber, any problem with fixed lookback distance. Cannot optimize if dependencies are variable or need entire history. For 2D DP, sometimes reduce to 1D by processing row by row if only previous row needed. Key: identify minimum state needed for next computation.',
                    keyPoints: [
                        'Optimize when dp[i] depends on fixed prev states',
                        'Use variables instead of array',
                        'Example: Fibonacci needs only prev two',
                        'O(n) → O(1) for fixed lookback',
                        '2D → 1D if only previous row needed',
                    ],
                },
                {
                    id: 'q2',
                    question:
                        'Describe rolling array for 2D DP space optimization. When is it applicable?',
                    sampleAnswer:
                        'Rolling array reduces 2D DP from O(n×m) to O(m) space when computing row i only needs row i-1. Instead of full 2D table, maintain two 1D arrays: current row and previous row. Process row by row: compute current row using previous row, then swap (previous = current, reset current). For Knapsack: dp[i][w] only depends on dp[i-1][w] and dp[i-1][w-weight], so we only need previous row. Can further optimize to single array by iterating weights right to left (prevents overwriting needed values). This technique works for: Knapsack, LCS, Edit Distance, grid path problems. Key requirement: dependencies only on previous row, not arbitrary past rows.',
                    keyPoints: [
                        'For 2D where row i only needs row i-1',
                        'Two 1D arrays: current and previous',
                        'Process row by row, swap arrays',
                        'Further: single array with right-to-left',
                        'Works: Knapsack, LCS, Edit Distance, grid paths',
                    ],
                },
                {
                    id: 'q3',
                    question:
                        'What are common mistakes when optimizing DP space? How do you avoid them?',
                    sampleAnswer:
                        'First mistake: optimizing space before getting correct solution. Always solve with full space first, verify correctness, then optimize. Second: wrong iteration order after space optimization. For single array Knapsack, must iterate weights right to left to avoid overwriting needed values. Left to right would use already-updated values. Third: losing ability to reconstruct solution path. If problem asks for actual path, not just value, might need to keep full table. Fourth: incorrect variable updates - forgetting to save old values before overwriting. Fifth: optimizing when dependencies are not strictly local. My strategy: solve correctly first, identify dependencies, choose right optimization, test thoroughly, verify iteration order.',
                    keyPoints: [
                        'Always solve correctly with full space first',
                        'Wrong iteration order (esp. single array)',
                        'Knapsack: right-to-left to avoid overwrite',
                        'Might lose path reconstruction ability',
                        'Test thoroughly after optimization',
                    ],
                },
            ],
            multipleChoice: [
                {
                    id: 'mc1',
                    question: 'What is space optimization in DP?',
                    options: [
                        'Delete variables',
                        'Reduce space from O(N²) to O(N) or O(N) to O(1) by keeping only needed previous states',
                        'Random',
                        'Compress data',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Space optimization: observe that dp[i] often only depends on few previous values (dp[i-1], dp[i-2]). Keep only those instead of entire array. Example: Fibonacci O(N)→O(1) with two variables.',
                },
                {
                    id: 'mc2',
                    question: 'How do you optimize Fibonacci from O(N) space to O(1)?',
                    options: [
                        'Different algorithm',
                        'Keep only prev2 and prev1 variables instead of entire dp array',
                        'Cannot optimize',
                        'Random',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Fibonacci only needs last 2 values. Instead of dp array, use prev2=0, prev1=1, curr=prev1+prev2. Update: prev2=prev1, prev1=curr. O(1) space vs O(N).',
                },
                {
                    id: 'mc3',
                    question: 'How do you optimize 2D DP from O(M*N) to O(N)?',
                    options: [
                        'Cannot',
                        'Keep only current and previous row if dp[i][j] only depends on current and previous row',
                        'Use hash map',
                        'Random',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'For grid DP where dp[i][j] depends on dp[i-1][j] and dp[i][j-1], keep only 2 rows (prev and curr) instead of entire M×N table. Update row-by-row. O(N) space.',
                },
                {
                    id: 'mc4',
                    question: 'What is the trade-off between top-down and bottom-up space?',
                    options: [
                        'No difference',
                        'Top-down: O(N) recursion stack + O(N) memo. Bottom-up: O(N) table only (can optimize further)',
                        'Top-down always better',
                        'Random',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Top-down uses recursion stack O(N) plus memo O(N). Bottom-up uses table O(N) and can be optimized to O(1) by keeping only needed states. Bottom-up generally more space-efficient.',
                },
                {
                    id: 'mc5',
                    question: 'What is state compression?',
                    options: [
                        'Data compression',
                        'Use bitmask or compact representation for DP state - 2D→1D using encoding',
                        'Random',
                        'Delete states',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'State compression: encode complex state compactly. Example: use bitmask for subset instead of array. Traveling salesman: dp[mask][i] where mask is visited cities as bits. Reduces dimensions.',
                },
            ],
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
            quiz: [
                {
                    id: 'q1',
                    question:
                        'Explain why 2D DP is often O(n × m). What do n and m typically represent?',
                    sampleAnswer:
                        'Two nested loops create O(n × m) time. Outer loop iterates first dimension (n), inner loop iterates second dimension (m). For Knapsack: n items, m weight limits, so O(n × m). For LCS: n characters in first string, m in second, so O(n × m). Each cell computed in O(1) using values from already-computed cells. Total cells = n × m, each O(1), giving O(n × m). The dimensions represent the two independent variables in the DP state. Space is also O(n × m) for full table, but can sometimes optimize to O(m) with rolling array if only previous row needed. Common pattern: two parameters in state definition leads to 2D table and O(n × m) complexity.',
                    keyPoints: [
                        'Two nested loops: outer n, inner m',
                        'n and m: independent variables in state',
                        'Knapsack: n items, m weights',
                        'LCS: n and m are string lengths',
                        'Time O(n × m), space can optimize to O(m)',
                    ],
                },
                {
                    id: 'q2',
                    question:
                        'Compare time complexity of top-down vs bottom-up DP. Which is faster and why?',
                    sampleAnswer:
                        'Both have same asymptotic complexity (same number of subproblems), but bottom-up is faster in practice. Top-down has recursion overhead: function call stack, parameter passing, return values. Each recursive call costs cycles. Bottom-up uses simple loops with array access - much faster per operation. Top-down only computes needed subproblems which can help if many subproblems unused. Bottom-up computes all subproblems. For most interview problems, all subproblems needed anyway. Also bottom-up has better cache locality (sequential memory access) vs top-down (scattered recursive calls). In practice, bottom-up can be 2-3x faster. Use top-down when dependencies complex or not all subproblems needed. Use bottom-up for best performance.',
                    keyPoints: [
                        'Same asymptotic complexity',
                        'Bottom-up faster: no recursion overhead',
                        'Top-down: only needed subproblems',
                        'Bottom-up: better cache locality',
                        'Practice: bottom-up 2-3x faster',
                    ],
                },
                {
                    id: 'q3',
                    question:
                        'Explain the relationship between DP complexity and problem constraints. How do you estimate if DP will TLE (Time Limit Exceeded)?',
                    sampleAnswer:
                        'DP complexity determined by: number of states × transition cost. For n=1000, O(n^2) gives 1M operations - acceptable. O(n^3) gives 1B operations - might TLE depending on time limit. General rule: 10^8 operations per second, so 10^8 ops takes 1 second. For n=10^4, O(n^2)=10^8 is borderline. O(n^3)=10^12 will definitely TLE. Check problem constraints: if n <= 100, O(n^3) okay. If n <= 1000, need O(n^2). If n <= 10^5, need O(n log n) or better. DP states usually O(n) or O(n^2). Transition usually O(1) to O(n). Multiply to get total. If exceeds 10^8, consider optimizations or different approach.',
                    keyPoints: [
                        'Complexity: number of states × transition cost',
                        'Rule: 10^8 operations per second',
                        'n=10^4: O(n^2) borderline, O(n^3) TLE',
                        'Check constraints: n <= 100 → O(n^3) okay',
                        'Estimate: states × transitions, compare to 10^8',
                    ],
                },
            ],
            multipleChoice: [
                {
                    id: 'mc1',
                    question: 'What is the time complexity of typical 1D DP?',
                    options: [
                        'O(log N)',
                        'O(N) - iterate through N states, each state O(1) or O(K) work',
                        'O(N²)',
                        'O(2^N)',
                    ],
                    correctAnswer: 1,
                    explanation:
                        '1D DP: N states, each computed once. If each state does O(1) work, total O(N). If trying K choices per state, O(N*K). Example: climbing stairs O(N), coin change O(N*coins).',
                },
                {
                    id: 'mc2',
                    question: 'What is the time complexity of typical 2D DP?',
                    options: [
                        'O(N)',
                        'O(M*N) - M×N states, each state O(1) work',
                        'O(log N)',
                        'O(N³)',
                    ],
                    correctAnswer: 1,
                    explanation:
                        '2D DP: M×N states. If each state does O(1) work, total O(M*N). Examples: LCS O(M*N), edit distance O(M*N), grid paths O(M*N). With K choices per state: O(M*N*K).',
                },
                {
                    id: 'mc3',
                    question: 'What is the space complexity of bottom-up DP?',
                    options: [
                        'Always O(1)',
                        'O(number of states) but can often optimize to O(N) or O(1) by keeping only needed previous states',
                        'O(2^N)',
                        'Random',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Bottom-up: space = DP table size. 1D: O(N), 2D: O(M*N). But often optimizable: if dp[i] only needs dp[i-1], keep O(1). If dp[i][j] needs previous row, keep O(N).',
                },
                {
                    id: 'mc4',
                    question: 'What is the space complexity of top-down DP?',
                    options: [
                        'O(1)',
                        'O(states) for memo + O(depth) for recursion stack',
                        'O(N³)',
                        'Random',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Top-down: memo size O(number of states) + recursion stack O(maximum depth). Often both O(N), so total O(N). Less optimizable than bottom-up due to recursion stack.',
                },
                {
                    id: 'mc5',
                    question: 'How does DP complexity compare to brute force?',
                    options: [
                        'Same',
                        'DP avoids recomputation - often exponential O(2^N) → polynomial O(N²)',
                        'DP slower',
                        'Random',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'DP dramatically reduces complexity by caching. Fibonacci: O(2^N) → O(N). Knapsack: O(2^N) → O(N*W). DP polynomial vs brute force exponential. Trade-off: O(N) space.',
                },
            ],
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
            quiz: [
                {
                    id: 'q1',
                    question:
                        'Walk me through the 1D DP bottom-up template. What are the key steps?',
                    sampleAnswer:
                        'The 1D template has five parts. First, initialize DP array of size n+1 with default values (often 0 or infinity). Second, set base cases explicitly (dp[0], dp[1], etc.). Third, iterate from base cases to n. Fourth, for each i, compute dp[i] using recurrence relation (typically dp[i-1], dp[i-2], etc.). Fifth, return dp[n] as final answer. For Climbing Stairs: dp = array of size n+1, dp[0]=1, dp[1]=1 (base), loop i from 2 to n, dp[i] = dp[i-1] + dp[i-2] (recurrence), return dp[n] (answer). This template works for: Fibonacci, House Robber, Min Cost Climbing, Jump Game - any 1D problem with dependencies on previous states.',
                    keyPoints: [
                        'Initialize dp array (size n+1)',
                        'Set base cases explicitly',
                        'Iterate from base to n',
                        'Compute dp[i] using recurrence',
                        'Return dp[n] as answer',
                    ],
                },
                {
                    id: 'q2',
                    question:
                        'Explain the 2D DP template for problems like Knapsack or LCS. How do you handle the nested loops?',
                    sampleAnswer:
                        'The 2D template creates table dp[n+1][m+1]. First, initialize entire table (often with 0 or infinity). Second, set base cases for row 0 and column 0. Third, nested loops: outer loop i from 1 to n (first dimension), inner loop j from 1 to m (second dimension). Fourth, for each cell dp[i][j], compute using values from dp[i-1][...] and dp[...][j-1] based on recurrence. Fifth, return dp[n][m]. For LCS: outer loop for first string, inner loop for second string. If characters match: dp[i][j] = 1 + dp[i-1][j-1]. Else: dp[i][j] = max(dp[i-1][j], dp[i][j-1]). This template extends to: Knapsack, Edit Distance, Grid Paths.',
                    keyPoints: [
                        'Create 2D table dp[n+1][m+1]',
                        'Base cases: row 0 and column 0',
                        'Nested loops: i from 1 to n, j from 1 to m',
                        'Compute dp[i][j] from neighbors',
                        'Return dp[n][m]',
                    ],
                },
                {
                    id: 'q3',
                    question:
                        'Describe the top-down memoization template. How does it differ from bottom-up?',
                    sampleAnswer:
                        'Top-down uses recursion with caching. First, create memo dictionary or array. Second, write recursive helper with parameters representing state. Third, check if state in memo; if yes, return cached value. Fourth, if base case, return base value. Fifth, compute result using recursive calls to smaller subproblems. Sixth, store result in memo before returning. Seventh, call helper with initial state. For Fibonacci: helper(n) checks memo, if n <= 1 return n, else compute memo[n] = helper(n-1) + helper(n-2), return memo[n]. Differs from bottom-up: solves top-down (big to small), only computes needed states, has recursion overhead. Bottom-up: solves bottom-up (small to big), computes all states, uses iteration.',
                    keyPoints: [
                        'Create memo, write recursive helper',
                        'Check memo first, return if cached',
                        'Base case: return directly',
                        'Recurse for smaller subproblems',
                        'Store and return result',
                        'vs Bottom-up: top-down solving, only needed states',
                    ],
                },
            ],
            multipleChoice: [
                {
                    id: 'mc1',
                    question: 'What is the standard 1D DP template?',
                    options: [
                        'Random',
                        'Initialize dp array, set base cases, loop i from start to end, compute dp[i] from previous, return dp[n]',
                        'No template',
                        'Always recursive',
                    ],
                    correctAnswer: 1,
                    explanation:
                        '1D template: 1) Create dp[n+1], 2) Base case dp[0], 3) for i in 1..n: dp[i] = f(dp[i-1], dp[i-2],...), 4) return dp[n]. Works for most 1D problems.',
                },
                {
                    id: 'mc2',
                    question: 'What is the standard 2D DP template?',
                    options: [
                        'Random',
                        'Create dp[m][n], base cases first row/column, nested loops, dp[i][j] from neighbors, return dp[m-1][n-1]',
                        'No template',
                        'Only 1D',
                    ],
                    correctAnswer: 1,
                    explanation:
                        '2D template: 1) Create dp[m][n], 2) Base: dp[0][j] and dp[i][0], 3) for i,j: dp[i][j] = f(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]), 4) return dp[m-1][n-1] or max.',
                },
                {
                    id: 'mc3',
                    question: 'What is the top-down memoization template?',
                    options: [
                        'Random',
                        'Recursive function with memo dict, check memo first, compute and store if not cached, return',
                        'Only iterative',
                        'No template',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Memoization template: def solve(params, memo={}): if base_case: return value; if params in memo: return memo[params]; result = recurse; memo[params] = result; return result.',
                },
                {
                    id: 'mc4',
                    question: 'What is the knapsack template?',
                    options: [
                        'Random',
                        'dp[i][w] = max value with i items, capacity w. For each item: max(take, skip)',
                        'Sorting only',
                        'No template',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Knapsack: dp[i][w] for i items, capacity w. Recurrence: dp[i][w] = max(dp[i-1][w] skip, dp[i-1][w-weight[i]]+value[i] take). Return dp[n][W].',
                },
                {
                    id: 'mc5',
                    question: 'When should you use DP templates?',
                    options: [
                        'Never',
                        'Starting point to understand structure, adapt to specific problem, not rigid formula',
                        'Always exactly',
                        'Random',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Templates provide structure and common patterns. Use as starting point, adapt to problem specifics. Understand principles, don\'t memorize blindly. Each problem may need variations.',
                },
            ],
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
            quiz: [
                {
                    id: 'q1',
                    question:
                        'Walk me through your complete DP interview approach from problem statement to coded solution.',
                    sampleAnswer:
                        'First, recognize its DP: keywords like max/min/count, or recursive but slow. Second, clarify: constraints, edge cases, input size. Third, explain approach using 5 steps: state definition (what does dp[i] mean), recurrence (how to compute dp[i]), base cases, computation order, where answer is. Fourth, state complexity: number of states gives space, states times transition gives time. Fifth, draw small example: show first few values being computed. Sixth, code bottom-up solution with clear variable names and comments. Seventh, test with example: trace through computation. Eighth, discuss optimization: can we reduce space? Finally, mention related problems or variations. This systematic approach shows deep understanding and clear communication.',
                    keyPoints: [
                        'Recognize DP keywords and patterns',
                        'Clarify constraints and edge cases',
                        'Explain using 5-step framework',
                        'State complexity with reasoning',
                        'Draw example, trace computation',
                        'Code clearly with comments, test, optimize',
                    ],
                },
                {
                    id: 'q2',
                    question:
                        'How do you debug DP solutions when they give wrong answers? What is your systematic debugging process?',
                    sampleAnswer:
                        'First, test with smallest possible input: n=0, n=1, n=2. Does base case work? Second, manually compute dp table for small example and compare with code output. Third, add print statements showing dp values at each step. Fourth, check recurrence: is it mathematically correct? Draw decision tree. Fifth, verify computation order: are we accessing values before computing them? Sixth, check bounds: off-by-one errors in loops or array indices? Seventh, verify initialization: are default values correct? Eighth, for 2D, print entire table to visualize. Common bugs: wrong base case, wrong recurrence, wrong loop bounds, wrong initialization. Systematic approach: start small, verify manually, compare with code, identify first divergence.',
                    keyPoints: [
                        'Test smallest inputs: n=0, 1, 2',
                        'Manually compute and compare with code',
                        'Print dp values at each step',
                        'Verify recurrence mathematically',
                        'Check: computation order, bounds, initialization',
                        'Find first divergence between manual and code',
                    ],
                },
                {
                    id: 'q3',
                    question:
                        'What do you do when you cannot figure out the DP state or recurrence? What is your fallback strategy?',
                    sampleAnswer:
                        'First, solve smaller version manually (n=3, n=4) and look for pattern. How does answer for n=4 relate to n=3, n=2? Second, try writing brute force recursive solution. What parameters does recursion need? Those become DP state. What subproblems does it call? That hints at recurrence. Third, look for similar problems: is it like Fibonacci, Knapsack, LCS? Fourth, think about choices: at each step, what decisions can I make? Include/exclude, take/skip, etc. Fifth, define state as "answer for subproblem ending at i" or "answer using first i elements". Sixth, if stuck, mention what you know and ask for hint. Showing thought process is valuable even without complete solution.',
                    keyPoints: [
                        'Solve small manually, find pattern',
                        'Write brute force, parameters → state',
                        'Compare to known patterns',
                        'Think about choices at each step',
                        'State: answer for subproblem at/using i',
                        'Show thought process, ask for hints',
                    ],
                },
            ],
            multipleChoice: [
                {
                    id: 'mc1',
                    question: 'What keywords signal a DP problem?',
                    options: [
                        'Sort, search',
                        'Maximum/minimum, count ways, longest/shortest, optimal, can you reach',
                        'Shortest path only',
                        'Random',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'DP keywords: "maximum/minimum" (optimization), "count ways" (combinations), "longest/shortest" subsequence, "optimal", "can you reach/make". Suggests trying all possibilities optimally.',
                },
                {
                    id: 'mc2',
                    question: 'How do you approach a DP problem in an interview?',
                    options: [
                        'Code immediately',
                        'Define state, find recurrence, identify base cases, implement (top-down or bottom-up), optimize',
                        'Random',
                        'Guess pattern',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'DP approach: 1) Define state clearly (dp[i] means?), 2) Recurrence relation, 3) Base cases, 4) Implement (start top-down if easier), 5) Optimize space if needed. Communicate throughout.',
                },
                {
                    id: 'mc3',
                    question: 'What should you clarify in a DP interview?',
                    options: [
                        'Nothing',
                        'Constraints (N size affects O(N²) feasibility), output format (value vs path), edge cases',
                        'Random',
                        'Language only',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Clarify: 1) Input constraints (N≤1000 allows O(N²), N≤10^6 needs O(N)), 2) Output (optimal value vs actual solution path), 3) Edge cases (empty, single element), 4) Multiple solutions or one.',
                },
                {
                    id: 'mc4',
                    question: 'What is a common DP mistake?',
                    options: [
                        'Using arrays',
                        'Wrong state definition, incorrect base cases, wrong iteration order (dependencies)',
                        'Good naming',
                        'Comments',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Common mistakes: 1) Vague state definition, 2) Missing/wrong base cases, 3) Computing dp[i] before dependencies ready, 4) Off-by-one errors in indices, 5) Not handling edge cases.',
                },
                {
                    id: 'mc5',
                    question: 'How should you communicate your DP solution?',
                    options: [
                        'Just code',
                        'Explain state definition, recurrence relation, why it works, walk through example, complexity',
                        'No explanation',
                        'Random',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Communication: 1) Clear state definition (dp[i] represents...), 2) Recurrence with reasoning, 3) Base cases and why, 4) Walk through small example showing dp table, 5) Time O(?), space O(?), optimization possible.',
                },
            ],
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
