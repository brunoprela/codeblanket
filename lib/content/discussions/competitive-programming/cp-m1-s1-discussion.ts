export default {
  id: 'cp-m1-s1-discussion',
  title: 'Building Algorithmic Intuition - Discussion Questions',
  questions: [
    {
      question:
        'Why is it important to develop algorithmic intuition rather than just memorizing solutions, and how can you build this intuition systematically?',
      answer: `Developing algorithmic intuition is crucial for several reasons:

**Why Intuition Matters:**

1. **Novel Problems**: In contests, you face problems you've never seen before. Memorized solutions won't help—you need to understand WHY an approach works so you can adapt it.

2. **Pattern Recognition**: Intuition allows you to quickly recognize that a new problem is similar to patterns you've seen, even if the problem statement is completely different.

3. **Faster Problem Solving**: Instead of trying random approaches, intuition guides you toward promising solutions, saving precious time in contests.

4. **Debugging Ability**: Understanding why your algorithm works helps you debug when it doesn't—you can reason about edge cases and correctness.

**Building Intuition Systematically:**

1. **Solve Diverse Problems**: Don't just solve 100 array problems. Solve across all topics—trees, graphs, DP, greedy, etc. Each topic builds different intuition patterns.

2. **Understand, Don't Memorize**: After solving a problem, ask yourself:
   - WHY does this solution work?
   - What was the KEY insight?
   - When would this approach fail?
   - What similar problems could use this technique?

3. **Analyze Editorial Solutions**: When stuck, read editorials not to copy code but to understand the thought process. Try to identify the "aha moment" that led to the solution.

4. **Practice Pattern Recognition**: Group similar problems together. Notice that "find maximum sum subarray" and "best time to buy/sell stock" both use the same Kadane's algorithm pattern.

5. **Solve Without IDE First**: Spend 10-15 minutes with paper and pencil. Work through small examples. Draw diagrams. This builds the mental model that IS intuition.

6. **Teach Others**: Explaining a solution to someone else forces you to articulate the intuition, which solidifies your understanding.

7. **Upsolve Regularly**: After contests, solve problems you couldn't solve during the contest. This exposes you to new patterns and builds your intuition library.

**The 70/30 Rule**: Spend 70% of your time solving problems slightly above your level, and 30% reviewing and understanding solutions you couldn't solve. This balance builds intuition while challenging you.

**Example**: Instead of memorizing "use BFS for shortest path in unweighted graph," understand WHY: BFS explores layer by layer, guaranteeing we find each node at its shortest distance first. This understanding helps you recognize when BFS applies to seemingly different problems like "minimum moves to reach target" or "word ladder transformations."`,
    },
    {
      question:
        'Explain the difference between brute force, optimization, and the optimal solution. Why is it often helpful to start with a brute force approach even if you know it will be too slow?',
      answer: `Understanding the progression from brute force to optimal solution is fundamental to problem-solving in competitive programming.

**The Three Stages:**

1. **Brute Force**: Try all possible solutions, check each one
   - Example: Finding max in array → check every element
   - Time: Usually O(n²), O(2ⁿ), or O(n!)
   - Easy to implement, guaranteed correct (if implemented right)

2. **Optimization**: Improve brute force by eliminating redundant work
   - Example: Finding max → keep track of current max as you go
   - Time: Better than brute force, maybe O(n log n) or O(n)
   - Requires insight into what's redundant

3. **Optimal Solution**: The best possible time/space complexity
   - Example: Finding max → single pass, O(n)
   - Cannot be improved further (proven optimal)

**Why Start with Brute Force:**

1. **Correctness First**: A slow correct solution is better than a fast wrong solution. Brute force helps you understand the problem correctly.

2. **Baseline for Testing**: Brute force gives you a reference to test optimizations against. You can generate small test cases, run both solutions, and verify they match.

3. **Reveals Patterns**: Writing out the brute force often reveals redundant computations—these are exactly what you need to optimize!

4. **Partial Credit**: In some contests (ICPC, IOI), a brute force solution that passes small test cases earns partial points.

5. **Time Management**: If you're running low on time, a brute force that passes smaller constraints might be better than spending 20 more minutes trying to find the optimal solution and ending up with nothing.

**Real Example: Counting Pairs with Sum K**

Problem: Count pairs in array where a[i] + a[j] = k

**Brute Force O(n²):**
\`\`\`cpp
int count = 0;
for(int i = 0; i < n; i++) {
    for(int j = i+1; j < n; j++) {
        if(arr[i] + arr[j] == k) count++;
    }
}
\`\`\`

**Observation from Brute Force**: We're checking every pair and doing lookup—lots of repeated work!

**Optimization O(n log n):**
\`\`\`cpp
sort(arr, arr + n);  // Sort first
int count = 0;
int left = 0, right = n-1;
while(left < right) {
    int sum = arr[left] + arr[right];
    if(sum == k) { count++; left++; right--; }
    else if(sum < k) left++;
    else right--;
}
\`\`\`

**Optimal O(n):**
\`\`\`cpp
unordered_map<int, int> seen;
int count = 0;
for(int i = 0; i < n; i++) {
    if(seen.count(k - arr[i])) count += seen[k - arr[i]];
    seen[arr[i]]++;
}
\`\`\`

**The Journey**: Brute force → realize we're doing redundant lookups → use sorting to eliminate some → use hash map to make lookups O(1). Each step was guided by understanding the previous approach!

**Pro Tip**: In contests, if constraints are small (n ≤ 1000), brute force might actually be the optimal approach! Don't over-optimize.`,
    },
    {
      question:
        'Describe a systematic approach to solving competitive programming problems. What steps should you take from reading the problem to submitting your solution?',
      answer: `A systematic approach is crucial for consistency in competitive programming. Here's a battle-tested workflow:

**Phase 1: Understanding (5-10% of time)**

1. **Read Carefully**: Don't skim! Read every word, including constraints
2. **Identify Key Information**:
   - What is the input?
   - What output is required?
   - What are the constraints? (This tells you which algorithms are feasible!)
   - Are there multiple test cases?
3. **Understand Examples**: Work through sample inputs by hand. Why does input A produce output B?
4. **Clarify Edge Cases**: What happens when n=1? When all elements are the same? When array is sorted?

**Phase 2: Problem Solving (30-40% of time)**

1. **Restate the Problem**: In your own words, what are you actually trying to compute?
2. **Brainstorm Approaches**:
   - Start with brute force: "I could try every possibility..."
   - Think about problem category: Is this a graph problem? DP? Greedy?
   - Pattern recognition: Does this remind you of problems you've solved?
3. **Analyze Time Complexity**:
   - Calculate: With n=10⁵, can you afford O(n²)? (No!)
   - Constraints guide algorithms:
     - n ≤ 20: O(2ⁿ) or O(n!) might work (brute force/backtracking)
     - n ≤ 1000: O(n²) works
     - n ≤ 10⁵: Need O(n log n) or O(n)
     - n ≤ 10⁶: Need O(n) or O(log n)
4. **Verify on Examples**: Before coding, trace your algorithm on paper using the sample inputs. Does it work?

**Phase 3: Implementation (30-40% of time)**

1. **Start from Template**: Use your pre-prepared template
2. **Write Clean Code**:
   - Descriptive variable names (at least somewhat)
   - Break into functions if complex
   - Add comments for tricky parts
3. **Handle Input/Output Carefully**:
   - Check output format (spaces, newlines, "Case #1:", etc.)
   - Handle multiple test cases if needed
4. **Code Incrementally**: Write a bit, think about it, write more. Don't rush!

**Phase 4: Testing (10-20% of time)**

1. **Test on Samples**: Run on ALL provided examples
2. **Test Edge Cases**:
   - Minimum constraints (n=1)
   - Maximum constraints (n=10⁵)
   - All same values
   - Sorted/reverse sorted arrays
   - Zero, negative numbers
3. **Trace Through Manually**: Pick a small test case (n=3 or 4) and trace your code line by line
4. **Check for Common Bugs**:
   - Integer overflow? (Use long long)
   - Array out of bounds?
   - Off-by-one errors?
   - Uninitialized variables?

**Phase 5: Submission (1-2% of time)**

1. **Final Checks**:
   - Correct problem selected?
   - Output format matches exactly?
   - Fast I/O enabled?
   - No debug prints to stdout?
2. **Submit**: Click that button!
3. **While Waiting**: Don't refresh frantically. Start thinking about next problem.

**Phase 6: Post-Submission**

- **If AC**: Celebrate! Move to next problem
- **If WA**: Debug systematically (don't panic!)
  - Re-read problem (did you misunderstand?)
  - Check edge cases
  - Test on larger custom inputs
  - Add debug prints
- **If TLE**: Algorithm too slow, need better approach
- **If RTE**: Array bounds, division by zero, stack overflow

**Time Distribution Example (30-minute problem):**
- Understanding: 2-3 minutes
- Problem solving: 10-12 minutes
- Implementation: 10-12 minutes
- Testing: 3-5 minutes
- Submission: 1 minute

**Key Principles:**

1. **Measure Twice, Cut Once**: Time spent understanding and planning saves debugging time later
2. **Test Before Submit**: Catching bugs locally is faster than submitting and waiting for WA
3. **Stay Systematic**: Don't skip steps, especially under pressure
4. **Learn from Mistakes**: After contest, review what went wrong and adjust your process

**Pro Tip**: Practice this workflow during practice! In contests, muscle memory kicks in and you'll follow it naturally without thinking.`,
    },
  ],
} as const;
