export default {
  id: 'cp-m1-s15-discussion',
  title: 'Contest-Day C++ Tips - Discussion Questions',
  questions: [
    {
      question:
        'Contest day introduces unique pressures and time constraints that affect how you should write C++. Describe a complete contest-day workflow including pre-contest preparation, during-contest strategies, and common pitfalls to avoid.',
      answer: `Contest day requires a different mindset than practice. Here's the complete competitive workflow:

**Pre-Contest Preparation (30 minutes before):**

**1. Environment Setup**
\`\`\`bash
# Check compiler version
g++ --version  # Should be C++17 or later

# Test compilation
echo 'int main(){}' > test.cpp
g++ -std=c++17 -O2 test.cpp
./a.out

# Verify fast I/O works
echo '#include<iostream>
using namespace std;
int main(){
    ios_base::sync_with_stdio(0);cin.tie(0);
    int x;cin>>x;cout<<x;
}' > test.cpp
g++ test.cpp && echo "5" | ./a.out
\`\`\`

**2. Template Ready**
\`\`\`cpp
// Save as template.cpp
#include <bits/stdc++.h>
using namespace std;

#define ll long long
#define pii pair<int,int>
#define vi vector<int>
#define pb push_back
#define all(x) (x).begin(),(x).end()
#define rep(i,n) for(int i=0;i<(n);i++)

#ifdef LOCAL
#define debug(x) cerr << #x << " = " << x << endl
#else
#define debug(x)
#endif

void solve() {
    // Solution goes here
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int t = 1;
    // cin >> t;  // Uncomment for multiple test cases
    
    while(t--) {
        solve();
    }
    
    return 0;
}
\`\`\`

**3. Shortcuts/Aliases Ready**
\`\`\`bash
# Add to shell (bash/zsh)
alias c='g++ -std=c++17 -O2 -Wall -Wextra -DLOCAL'
alias r='./a.out < input.txt'
alias t='c solution.cpp && r'
\`\`\`

**4. Directory Structure**
\`\`\`
contest/
├── A/
│   ├── solution.cpp
│   ├── input.txt
│   └── output.txt
├── B/
│   ├── solution.cpp
│   ├── input.txt
│   └── output.txt
├── C/ ... etc
└── template.cpp
\`\`\`

**5. Mental Preparation**
- ✅ Bathroom break
- ✅ Water bottle ready
- ✅ Snacks nearby
- ✅ Phone on silent
- ✅ Distractions minimized

**Contest Start (First 5 minutes):**

**1. Read ALL problems quickly (skim)**
- Don't solve yet!
- Get overview of difficulty
- Identify easiest problems

**2. Choose starting problem**
- Usually A or B (easiest)
- Check problem constraints
- Note time limit

**During Contest Workflow:**

**For Each Problem:**

**Step 1: Read Carefully (2-3 minutes)**
\`\`\`
✓ Read problem statement twice
✓ Understand input/output format
✓ Note constraints (n ≤ ?, time limit?)
✓ Check for special cases
✓ Read sample explanations
\`\`\`

**Step 2: Plan Solution (2-5 minutes)**
\`\`\`
✓ What algorithm? (brute force, greedy, DP, etc.)
✓ Time complexity OK?
✓ Edge cases?
✓ Write pseudocode mentally
\`\`\`

**Step 3: Code (5-15 minutes)**
\`\`\`cpp
// Copy template
cp ../template.cpp solution.cpp

// Write solution
void solve() {
    int n; cin >> n;
    // ... your code ...
    cout << result << '\\n';
}
\`\`\`

**Key coding principles:**
- Write simple code first
- Avoid clever tricks unless necessary
- Use standard library when possible
- Comment complex parts

**Step 4: Test on Samples (2-3 minutes)**
\`\`\`bash
# Copy sample input
echo "3
1 2 3" > input.txt

# Compile and run
c solution.cpp && r

# Check output matches expected
\`\`\`

**Step 5: Test Edge Cases (2 minutes)**
\`\`\`
✓ n = 1 (minimum)
✓ n = max (maximum from constraints)
✓ All elements same
✓ Sorted input
✓ Reverse sorted
\`\`\`

**Step 6: Submit!**

**Time Management:**

**Typical 2-hour contest (4-5 problems):**
- Problem A: 10-15 minutes
- Problem B: 15-20 minutes
- Problem C: 20-30 minutes
- Problem D: 30-45 minutes
- Problem E: If time remains

**Budget per problem:**
- Reading: 3 min
- Planning: 5 min
- Coding: 10-15 min
- Testing: 3 min
- Debugging (if needed): 5 min
- **Total: ~25-30 minutes**

**Critical Contest-Day Tips:**

**1. Fast I/O Always**
\`\`\`cpp
ios_base::sync_with_stdio(false);
cin.tie(nullptr);
\`\`\`
**Never forget this for large inputs!**

**2. Use Long Long Liberally**
\`\`\`cpp
// When in doubt, use ll
ll sum = 0;  // Not int
for(int i = 0; i < n; i++) {
    sum += arr[i];  // Might overflow with int
}
\`\`\`

**3. Read Constraints Carefully**
\`\`\`cpp
// If n ≤ 10^5 and need O(n^2)?
// That's 10^10 operations - TLE!
// Need O(n log n) or O(n)
\`\`\`

**4. Check Output Format**
\`\`\`cpp
// Problem says: Print "YES" or "NO"
cout << "YES" << endl;  // Not "Yes" or "yes"!

// Problem says: Print answer modulo 10^9+7
const ll MOD = 1e9 + 7;
cout << (ans % MOD) << endl;
\`\`\`

**5. Multiple Test Cases**
\`\`\`cpp
// If problem says "t test cases"
int t; cin >> t;
while(t--) {
    solve();
}

// Common mistake: Forgetting to reset variables
void solve() {
    // Clear global arrays/variables!
    memset(dp, 0, sizeof(dp));
    graph.clear();
}
\`\`\`

**Common Contest-Day Mistakes:**

**Mistake 1: Not Testing Before Submit**
- ALWAYS test on samples
- Test at least one edge case
- Cost: Wrong Answer penalty

**Mistake 2: Integer Overflow**
\`\`\`cpp
int a = 100000, b = 100000;
int product = a * b;  // OVERFLOW!

// Fix:
ll product = (ll)a * b;
\`\`\`

**Mistake 3: Array Out of Bounds**
\`\`\`cpp
int arr[100000];
// If n can be 100000, and you access arr[n], it's out of bounds!

// Fix: Size = max + 1
int arr[100001];
\`\`\`

**Mistake 4: Wrong Loop Bounds**
\`\`\`cpp
// If array is 0-indexed but problem is 1-indexed
for(int i = 1; i <= n; i++) {
    cin >> arr[i];  // arr starts at index 1
}
\`\`\`

**Mistake 5: Not Clearing Between Test Cases**
\`\`\`cpp
vector<int> graph[MAXN];  // Global

void solve() {
    // BUG: Not clearing from previous test case!
    
    // Fix:
    for(int i = 0; i <= n; i++) {
        graph[i].clear();
    }
}
\`\`\`

**Mistake 6: Printing Extra Spaces/Lines**
\`\`\`cpp
// Problem wants: "1 2 3"
// You print: "1 2 3 " (extra space at end)

// Fix:
for(int i = 0; i < n; i++) {
    if(i > 0) cout << " ";
    cout << arr[i];
}
cout << endl;
\`\`\`

**Mistake 7: Wrong Answer, Keep Trying Same Approach**
- If 2 wrong answers, RETHINK approach
- Check edge cases again
- Read problem statement again
- Consider different algorithm

**Advanced Contest Strategies:**

**Strategy 1: Solve Easiest First**
- Build confidence
- Gain points quickly
- More time for hard problems

**Strategy 2: Skip if Stuck**
- Stuck >15 minutes? Move on
- Come back later with fresh perspective
- Don't lose the contest on one problem

**Strategy 3: Penalty Awareness**
- Wrong answer = time penalty
- Make sure before submitting
- One correct submit > three wrong submits

**Strategy 4: Read Announcements**
- Clarifications might be posted
- Check every 15-20 minutes

**Strategy 5: Strategic Guessing**
- If no idea and time running out
- Submit brute force (might pass small tests)
- Better than nothing

**Contest-Day Debugging:**

**When you get WA (Wrong Answer):**

1. **Re-test samples** (did you break something?)
2. **Test edge cases:**
   \`\`\`
   n = 1
   n = maximum
   All same elements
   All different elements
   \`\`\`
3. **Check for:**
   - Integer overflow
   - Array out of bounds
   - Wrong output format
   - Not clearing between test cases

**When you get TLE (Time Limit Exceeded):**

1. **Check complexity:**
   - O(n²) with n=10^5 = TLE
   - Need better algorithm
2. **Missing fast I/O?**
3. **Infinite loop?**

**When you get RTE (Runtime Error):**

1. **Array out of bounds?**
2. **Division by zero?**
3. **Stack overflow? (too much recursion)**

**Last 15 Minutes of Contest:**

- ✅ Check all submissions
- ✅ If stuck, submit something (even brute force)
- ✅ Don't start new complex problem
- ✅ Review previous solutions for bugs

**Post-Contest:**

1. **Check editorial** immediately
2. **Upsolve problems** you couldn't solve
3. **Analyze mistakes**
4. **Practice weak areas**

**Contest Day Checklist:**

**Before contest:**
\`\`\`
✓ Template ready
✓ Compiler working
✓ Shortcuts configured
✓ Internet stable
✓ Mentally prepared
\`\`\`

**During contest (per problem):**
\`\`\`
✓ Read carefully
✓ Check constraints
✓ Plan algorithm
✓ Code solution
✓ Test samples
✓ Test edge cases
✓ Submit
\`\`\`

**Mental State:**

- 🔥 Stay calm under pressure
- 🔥 Don't panic if stuck
- 🔥 Time management crucial
- 🔥 One problem at a time
- 🔥 Learn from every contest

**Bottom Line:**

Contest-day success:
- ✅ Preparation before contest
- ✅ Time management during
- ✅ Test before submit
- ✅ Check constraints
- ✅ Stay calm
- ✅ Learn from mistakes

**Practice makes perfect. The more contests you do, the better you'll get!**`,
    },
    {
      question:
        "Time pressure can lead to bugs that wouldn't happen during practice. Describe the most common contest-induced bugs and preventive measures to avoid them.",
      answer: `Time pressure creates specific bug patterns. Here's how to recognize and prevent them:

**Bug Category 1: Integer Overflow**

**Why it happens:** Rushed, forget to use long long

**Example:**
\`\`\`cpp
// Bug:
int n = 100000;
int sum = 0;
for(int i = 1; i <= n; i++) {
    sum += i;  // sum of 1 to 100000 = 5,000,050,000
}
// Overflow! int max = 2,147,483,647
cout << sum;  // Wrong answer
\`\`\`

**Fix:**
\`\`\`cpp
int n = 100000;
ll sum = 0;  // Use long long!
for(int i = 1; i <= n; i++) {
    sum += i;
}
cout << sum;  // Correct
\`\`\`

**Prevention:**
- ✅ Use \`ll\` for sums, products, cumulative values
- ✅ Cast before multiplication: \`(ll)a * b\`
- ✅ Check: Can result exceed 10^9? Use long long!

**Bug Category 2: Array Out of Bounds**

**Why it happens:** Off-by-one errors under pressure

**Example:**
\`\`\`cpp
// Bug:
int arr[100000];
int n = 100000;
for(int i = 0; i <= n; i++) {  // Goes to 100000!
    arr[i] = i;  // arr[100000] is out of bounds!
}
\`\`\`

**Fix:**
\`\`\`cpp
int arr[100001];  // Size = max + 1
// Or:
for(int i = 0; i < n; i++) {  // Correct bound
    arr[i] = i;
}
\`\`\`

**Prevention:**
- ✅ Use \`i < n\`, not \`i <= n\` for 0-indexed
- ✅ Array size = max + 1
- ✅ Double-check loop bounds

**Bug Category 3: Uninitialized Variables**

**Why it happens:** Forget to initialize in rush

**Example:**
\`\`\`cpp
// Bug:
int max_val;  // Uninitialized!
for(int i = 0; i < n; i++) {
    if(arr[i] > max_val) {  // Comparing with garbage
        max_val = arr[i];
    }
}
\`\`\`

**Fix:**
\`\`\`cpp
int max_val = -1e9;  // Or INT_MIN
for(int i = 0; i < n; i++) {
    if(arr[i] > max_val) {
        max_val = arr[i];
    }
}

// Or use STL:
int max_val = *max_element(all(arr));
\`\`\`

**Prevention:**
- ✅ Always initialize variables
- ✅ Use: \`int x = 0;\` not \`int x;\`
- ✅ For max: Initialize to -INF
- ✅ For min: Initialize to +INF

**Bug Category 4: Not Clearing Between Test Cases**

**Why it happens:** Forget global state persists

**Example:**
\`\`\`cpp
// Bug:
vector<int> graph[MAXN];  // Global
bool visited[MAXN];

void solve() {
    // Not cleared from previous test case!
    // Old data still there!
    
    bfs(start);  // Uses old graph + visited data!
}

int main() {
    int t; cin >> t;
    while(t--) solve();
}
\`\`\`

**Fix:**
\`\`\`cpp
void solve() {
    // Clear everything!
    for(int i = 0; i <= n; i++) {
        graph[i].clear();
        visited[i] = false;
    }
    
    // Or use memset:
    memset(visited, false, sizeof(visited));
    
    bfs(start);  // Now correct
}
\`\`\`

**Prevention:**
- ✅ Clear global arrays/vectors at start of solve()
- ✅ Or use local variables instead
- ✅ Test with multiple test cases locally

**Bug Category 5: Wrong Output Format**

**Why it happens:** Don't read output format carefully

**Example:**
\`\`\`cpp
// Problem wants: "YES" or "NO"
// Bug:
cout << "Yes" << endl;  // Wrong case!
cout << "yes" << endl;  // Wrong case!

// Problem wants: Case #X: answer
// Bug:
cout << result << endl;  // Missing "Case #X:"

// Problem wants: Space-separated, no trailing space
// Bug:
for(int i = 0; i < n; i++) {
    cout << arr[i] << " ";  // Trailing space!
}
\`\`\`

**Fix:**
\`\`\`cpp
// Exact format:
cout << "YES" << endl;  // Correct case

// Case numbering:
cout << "Case #" << case_num << ": " << result << endl;

// No trailing space:
for(int i = 0; i < n; i++) {
    if(i > 0) cout << " ";
    cout << arr[i];
}
cout << endl;
\`\`\`

**Prevention:**
- ✅ Read output format twice
- ✅ Check sample output exactly
- ✅ Test output format on samples

**Bug Category 6: Wrong Loop Direction**

**Why it happens:** Confused about indexing

**Example:**
\`\`\`cpp
// Problem uses 1-indexed
// Bug:
for(int i = 0; i < n; i++) {
    cin >> arr[i];  // Reading into 0-indexed!
}

// But problem expects 1-indexed:
// "Process elements 1 to n"
for(int i = 1; i <= n; i++) {
    process(arr[i]);  // Accessing arr[n] out of bounds!
}
\`\`\`

**Fix:**
\`\`\`cpp
// Be consistent!
for(int i = 1; i <= n; i++) {
    cin >> arr[i];  // 1-indexed
}

for(int i = 1; i <= n; i++) {
    process(arr[i]);  // Same indexing
}

// Or convert problem to 0-indexed mentally
\`\`\`

**Prevention:**
- ✅ Stick to one indexing (prefer 0-indexed)
- ✅ If problem is 1-indexed, be explicit
- ✅ Array size = n + 1 for 1-indexed

**Bug Category 7: Comparison Mistake**

**Why it happens:** Typo under pressure

**Example:**
\`\`\`cpp
// Bug:
if(x = 5) {  // Assignment, not comparison!
    // Always executes!
}

// Bug:
if(x > 5); {  // Semicolon!
    // Always executes!
}
\`\`\`

**Fix:**
\`\`\`cpp
if(x == 5) {  // Correct comparison
    // ...
}

if(x > 5) {  // No semicolon
    // ...
}
\`\`\`

**Prevention:**
- ✅ Enable compiler warnings (-Wall)
- ✅ Double-check conditions
- ✅ Use \`if(5 == x)\` (won't compile if typo)

**Bug Category 8: Modulo Mistakes**

**Why it happens:** Forget to apply modulo, or apply incorrectly

**Example:**
\`\`\`cpp
// Problem: Output answer modulo 10^9+7

// Bug:
int ans = 0;
for(int i = 0; i < n; i++) {
    ans += arr[i];  // No modulo - might overflow!
}
cout << ans % MOD;  // Too late!

// Bug:
ll ans = (a - b) % MOD;  // If a < b, negative result!
\`\`\`

**Fix:**
\`\`\`cpp
const ll MOD = 1e9 + 7;

ll ans = 0;
for(int i = 0; i < n; i++) {
    ans = (ans + arr[i]) % MOD;  // Apply at each step
}
cout << ans;

// For subtraction:
ll ans = ((a - b) % MOD + MOD) % MOD;  // Handle negative
\`\`\`

**Prevention:**
- ✅ Apply modulo at every operation
- ✅ Handle negative with +MOD
- ✅ Use \`ll\` for modulo operations

**Bug Category 9: Sorting with Wrong Comparator**

**Why it happens:** Rush custom comparator

**Example:**
\`\`\`cpp
// Bug:
sort(all(arr), [](int a, int b) {
    return a <= b;  // Should be <, not <=!
});
// Undefined behavior!
\`\`\`

**Fix:**
\`\`\`cpp
sort(all(arr), [](int a, int b) {
    return a < b;  // Strict weak ordering
});

// For descending:
sort(all(arr), greater<int>());
\`\`\`

**Prevention:**
- ✅ Use \`<\`, never \`<=\`
- ✅ Test comparator logic
- ✅ Use standard \`greater<>()\` when possible

**Bug Category 10: Reading Input Wrong**

**Why it happens:** Misread format under pressure

**Example:**
\`\`\`cpp
// Input format: "n m" then "n numbers" then "m numbers"

// Bug:
int n, m;
cin >> n >> m;
vector<int> arr(n + m);  // Combined!
for(int i = 0; i < n + m; i++) {
    cin >> arr[i];  // Wrong!
}

// Should be separate:
vector<int> a(n), b(m);
for(int i = 0; i < n; i++) cin >> a[i];
for(int i = 0; i < m; i++) cin >> b[i];
\`\`\`

**Prevention:**
- ✅ Read input format carefully
- ✅ Test on all samples
- ✅ Check if input matches expected

**Contest-Specific Prevention Strategies:**

**Strategy 1: Defensive Initialization**
\`\`\`cpp
// Always initialize
int max_val = -1e9;
int min_val = 1e9;
ll sum = 0;
bool found = false;
\`\`\`

**Strategy 2: Boundary Assertions**
\`\`\`cpp
#ifdef LOCAL
assert(i >= 0 && i < n);
assert(sum <= 1e18);
#endif
\`\`\`

**Strategy 3: Template Checklist**
\`\`\`cpp
// In template, check:
✓ Fast I/O included?
✓ Long long defined?
✓ Common macros ready?
✓ Debug macros with #ifdef?
\`\`\`

**Strategy 4: Pre-Submit Checklist**

Before clicking submit:
\`\`\`
✓ Tested on all samples?
✓ Tested n=1?
✓ Tested n=max?
✓ Used long long where needed?
✓ Output format correct?
✓ Cleared between test cases?
\`\`\`

**Bug Frequency (My Experience):**

1. Integer overflow: **40%**
2. Array out of bounds: **20%**
3. Wrong output format: **15%**
4. Not clearing between tests: **10%**
5. Uninitialized variables: **5%**
6. Other: **10%**

**Time-Saving Debug Techniques:**

**Technique 1: Binary Search Debugging**
\`\`\`cpp
void solve() {
    cerr << "Checkpoint 1" << endl;
    // ... code ...
    cerr << "Checkpoint 2" << endl;
    // ... code ...
    cerr << "Checkpoint 3" << endl;
}
// Find which checkpoint doesn't print
\`\`\`

**Technique 2: Minimal Test Case**
\`\`\`
// Don't test n=100000 first!
// Test n=1, n=2, n=3
// Find smallest failing case
\`\`\`

**Technique 3: Compare with Brute Force**
\`\`\`cpp
// If optimized solution wrong
// Code simple O(n²) version
// Compare outputs
\`\`\`

**Bottom Line:**

Contest bugs:
- ✅ Overflow: Use long long
- ✅ Bounds: Check loop limits
- ✅ Init: Initialize all variables
- ✅ Clear: Reset between test cases
- ✅ Format: Read output carefully
- ✅ Test: Always test before submit

**Prevention > Debugging. Build good habits in practice!**`,
    },
    {
      question:
        'Managing stress and making strategic decisions under time pressure is crucial. Discuss psychological and strategic approaches to handling difficult contests, including when to skip problems, when to rewrite code, and how to recover from mistakes.',
      answer: `Contest psychology and strategy are as important as coding skills. Here's the complete mental game:

**The Psychological Reality:**

Contests are stressful:
- ⏰ Ticking clock
- 📊 Real-time rankings
- 💭 Self-doubt when stuck
- 😤 Frustration after WA
- 🏃 Rush to finish

**Managing stress is a skill you must develop!**

**Strategic Decision-Making:**

**Decision 1: When to Skip a Problem**

**Signs to skip:**
- ❌ No idea after 5 minutes
- ❌ Implementation seems very complex
- ❌ Tried 2 approaches, both wrong
- ❌ Other easier problems available
- ❌ Time running out

**When to skip:**
\`\`\`
Time spent: 15-20 minutes
Progress: Minimal
Better option: Yes (easier problem available)
→ SKIP NOW
\`\`\`

**How to skip:**
1. Mark problem mentally (come back?)
2. Move to next problem
3. Don't dwell on it
4. Fresh problem = fresh mindset

**Example scenario:**
\`\`\`
120-minute contest, 4 problems
Current time: 60 minutes
Solved: A, B (50 minutes)
Stuck: C (10 minutes, no progress)
Remaining: D

Decision: Skip C, try D
Reasoning: D might be easier than C for you
           Can return to C with remaining time
\`\`\`

**Decision 2: When to Rewrite Code**

**Rewrite if:**
- ✅ Bug is deep in logic
- ✅ Tried 3+ fixes, still wrong
- ✅ Code is messy/confusing
- ✅ Have clearer approach now
- ✅ <10 minutes investment

**Don't rewrite if:**
- ❌ Minor bug likely
- ❌ >20 minutes to rewrite
- ❌ Not confident in new approach
- ❌ Time running out

**Example:**
\`\`\`cpp
// Current code: 50 lines, buggy, can't find issue
// Time spent debugging: 10 minutes
// Time to rewrite with cleaner approach: 8 minutes

→ REWRITE!

// Current code: 150 lines, mostly working, one small bug
// Time spent debugging: 5 minutes
// Time to rewrite: 30 minutes

→ KEEP DEBUGGING!
\`\`\`

**Decision 3: When to Change Approach**

**Change if:**
- ✅ Current approach clearly wrong
- ✅ TLE on samples
- ✅ Constraint analysis shows won't work
- ✅ Have better idea

**Stick if:**
- ✅ Approach is sound
- ✅ Just implementation bugs
- ✅ No better alternative

**Example:**
\`\`\`
Problem: n ≤ 10^5
Your solution: O(n²) = 10^10 operations

→ CHANGE APPROACH (will TLE)

Problem: n ≤ 100
Your solution: O(n²) = 10^4 operations  

→ KEEP APPROACH (fast enough)
\`\`\`

**Strategic Time Management:**

**The 2-Hour Contest Strategy:**

\`\`\`
0-10 min:   Read all problems, choose easiest
10-30 min:  Solve problem A
30-50 min:  Solve problem B
50-80 min:  Solve problem C or D
80-110 min: Solve remaining solvable problem
110-120 min: Desperate attempts / review
\`\`\`

**Key principle: Solve what you can solve!**

**Problem Selection Strategy:**

**Option 1: Easiest First (Recommended)**
- Solve A, B, C in order
- Build confidence
- Secure guaranteed points
- More time for hard problems

**Option 2: High-Value First**
- If point values differ
- Solve highest-value solvable problems
- Risk: Might miss easy points

**Option 3: Personal Strength**
- Some people good at DP
- Some good at graphs
- Play to your strengths

**My recommendation: Easiest first for most contests**

**Handling Wrong Answer (WA):**

**First WA:**
\`\`\`
1. Don't panic! (Everyone gets WA)
2. Re-test on samples (did you break something?)
3. Check obvious bugs:
   - Integer overflow?
   - Array bounds?
   - Output format?
4. Test edge cases:
   - n = 1
   - n = max
   - All same
5. Fix and resubmit
\`\`\`

**Second WA:**
\`\`\`
1. Something fundamentally wrong
2. Re-read problem statement
3. Check understanding of problem
4. Test more edge cases
5. Consider different approach
\`\`\`

**Third WA:**
\`\`\`
1. Maybe skip this problem
2. Come back later
3. Or accept penalty and move on
4. Don't let one problem ruin contest
\`\`\`

**Psychological Strategies:**

**Strategy 1: Positive Self-Talk**

**Bad:** "I'm stuck, I'm terrible, I'll never solve this"
**Good:** "This is challenging, but I can figure it out"

**Bad:** "Everyone else solved it, I'm the worst"
**Good:** "Others have different strengths, I'll solve what I can"

**Bad:** "One WA, contest ruined"
**Good:** "WA is normal, learn and resubmit"

**Strategy 2: Breathing and Reset**

When frustrated:
\`\`\`
1. Close eyes
2. Deep breath for 10 seconds
3. Stand up, stretch
4. Fresh perspective
5. Continue with clear mind
\`\`\`

**Takes 30 seconds, saves 10 minutes of tilted coding!**

**Strategy 3: Compartmentalize**

Don't think about:
- ❌ Your rank (distracting)
- ❌ How others are doing (irrelevant)
- ❌ Previous contests (past is past)
- ❌ Rating change (can't control)

Focus on:
- ✅ Current problem
- ✅ Your code
- ✅ Testing thoroughly
- ✅ Solving correctly

**Strategy 4: Embrace the Struggle**

**Mindset shift:**
- "I'm stuck" → "I'm learning"
- "This is hard" → "This is growth"
- "I failed" → "I practiced"

**Every contest makes you better, regardless of result!**

**Recovering from Mistakes:**

**Scenario 1: Wasted 30 Minutes on Wrong Approach**

**Bad reaction:**
- Panic
- Rage-quit
- Give up

**Good reaction:**
1. Accept the loss
2. Move to next problem
3. Focus on what's left
4. Learn for next time

\`\`\`
Remaining: 90 minutes
Can still solve: 2-3 problems
→ Focus and recover!
\`\`\`

**Scenario 2: Silly Bug Cost 20 Minutes**

**Bad reaction:**
- Beat yourself up
- Lose confidence
- Make more mistakes

**Good reaction:**
1. It happens to everyone
2. Learn from it
3. Add to checklist
4. Move forward

**Scenario 3: Misread Problem**

**Bad reaction:**
- "I'm so stupid"
- Lose focus

**Good reaction:**
1. Understandable under pressure
2. Read more carefully next time
3. Solve correctly now
4. Continue

**Contest Types and Strategies:**

**Short Contest (1-2 hours):**
- Speed is critical
- Solve fast, test fast
- No time for complex problems
- Focus on accuracy

**Long Contest (3-5 hours):**
- Can afford to think
- Try harder problems
- Careful implementation
- Thorough testing

**Virtual Contest (Practice):**
- Experiment with strategies
- Try hard problems
- Learn without pressure

**Endgame Strategies (Last 15 minutes):**

**If stuck on current problem:**
1. Submit something (even if uncertain)
2. Don't waste last minutes on hopeless case
3. Review previous solutions for bugs

**If confident in current problem:**
1. Test thoroughly
2. Submit carefully
3. Don't rush and make mistakes

**If everything solved:**
1. Review all solutions
2. Look for bugs
3. Try next problem (if confident)

**Mental Resilience Building:**

**Practice:**
- Do many contests (builds tolerance to pressure)
- Virtual contests (simulate pressure)
- Upsolve after every contest (learn from mistakes)

**Preparation:**
- Strong fundamentals (reduces uncertainty)
- Good template (saves mental energy)
- Practiced workflow (automatic actions)

**Perspective:**
- One contest ≠ your ability
- Bad days happen to everyone
- Long-term improvement matters

**Common Mental Traps:**

**Trap 1: Comparison**
"Others solved 5, I solved 3, I'm bad"
→ Everyone has different backgrounds, experience

**Trap 2: Perfectionism**
"Must solve perfectly, no mistakes"
→ Mistakes are normal, learn and improve

**Trap 3: All-or-Nothing**
"If I don't solve all, I failed"
→ Progress is progress, every solve counts

**Trap 4: Rating Obsession**
"Rating must go up every contest"
→ Focus on learning, rating follows

**My Personal Contest Mantra:**

\`\`\`
1. Read carefully
2. Think clearly
3. Code simply
4. Test thoroughly
5. Learn constantly
\`\`\`

**Bottom Line:**

Contest psychology:
- ✅ Stay calm under pressure
- ✅ Skip when stuck (15-20 min limit)
- ✅ Rewrite if deeply bugged
- ✅ Change approach if clearly wrong
- ✅ Recover quickly from mistakes
- ✅ Focus on your own progress
- ✅ Learn from every contest

**Mental game is 50% of competitive programming success!**

Remember: Every red coder was once gray. Every hard problem was once impossible. You'll get there with practice and persistence! 🚀`,
    },
  ],
} as const;
