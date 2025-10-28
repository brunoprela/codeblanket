export const contestDayCppTipsSection = {
    id: 'cp-m1-s15',
    title: 'Contest-Day C++ Tips',
    content: `

# Contest-Day C++ Tips

## Introduction

It's contest day. The timer starts. Your heart races. You open the first problem... and suddenly everything you practiced goes out the window. You make silly mistakes, waste time on bugs, and watch precious minutes tick away.

**The difference between a good contest and a great one often isn't algorithmic knowledge—it's execution under pressure.** The ability to write fast, correct code while the clock is ticking. The discipline to follow a systematic approach. The mental clarity to avoid panic-induced bugs.

This section is your **contest-day survival guide**. These are battle-tested tips from thousands of contests: pre-contest preparation, during-contest workflow, code templates that save time, common mistakes to avoid, time management, and mental strategies.

**Goal**: Master practical tips and workflows to maximize performance on contest day.

---

## Pre-Contest Preparation

### The Night Before

**DO:**
✅ Get good sleep (7-8 hours minimum)
✅ Prepare your workspace
✅ Test your setup (compiler, IDE, internet)
✅ Review your template code
✅ Light practice (1-2 easy problems for warmup)
✅ Prepare snacks and water

**DON'T:**
❌ Stay up late grinding problems
❌ Learn new algorithms the night before
❌ Drink excessive caffeine late
❌ Stress about the contest

**Sleep is more important than an extra hour of practice!**

### Setup Checklist

\`\`\`
✅ IDE configured and tested
✅ Compiler working (test with a simple program)
✅ Internet connection stable
✅ Contest platform account logged in
✅ Template file ready
✅ Notepad/paper for sketching
✅ Calculator nearby (if needed)
✅ Phone on silent
✅ Comfortable chair and good lighting
\`\`\`

### Your Template File

Have a well-tested template ready:

\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

#define ll long long
#define all(v) v.begin(), v.end()
#define rep(i,n) for(int i=0; i<(n); i++)

// Add your commonly used functions here
template<typename T>
T gcd(T a, T b) { return b ? gcd(b, a%b) : a; }

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

**Customize this template based on your preferences!**

---

## Contest Start Strategy

### First 5 Minutes

**1. Skim all problems (1-2 minutes)**
- Get overview of difficulty
- Identify easy problems
- Note any unusual constraints

**2. Read A completely (2 minutes)**
- Understand statement
- Check constraints
- Look at examples

**3. Decide on approach (1 minute)**
- Can I solve it?
- What algorithm?
- Any edge cases?

**DON'T** spend 30 minutes reading all problems in detail. Read, solve, submit, repeat!

### Problem Selection Strategy

**Start with easiest problem** (usually A or B)

**Why?**
- Build confidence
- Get points on board
- Understand contest difficulty
- Warm up your brain

**After solving A:**
- Quickly read B
- If B looks easy, solve it
- If B looks hard, check C
- Find the next easiest problem

**Rule of thumb:** Solve problems you can solve in 15-30 minutes. Don't get stuck on hard problems early!

---

## During-Contest Workflow

### The Systematic Approach

**For EACH problem:**

\`\`\`
1. READ carefully (don't skim!)
   - Understand what's asked
   - Note constraints
   - Check examples

2. THINK of approach
   - What algorithm?
   - Time complexity OK?
   - Edge cases?

3. VERIFY approach on examples
   - Trace by hand
   - Does it work?

4. CODE the solution
   - Start from template
   - Write clean code
   - Add comments for complex parts

5. TEST locally
   - Run on sample inputs
   - Try edge cases
   - Check output format

6. SUBMIT
   - Double-check problem letter
   - Submit!

7. While waiting for verdict:
   - Think about next problem
   - Don't refresh obsessively
   - Stay calm

8. IF AC:
   - Celebrate briefly
   - Move to next problem

9. IF WA/TLE/RTE:
   - Debug systematically
   - Don't panic
   - Fix and resubmit
\`\`\`

### Time Management

**Example 2-hour contest:**
- Problem A: 10-15 minutes
- Problem B: 15-20 minutes
- Problem C: 20-30 minutes
- Problem D: 30-40 minutes
- Buffer: 15 minutes for debugging

**Golden rule:** If stuck on a problem for 30+ minutes with no progress, **move on**. Come back later with fresh eyes.

---

## Fast Coding Techniques

### Use Macros Wisely

**Helpful macros:**
\`\`\`cpp
#define ll long long
#define all(v) v.begin(), v.end()
#define rep(i,n) for(int i=0; i<(n); i++)
#define pb push_back
#define mp make_pair
#define F first
#define S second
\`\`\`

**Don't overdo it:**
\`\`\`cpp
// TOO MUCH:
#define si(x) scanf("%d",&x)
#define pi(x) printf("%d\\n",x)
#define fr(i,n) for(i=0;i<n;i++)
#define frd(i,n) for(i=n-1;i>=0;i--)
// This is hard to read and error-prone
\`\`\`

**Balance:** Use macros for common patterns, but keep code readable!

### Fast Input/Output

\`\`\`cpp
ios_base::sync_with_stdio(false);
cin.tie(nullptr);
\`\`\`

**Add this to every solution!** Makes cin/cout as fast as scanf/printf.

**WARNING:** Don't mix cin/cout with scanf/printf after using this!

### Reading Arrays Quickly

\`\`\`cpp
// Pattern 1: Traditional
vector<int> a(n);
for(int i = 0; i < n; i++) cin >> a[i];

// Pattern 2: Range-based
vector<int> a(n);
for(auto& x : a) cin >> x;

// Pattern 3: Function
vector<int> read(int n) {
    vector<int> v(n);
    for(auto& x : v) cin >> x;
    return v;
}
auto a = read(n);
\`\`\`

Pick one pattern and stick with it!

### Common Operations Shortcuts

\`\`\`cpp
// Sorting
sort(all(v));  // With macro: all(v) = v.begin(), v.end()

// Reversing
reverse(all(v));

// Finding max/min element
int mx = *max_element(all(v));
int mn = *min_element(all(v));

// Counting occurrences
int cnt = count(all(v), x);

// Unique elements
sort(all(v));
v.erase(unique(all(v)), v.end());

// Sum
ll sum = accumulate(all(v), 0LL);  // 0LL for long long
\`\`\`

---

## Common Contest-Day Mistakes

### Mistake 1: Not Reading Carefully

**Wrong:**
- Skim problem statement
- Miss important details
- Solve wrong problem

**Right:**
- Read EVERY word
- Check constraints
- Understand examples
- Ask yourself: "What exactly is being asked?"

**Time saved by reading carefully: 10-30 minutes**

### Mistake 2: Not Testing Samples

**Wrong:**
- Write solution
- Submit immediately
- Get WA
- Waste time debugging

**Right:**
- Write solution
- Test on ALL sample inputs
- Verify output format
- Then submit

**Sample tests are FREE—use them!**

### Mistake 3: Forgetting Edge Cases

**Common edge cases:**
\`\`\`cpp
// Test these:
n = 1           // Minimum
n = 10^5        // Maximum
All same values
All different values
Zeros
Negative numbers
Sorted array
Reverse sorted array
\`\`\`

**Make a mental checklist!**

### Mistake 4: Integer Overflow

\`\`\`cpp
// WRONG:
int a = 100000;
int b = 100000;
int c = a * b;  // Overflow!

// RIGHT:
ll a = 100000;
ll b = 100000;
ll c = a * b;  // Correct

// OR:
int a = 100000;
int b = 100000;
ll c = (ll)a * b;  // Cast before multiply
\`\`\`

**Rule:** If result might exceed 2×10^9, use \`long long\`!

### Mistake 5: Wrong Output Format

\`\`\`cpp
// Problem asks for: "Case #1: 42"
// You output: "42"
// Result: WA

// Always check output format!
\`\`\`

**Read output section carefully!**

### Mistake 6: Modifying While Iterating

\`\`\`cpp
// WRONG:
for(int i = 0; i < v.size(); i++) {
    if(condition) v.erase(v.begin() + i);  // Size changes!
}

// RIGHT:
for(int i = v.size()-1; i >= 0; i--) {
    if(condition) v.erase(v.begin() + i);  // Backwards
}
\`\`\`

### Mistake 7: Forgetting Multiple Test Cases

\`\`\`cpp
// Problem: "First line contains T, number of test cases"

// WRONG:
void solve() { ... }
int main() {
    solve();  // Only solves one test case!
}

// RIGHT:
void solve() { ... }
int main() {
    int t; cin >> t;
    while(t--) solve();
}
\`\`\`

**Check if problem has multiple test cases!**

### Mistake 8: Array Index Out of Bounds

\`\`\`cpp
// WRONG:
int arr[n];
for(int i = 0; i <= n; i++) {  // Goes to arr[n]!
    arr[i] = 0;  // Out of bounds
}

// RIGHT:
int arr[n];
for(int i = 0; i < n; i++) {  // Only to arr[n-1]
    arr[i] = 0;
}
\`\`\`

**Always check loop bounds!**

---

## Code Quality vs Speed Trade-offs

### What to Prioritize in Contest

**DO prioritize:**
✅ Correctness
✅ Speed of writing
✅ Easy to debug

**DON'T prioritize:**
❌ Perfect variable names
❌ Detailed comments
❌ Modular design
❌ Reusability

**Contest code is throwaway code!**

### When to Write Clean vs Fast

**Write cleaner code when:**
- Problem is complex
- Likely to have bugs
- Need to debug
- You have time

**Write faster code when:**
- Problem is simple
- Time is tight
- You're confident in solution
- Pattern is familiar

**Balance based on situation!**

---

## Debugging Under Time Pressure

### Quick Debug Strategies

**Strategy 1: Print Everything**
\`\`\`cpp
cerr << "x = " << x << endl;
cerr << "After loop: i = " << i << endl;
\`\`\`

**Strategy 2: Test Smallest Case**
- n = 1
- n = 2
- Trace by hand

**Strategy 3: Comment Out Sections**
- Isolate buggy part
- Test each section separately

**Strategy 4: Rewrite from Scratch**
- If truly stuck after 15 minutes
- Sometimes faster than debugging

**Strategy 5: Check Common Mistakes**
- Integer overflow?
- Array bounds?
- Off-by-one?
- Output format?

### When to Give Up and Move On

**Give up if:**
- Stuck for 30+ minutes
- No progress on debugging
- Other problems look easier
- Time is running out

**Come back later with:**
- Fresh perspective
- Clearer mind
- Better ideas

**Stubborn persistence can waste time!**

---

## Mental Game

### Staying Calm Under Pressure

**When you get WA:**
1. Take a deep breath
2. Don't panic
3. Read problem again
4. Check samples
5. Debug systematically

**When you're stuck:**
1. It's OK to be stuck
2. Everyone struggles
3. Move to next problem
4. Come back later

**When time is running out:**
1. Focus on what you CAN do
2. Finish current problem if close
3. Don't start new complex problem with 10 minutes left
4. Review previous solutions for bugs

### Building Contest Stamina

**Practice regularly:**
- Virtual contests
- Time yourself
- Simulate pressure
- Build endurance

**2-hour contest = mental marathon!**

---

## Platform-Specific Tips

### Codeforces

- **Pretest vs System Test:** AC on pretest doesn't guarantee final AC!
- **Hack Phase:** In Div 2, others can hack your solution
- **Point decay:** Faster submissions earn more points
- **Rating changes:** Based on performance vs expected

**Strategy:**
- Don't rush too much (risk of hacks)
- Test edge cases thoroughly
- Fast submissions on A/B matter for rating

### AtCoder

- **English/Japanese:** Choose your language
- **Time limits:** Often tight
- **Strong test cases:** Less likely to pass with wrong solution
- **Editorial:** Available after contest (in Japanese and English)

**Strategy:**
- Time limits are real—optimize if needed
- Test cases are thorough—if AC, likely correct

### LeetCode Contests

- **Weekly/Biweekly:** Regular schedule
- **4 problems:** Usually increasing difficulty
- **Rank by:** Solved count, then time
- **Fast testing:** Run tests quickly

**Strategy:**
- Speed matters for ranking
- Optimize fast solutions
- Submit early if confident

---

## Pre-Submit Checklist

**Before clicking Submit:**

\`\`\`
✅ Tested on ALL sample inputs?
✅ Output format correct?
✅ Edge cases considered?
✅ Array bounds checked?
✅ Integer overflow handled?
✅ Multiple test cases handled (if applicable)?
✅ Fast I/O added?
✅ Correct problem selected?
✅ No debug prints to stdout?
\`\`\`

**30 seconds of checking can save 10 minutes of debugging!**

---

## Post-Contest Routine

### After Contest Ends

**1. Celebrate achievements**
- Solved problems?
- Personal best?
- Learned something?

**2. Review mistakes**
- What went wrong?
- What bugs occurred?
- What could be better?

**3. Read editorials**
- Understand solutions
- Learn new techniques
- See alternate approaches

**4. Read others' code**
- See clean implementations
- Learn coding tricks
- Compare approaches

**5. Upsolve**
- Solve problems you couldn't during contest
- No time pressure
- Solidify learning

**6. Update your template**
- Add useful functions
- Fix any template bugs
- Improve for next time

**Every contest is learning!**

---

## Summary

**Pre-Contest:**
✅ Good sleep
✅ Test setup
✅ Prepare template
✅ Stay calm

**During Contest:**
✅ Read carefully
✅ Start with easiest
✅ Test before submit
✅ Manage time
✅ Stay systematic

**Common Mistakes to Avoid:**
❌ Not reading carefully
❌ Skipping sample tests
❌ Ignoring edge cases
❌ Integer overflow
❌ Array bounds
❌ Wrong output format
❌ Getting stuck too long

**Time Management:**
- Easy problems: 10-20 min
- Medium problems: 20-30 min
- If stuck 30+ min, move on
- Come back with fresh eyes

**Mental Tips:**
- Stay calm
- Don't panic on WA
- Take breaks when stuck
- Focus on what you can do

**Key Principles:**

1. **Correctness > Speed** (but both matter)
2. **Test before submit** (saves time overall)
3. **Read carefully** (solve right problem)
4. **Manage time** (don't get stuck)
5. **Stay calm** (panic causes bugs)

---

## Your Personal Checklist

Create your own pre-contest checklist based on YOUR common mistakes:

\`\`\`
My common mistakes:
[ ] _______________________
[ ] _______________________
[ ] _______________________

Pre-submit checks:
[ ] _______________________
[ ] _______________________
[ ] _______________________
\`\`\`

**Track your mistakes and improve!**

---

## Next Steps

Now let's complete the foundational knowledge with **Building Your Starter Template** - creating a robust, personalized template for competitive programming!

**Key Takeaway**: Contest performance isn't just about algorithms—it's about execution, discipline, and mental game. Develop a systematic workflow, avoid common mistakes, manage your time, and stay calm under pressure. Every contest is an opportunity to refine your process. Build good habits in practice, and they'll serve you well in competition!
`,
    quizId: 'cp-m1-s15-quiz',
    discussionId: 'cp-m1-s15-discussion',
} as const ;
