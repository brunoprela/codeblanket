export const debuggingCompetitiveEnvironmentSection = {
    id: 'cp-m1-s13',
    title: 'Debugging in Competitive Environment',
    content: `

# Debugging in Competitive Environment

## Introduction

Your code compiles perfectly. You hit run. You get... **Wrong Answer**.

This is where the real challenge begins. Unlike compilation errors that scream at you with error messages, logical bugs sit quietly in your code, sabotaging test cases one by one. In competitive programming, you don't have the luxury of stepping through code with a debugger or running comprehensive test suites. You need to find and fix bugs **fast**.

The difference between a successful competitor and a struggling one often comes down to debugging efficiency. Top coders can spot bugs in 2-3 minutes that might take beginners 30+ minutes to find. How? They have systematic debugging strategies, effective print debugging techniques, and know how to generate test cases that expose bugs.

In this comprehensive section, we'll master competitive programming debugging: print debugging techniques, systematic bug hunting, generating edge cases, common logical errors, debugging strategies for different verdicts (WA, TLE, RTE, MLE), and more.

**Goal**: Master rapid debugging techniques to find and fix logical errors efficiently in contest environment.

---

## Understanding Judge Verdicts

Before debugging, you need to understand what each verdict means:

| Verdict | Meaning | Typical Causes |
|---------|---------|---------------|
| **AC** | Accepted | Everything correct! |
| **WA** | Wrong Answer | Logic error, edge case missed |
| **TLE** | Time Limit Exceeded | Algorithm too slow, infinite loop |
| **RTE** | Runtime Error | Segfault, array bounds, assertion |
| **MLE** | Memory Limit Exceeded | Too much memory used |
| **CE** | Compilation Error | Syntax error, covered in previous section |

**Most common:** WA (Wrong Answer) - focus of this section!

---

## The Debugging Mindset

### The Scientific Method for Debugging

1. **Observe**: What's the actual output?
2. **Hypothesize**: What might cause this?
3. **Test**: Add prints, try test cases
4. **Analyze**: Was hypothesis correct?
5. **Fix**: Correct the bug
6. **Verify**: Test again

**Don't:**
- Randomly change code hoping it works
- Try every possible fix without thinking
- Give up too quickly

**Do:**
- Be systematic
- Understand the bug before fixing
- Verify the fix works

---

## Strategy 1: Print Debugging (The CP Staple)

Print debugging is your **#1 weapon** in competitive programming. IDEs with debuggers are often not available or practical in contests.

### Basic Print Debugging

\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    cin >> n;
    
    cout << "DEBUG: n = " << n << endl;  // Debug print
    
    int sum = 0;
    for (int i = 0; i <= n; i++) {
        cout << "DEBUG: i = " << i << ", sum = " << sum << endl;
        sum += i;
    }
    
    cout << sum << endl;  // Actual output
    return 0;
}
\`\`\`

**Key points:**
- Prefix debug output with "DEBUG:" to distinguish from real output
- Print variable values at key points
- Print inside loops to see iterations

### The DEBUG Macro

Create a reusable debug macro:

\`\`\`cpp
#define DEBUG(x) cerr << #x << " = " << x << endl

int main() {
    int x = 42;
    vector<int> v = {1, 2, 3};
    
    DEBUG(x);  // Prints: x = 42
    DEBUG(v.size());  // Prints: v.size() = 3
}
\`\`\`

**Why \`cerr\`?**
- \`cerr\` goes to **stderr** (error stream)
- \`cout\` goes to **stdout** (standard output)
- Judge only sees stdout, not stderr
- Safe to leave debug prints in submitted code!

### Advanced DEBUG Macro

\`\`\`cpp
#ifdef LOCAL
#define DEBUG(x) cerr << #x << " = " << (x) << endl
#else
#define DEBUG(x)
#endif

// Compile with: g++ -DLOCAL solution.cpp
// Debug prints only show when LOCAL is defined!
\`\`\`

**Usage:**
\`\`\`cpp
DEBUG(x);  // Shows locally, invisible on judge
\`\`\`

### Debugging Containers

\`\`\`cpp
template<typename T>
void printVector(const vector<T>& v) {
    cerr << "[";
    for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) cerr << ", ";
        cerr << v[i];
    }
    cerr << "]" << endl;
}

// Usage:
vector<int> v = {1, 2, 3, 4, 5};
cerr << "v = "; printVector(v);
// Prints: v = [1, 2, 3, 4, 5]
\`\`\`

### Debugging Pairs and Tuples

\`\`\`cpp
template<typename T1, typename T2>
ostream& operator<<(ostream& os, const pair<T1, T2>& p) {
    return os << "(" << p.first << ", " << p.second << ")";
}

int main() {
    pair<int, int> p = {3, 5};
    cerr << "p = " << p << endl;  // Prints: p = (3, 5)
}
\`\`\`

### Complete Debug Template

\`\`\`cpp
#ifdef LOCAL
#define DEBUG(x) cerr << #x << " = " << (x) << endl
#define DEBUGV(v) cerr << #v << " = "; for(auto x : v) cerr << x << " "; cerr << endl
#else
#define DEBUG(x)
#define DEBUGV(v)
#endif

int main() {
    int x = 42;
    vector<int> v = {1, 2, 3, 4, 5};
    
    DEBUG(x);   // x = 42
    DEBUGV(v);  // v = 1 2 3 4 5
}
\`\`\`

---

## Strategy 2: Manual Test Case Generation

### Creating Test Cases

**Small test cases** are easier to debug than large ones!

\`\`\`cpp
// Instead of testing with n=100000:
// Test with n=5 first!

Input:
5
1 2 3 4 5

// Trace through by hand, see what should happen
\`\`\`

### Edge Cases to Always Test

**1. Minimum Constraints:**
\`\`\`
n = 1
Empty array
Single element
\`\`\`

**2. Maximum Constraints:**
\`\`\`
n = 10^5
All same values
All different values
\`\`\`

**3. Special Values:**
\`\`\`
Zero
Negative numbers
Very large numbers (near overflow)
\`\`\`

**4. Boundary Values:**
\`\`\`
First element
Last element
Middle element
\`\`\`

**5. Extremes:**
\`\`\`
All maximum values
All minimum values
Alternating values
Sorted array
Reverse sorted array
\`\`\`

### Test Case Generator

\`\`\`cpp
// Generate random test cases
#include <bits/stdc++.h>
using namespace std;

int main() {
    srand(time(0));
    
    int n = 10;
    cout << n << endl;
    
    for (int i = 0; i < n; i++) {
        int val = rand() % 100;  // Random 0-99
        cout << val << " ";
    }
    cout << endl;
    
    return 0;
}
\`\`\`

**Usage:**
\`\`\`bash
g++ generator.cpp -o gen
./gen > input.txt
./solution < input.txt
\`\`\`

---

## Strategy 3: Binary Search Your Code

If you have **AC on sample but WA on hidden tests**, use binary search debugging:

### Approach

1. **Add assert statements** throughout code
2. **Submit**
3. **If RTE**: Bug is before that assert
4. **If still WA**: Bug is after that assert
5. **Binary search** the bug location

\`\`\`cpp
int solve(int n) {
    assert(n > 0);  // Test 1: n is positive?
    
    int result = 0;
    for (int i = 0; i < n; i++) {
        result += i;
    }
    
    assert(result >= 0);  // Test 2: result is valid?
    
    return result;
}
\`\`\`

**If RTE on Test 2:** Bug is in the loop!

---

## Common Logical Errors in CP

### Error 1: Off-by-One Errors

**Problem:** Loop bounds are wrong

\`\`\`cpp
// Bug: Should be i < n, not i <= n
for (int i = 0; i <= n; i++) {  // One iteration too many!
    // ...
}

// Bug: Accessing arr[n] which is out of bounds
int arr[n];
for (int i = 0; i <= n; i++) {  // Should be i < n
    arr[i] = 0;  // Out of bounds when i == n!
}
\`\`\`

**Fix:**
\`\`\`cpp
// Correct
for (int i = 0; i < n; i++) {
    arr[i] = 0;
}
\`\`\`

**Testing tip:** Always test with n=1 to catch off-by-one errors!

### Error 2: Integer Overflow

**Problem:** Result exceeds int range

\`\`\`cpp
int a = 1000000;
int b = 1000000;
int c = a * b;  // Overflow! (10^12 > 2×10^9)

cout << c << endl;  // Wrong answer!
\`\`\`

**Fix:**
\`\`\`cpp
long long a = 1000000;
long long b = 1000000;
long long c = a * b;  // Correct

// Or:
int a = 1000000;
int b = 1000000;
long long c = (long long)a * b;  // Cast before multiply!
\`\`\`

**Testing tip:** Test with maximum values!

### Error 3: Uninitialized Variables

**Problem:** Variable not initialized

\`\`\`cpp
int sum;  // Garbage value!
for (int i = 0; i < n; i++) {
    sum += arr[i];  // Adding to garbage!
}
\`\`\`

**Fix:**
\`\`\`cpp
int sum = 0;  // Initialize!
for (int i = 0; i < n; i++) {
    sum += arr[i];
}
\`\`\`

**Testing tip:** Uninitialized variables might work locally (0 by chance) but fail on judge!

### Error 4: Wrong Comparison

**Problem:** Using = instead of ==

\`\`\`cpp
if (x = 5) {  // Assignment, not comparison!
    // This always executes (x becomes 5, which is true)
}
\`\`\`

**Fix:**
\`\`\`cpp
if (x == 5) {  // Comparison
    // ...
}
\`\`\`

### Error 5: Array Index Off

**Problem:** 0-indexed vs 1-indexed

\`\`\`cpp
int arr[n];
cin >> n;
for (int i = 1; i <= n; i++) {  // BUG: starts at 1!
    cin >> arr[i];  // arr[0] never initialized, arr[n] out of bounds!
}
\`\`\`

**Fix:**
\`\`\`cpp
int arr[n];
cin >> n;
for (int i = 0; i < n; i++) {  // 0-indexed
    cin >> arr[i];
}
\`\`\`

### Error 6: Floating Point Precision

**Problem:** Comparing doubles with ==

\`\`\`cpp
double x = 0.1 + 0.2;
if (x == 0.3) {  // Might be false due to precision!
    // ...
}
\`\`\`

**Fix:**
\`\`\`cpp
const double EPS = 1e-9;
double x = 0.1 + 0.2;
if (abs(x - 0.3) < EPS) {  // Compare with epsilon
    // ...
}
\`\`\`

### Error 7: Modifying Container While Iterating

**Problem:** Vector/set modified during iteration

\`\`\`cpp
for (int i = 0; i < v.size(); i++) {
    if (condition) {
        v.erase(v.begin() + i);  // Size changes!
        // Next iteration skips an element
    }
}
\`\`\`

**Fix:**
\`\`\`cpp
for (int i = v.size() - 1; i >= 0; i--) {  // Iterate backwards
    if (condition) {
        v.erase(v.begin() + i);
    }
}
\`\`\`

### Error 8: Wrong Data Structure

**Problem:** Using wrong container

\`\`\`cpp
// Checking if element exists - using vector (O(n))
vector<int> v;
// ...
if (find(v.begin(), v.end(), x) != v.end()) {  // Slow!
    // ...
}

// Should use set (O(log n)) or unordered_set (O(1))
set<int> s;
// ...
if (s.count(x)) {  // Fast!
    // ...
}
\`\`\`

---

## Debugging Different Verdicts

### Debugging WA (Wrong Answer)

**Systematic approach:**

1. **Re-read problem statement carefully**
   - Did you misunderstand something?
   - Check constraints
   - Check input/output format

2. **Test sample cases manually**
   - Trace through algorithm by hand
   - Does output match expected?

3. **Create small test cases**
   - Start with n=1, n=2
   - Test edge cases

4. **Add debug prints**
   - Print intermediate values
   - Verify algorithm steps

5. **Check for:**
   - Integer overflow
   - Off-by-one errors
   - Uninitialized variables
   - Wrong output format (extra spaces, newlines)

### Debugging TLE (Time Limit Exceeded)

**Common causes:**

**1. Algorithm too slow:**
\`\`\`cpp
// O(n²) when need O(n log n)
for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {  // Too slow for n=10^5!
        // ...
    }
}
\`\`\`

**Fix:** Optimize algorithm

**2. Infinite loop:**
\`\`\`cpp
int i = 0;
while (i < n) {
    // Forgot to increment i!
    // Infinite loop!
}
\`\`\`

**Fix:** Ensure loop terminates

**3. Slow I/O:**
\`\`\`cpp
// Reading 10^6 integers with cin is slow without optimization
\`\`\`

**Fix:**
\`\`\`cpp
ios_base::sync_with_stdio(false);
cin.tie(nullptr);
\`\`\`

**Testing:**
- Time your code locally
- Generate large test cases
- Use time command: \`time./ solution < input.txt\`

### Debugging RTE (Runtime Error)

**Common causes:**

**1. Array out of bounds:**
\`\`\`cpp
int arr[10];
arr[15] = 5;  // Segmentation fault
\`\`\`

**Fix:** Check array bounds

**2. Division by zero:**
\`\`\`cpp
int x = 10 / 0;  // Runtime error
\`\`\`

**Fix:** Check divisor != 0

**3. Null pointer:**
\`\`\`cpp
int* ptr = nullptr;
*ptr = 5;  // Segmentation fault
\`\`\`

**Fix:** Check pointer validity

**4. Stack overflow:**
\`\`\`cpp
int arr[10000000];  // Too large for stack
\`\`\`

**Fix:** Use global array or vector

**5. Assertion failure:**
\`\`\`cpp
assert(x > 0);  // Fails if x <= 0
\`\`\`

**Fix:** Remove assert or fix logic

**Testing:**
- Run with valgrind (Linux): \`valgrind./ solution\`
- Compile with sanitizer: \`g++ - fsanitize=address solution.cpp\`
- Test edge cases thoroughly

### Debugging MLE (Memory Limit Exceeded)

**Common causes:**

**1. Array too large:**
\`\`\`cpp
int arr[100000000];  // 400 MB! Exceeds limit
\`\`\`

**Fix:** Optimize space complexity

**2. Memory leak:**
\`\`\`cpp
while (true) {
    int* arr = new int[1000000];  // Never freed!
}
\`\`\`

**Fix:** Delete allocated memory or use vector

**3. Unnecessary data structures:**
\`\`\`cpp
// Storing all intermediate results unnecessarily
\`\`\`

**Fix:** Only store what you need

---

## The Rubber Duck Method

When stuck, explain your code **out loud** (to a rubber duck, or yourself):

1. "I read n numbers"
2. "I sort them"
3. "I find the maximum"
4. "I... wait, why am I sorting? I only need the maximum!"

Often, explaining reveals the bug!

---

## Debugging Workflow

### Standard Debugging Process

\`\`\`
1. Code compiles ✓
2. Test sample inputs
   → Pass? Go to 3
   → Fail? Debug with prints, fix, repeat
3. Think of edge cases
   → Test n=1, n=max, zeros, negatives, etc.
   → All pass? Go to 4
   → Fail? Debug, fix, repeat
4. Submit
   → AC? Celebrate! 🎉
   → WA? Go to 5
   → TLE? Check time complexity
   → RTE? Check array bounds, division by zero
   → MLE? Check memory usage
5. Review problem statement
   → Misunderstood something? Fix and go to 2
   → Still stuck? Generate more test cases
6. Add assertions and debug prints
7. Submit with assertions
   → RTE? Bug location found!
8. Binary search the bug
9. Fix and go to 4
\`\`\`

---

## Tips and Tricks

### Quick Debugging Tips

**1. Comment out suspicious code:**
\`\`\`cpp
// If unsure which part is buggy, comment out sections
// to isolate the problem
\`\`\`

**2. Simplify the code:**
\`\`\`cpp
// Replace complex logic with simple brute force
// If brute force works, bug is in optimization
\`\`\`

**3. Check variable names:**
\`\`\`cpp
// Typos like:
for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; i++) {  // BUG: i++ instead of j++
        // ...
    }
}
\`\`\`

**4. Print before crash:**
\`\`\`cpp
for (int i = 0; i < n; i++) {
    cerr << "Processing i = " << i << endl;
    // If crashes, you'll see which i caused it
}
\`\`\`

**5. Use different input:**
\`\`\`cpp
// If stuck, try completely different test case
// Sometimes reveals pattern you missed
\`\`\`

---

## Summary

**Key Debugging Strategies:**

✅ **Print debugging with cerr** (doesn't affect judge output)
✅ **Test small cases first** (easier to trace)
✅ **Test edge cases** (n=1, n=max, zeros, etc.)
✅ **Binary search the bug** (use assertions)
✅ **Re-read problem** (might have misunderstood)
✅ **Check common errors** (overflow, off-by-one, etc.)

**Debugging Different Verdicts:**

✅ **WA**: Logic error, test edge cases
✅ **TLE**: Algorithm too slow or infinite loop
✅ **RTE**: Array bounds, division by zero, stack overflow
✅ **MLE**: Arrays too large, optimize space

**Golden Rules:**

1. **Be systematic** - don't randomly change code
2. **Test small first** - n=1, n=2, then larger
3. **Use print debugging** - cerr is your friend
4. **Check edge cases** - most bugs hide there
5. **Verify your fix** - test again after fixing

**Time-Saving Tips:**

- Build a debug template
- Practice common bug patterns
- Keep calm and debug systematically
- Learn from your bugs

---

## Next Steps

Now let's learn about **Reading Other People's C++ Code** - a crucial skill for learning from others and understanding editorial solutions!

**Key Takeaway**: Debugging is a skill that improves with practice. Build a systematic approach, use effective print debugging, test edge cases thoroughly, and learn to recognize common bug patterns. Every bug you fix makes you better at avoiding similar bugs in the future!
`,
    quizId: 'cp-m1-s13-quiz',
        discussionId: 'cp-m1-s13-discussion',
} as const ;
