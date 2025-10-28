export default {
  id: 'cp-m1-s13-discussion',
  title: 'Debugging in Competitive Environment - Discussion Questions',
  questions: [
    {
      question:
        'Debugging during a contest is different from normal development - you have time pressure and usually no debugger access. Describe effective debugging strategies specific to competitive programming contests.',
      answer: `Contest debugging is all about speed. Here's the complete strategic approach:

**The Core Principle: Output Inspection > Debugger**

In contests:
- No debugger available (usually)
- Time pressure is intense
- Need to debug in <2 minutes

**Solution: Strategic print statements**

**Level 1: Basic Debug Prints**

**The debug macro:**
\`\`\`cpp
#ifdef LOCAL
#define debug(x) cerr << #x << " = " << x << endl
#else
#define debug(x)
#endif

int main() {
    int x = 42;
    debug(x);  // Prints: x = 42
    
    // Submit without changing code!
    // On judge: debug() does nothing
}
\`\`\`

**Compile with:**
\`\`\`bash
g++ -DLOCAL solution.cpp  # Local testing
g++ solution.cpp          # Submission
\`\`\`

**Level 2: Debug Multiple Values**

\`\`\`cpp
#ifdef LOCAL
#define debug2(x, y) cerr << #x << " = " << x << ", " << #y << " = " << y << endl
#define debug3(x, y, z) cerr << #x << " = " << x << ", " << #y << " = " << y << ", " << #z << " = " << z << endl
#else
#define debug2(x, y)
#define debug3(x, y, z)
#endif

debug2(i, sum);  // i = 5, sum = 15
debug3(a, b, c); // a = 1, b = 2, c = 3
\`\`\`

**Level 3: Debug Containers**

\`\`\`cpp
#ifdef LOCAL
template<typename T>
void debug_vector(const vector<T>& v, const string& name) {
    cerr << name << " = [";
    for(size_t i = 0; i < v.size(); i++) {
        if(i > 0) cerr << ", ";
        cerr << v[i];
    }
    cerr << "]" << endl;
}
#else
#define debug_vector(v, name)
#endif

vector<int> arr = {1, 2, 3, 4, 5};
debug_vector(arr, "arr");  // arr = [1, 2, 3, 4, 5]
\`\`\`

**Systematic Debugging Process:**

**Step 1: Identify WHERE the bug is**

Binary search approach:
\`\`\`cpp
void solve() {
    // Part 1: Input
    int n; cin >> n;
    vector<int> arr(n);
    for(int i = 0; i < n; i++) cin >> arr[i];
    cerr << "✓ Input read" << endl;  // Checkpoint 1
    
    // Part 2: Processing
    int result = process(arr);
    cerr << "✓ Processing done, result = " << result << endl;  // Checkpoint 2
    
    // Part 3: Output
    cout << result << endl;
    cerr << "✓ Output printed" << endl;  // Checkpoint 3
}
\`\`\`

**If checkpoint 2 doesn't print, bug is in process()!**

**Step 2: Narrow down the exact line**

\`\`\`cpp
int process(vector<int>& arr) {
    int sum = 0;
    cerr << "Starting process, arr.size() = " << arr.size() << endl;
    
    for(int i = 0; i < arr.size(); i++) {
        sum += arr[i];
        cerr << "i = " << i << ", arr[i] = " << arr[i] << ", sum = " << sum << endl;
    }
    
    return sum;
}
\`\`\`

**Step 3: Check your assumptions**

Common wrong assumptions:
\`\`\`cpp
// Assumption: Array is sorted
debug_vector(arr, "Before sort");
sort(arr.begin(), arr.end());
debug_vector(arr, "After sort");

// Assumption: n is within bounds
cerr << "n = " << n << " (expected < 1000)" << endl;

// Assumption: No duplicates
set<int> s(arr.begin(), arr.end());
cerr << "Unique elements: " << s.size() << "/" << arr.size() << endl;
\`\`\`

**Common Bug Patterns:**

**Bug Type 1: Array Index Out of Bounds**

Symptom: Segmentation fault or wrong answer

Debug:
\`\`\`cpp
for(int i = 0; i < n; i++) {
    cerr << "Accessing i = " << i << " (size = " << arr.size() << ")" << endl;
    if(i >= arr.size()) {
        cerr << "ERROR: Index out of bounds!" << endl;
        break;
    }
    // ... use arr[i]
}
\`\`\`

**Bug Type 2: Integer Overflow**

Symptom: Negative result when positive expected

Debug:
\`\`\`cpp
int a = 1000000, b = 1000000;
cerr << "a = " << a << ", b = " << b << endl;
int product = a * b;  // Overflow!
cerr << "product (int) = " << product << endl;  // Negative or wrong

long long product_ll = (long long)a * b;
cerr << "product (ll) = " << product_ll << endl;  // Correct
\`\`\`

**Bug Type 3: Wrong Loop Bounds**

Symptom: Missing elements or extra iteration

Debug:
\`\`\`cpp
// Should be i < n, but wrote i <= n
for(int i = 0; i <= n; i++) {  // Bug: extra iteration
    cerr << "i = " << i << " (n = " << n << ")" << endl;
    // ...
}
\`\`\`

**Bug Type 4: Uninitialized Variables**

Symptom: Random results

Debug:
\`\`\`cpp
int sum;  // Uninitialized!
cerr << "sum before init = " << sum << endl;  // Garbage value
sum = 0;  // Fix
cerr << "sum after init = " << sum << endl;  // 0
\`\`\`

**Bug Type 5: Wrong Comparison**

Symptom: Logic errors

Debug:
\`\`\`cpp
if(x = 5) {  // Bug: Assignment instead of comparison!
    cerr << "This always executes!" << endl;
}

// Should be:
if(x == 5) {
    cerr << "x is 5" << endl;
}
\`\`\`

**Time-Saving Debug Strategies:**

**Strategy 1: Test on Sample Before Custom Cases**

\`\`\`cpp
// Sample input from problem
3
1 2 3

// Your code
cerr << "Testing sample input..." << endl;
// Run and compare with expected output
\`\`\`

**Strategy 2: Use Assertions**

\`\`\`cpp
#include <cassert>

void solve() {
    int n; cin >> n;
    assert(n > 0 && n <= 100000);  // Verify constraint
    
    vector<int> arr(n);
    for(int i = 0; i < n; i++) {
        cin >> arr[i];
        assert(arr[i] >= 0);  // Verify element constraint
    }
}
\`\`\`

**If assertion fails, you know exactly where!**

**Strategy 3: Compare with Brute Force**

\`\`\`cpp
// Your optimized solution
int fast_solve(int n) {
    // O(log n) but might have bugs
    return /* ... */;
}

// Simple but correct solution
int slow_solve(int n) {
    // O(n) brute force
    int result = 0;
    for(int i = 1; i <= n; i++) {
        result += i;
    }
    return result;
}

// Test
for(int n = 1; n <= 100; n++) {
    int fast = fast_solve(n);
    int slow = slow_solve(n);
    if(fast != slow) {
        cerr << "Mismatch at n = " << n << endl;
        cerr << "Fast = " << fast << ", Slow = " << slow << endl;
        break;
    }
}
\`\`\`

**Strategy 4: Minimal Test Cases**

Don't test complex cases first!

\`\`\`
// Start with:
n = 1, arr = [1]
n = 2, arr = [1, 2]
n = 2, arr = [2, 1]

// Then increase complexity
\`\`\`

**Strategy 5: Add Timing Checks**

\`\`\`cpp
#include <chrono>

auto start = chrono::high_resolution_clock::now();

solve();

auto end = chrono::high_resolution_clock::now();
auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
cerr << "Time: " << duration.count() << " ms" << endl;
\`\`\`

**Advanced: Conditional Debugging**

\`\`\`cpp
#ifdef LOCAL
bool DEBUG_MODE = true;
#define debug_if(cond, x) if(cond && DEBUG_MODE) cerr << #x << " = " << x << endl
#else
#define debug_if(cond, x)
#endif

// Only debug when i == 5
for(int i = 0; i < n; i++) {
    debug_if(i == 5, arr[i]);
}
\`\`\`

**Common Debugging Mistakes:**

**Mistake 1: Adding too many debug prints**
- Output becomes unreadable
- Solution: Use conditional debugging

**Mistake 2: Forgetting to remove debug code**
- Gets judged as wrong (extra output)
- Solution: Use #ifdef LOCAL

**Mistake 3: Not testing edge cases**
- n = 1
- n = maximum value  
- All elements same
- All elements different

**Mistake 4: Debugging wrong part**
- Bug is in input reading, but you debug logic
- Solution: Checkpoint approach

**Quick Debug Template:**

\`\`\`cpp
#ifdef LOCAL
#define debug(x) cerr << #x << " = " << x << endl
#define debug2(x,y) cerr << #x << " = " << x << ", " << #y << " = " << y << endl
template<typename T>
void debugv(const vector<T>& v, const string& name) {
    cerr << name << " = [";
    for(size_t i = 0; i < v.size(); i++) {
        if(i > 0) cerr << ", ";
        cerr << v[i];
    }
    cerr << "]" << endl;
}
#else
#define debug(x)
#define debug2(x,y)
#define debugv(v, name)
#endif
\`\`\`

**Debugging Checklist (2-minute drill):**

When you get wrong answer:
\`\`\`
1. Test on sample input (30 seconds)
2. Add checkpoints to find WHERE (30 seconds)
3. Print values around bug location (30 seconds)
4. Fix and retest (30 seconds)
\`\`\`

**If still stuck:**
\`\`\`
5. Test edge cases (n=1, n=max)
6. Compare with brute force
7. Check for: overflow, uninitialized vars, wrong bounds
\`\`\`

**Real Contest Example:**

Problem: Sum of array

Wrong solution:
\`\`\`cpp
int sum = 0;
for(int i = 1; i <= n; i++) {  // Bug: should start at 0
    sum += arr[i];
}
\`\`\`

Debug:
\`\`\`cpp
int sum = 0;
for(int i = 1; i <= n; i++) {
    debug2(i, arr[i]);  // Prints: i = 1, arr[1] = ...
    // Wait, array is 0-indexed!
    sum += arr[i];
}
\`\`\`

Fix:
\`\`\`cpp
for(int i = 0; i < n; i++) {  // Correct
    sum += arr[i];
}
\`\`\`

**Bottom Line:**

Contest debugging:
- ✅ Use debug macros with #ifdef LOCAL
- ✅ Checkpoint approach to find WHERE
- ✅ Print values to find WHAT
- ✅ Test edge cases
- ✅ Compare with brute force
- ✅ Budget 2 minutes max per bug

**Fast debugging = more time for solving!**`,
    },
    {
      question:
        'Creating custom test cases is crucial for debugging complex problems. Explain how to generate effective test cases, including edge cases, stress tests, and random tests.',
      answer: `Good test cases catch bugs BEFORE submission. Here's the complete testing strategy:

**The Testing Pyramid:**

\`\`\`
        /\\
       /  \\  Sample Tests (given)
      /____\\
     /      \\  Edge Cases (you create)
    /________\\
   /          \\  Random/Stress Tests (generated)
  /____________\\
\`\`\`

**Level 1: Sample Tests (Always Start Here)**

\`\`\`cpp
// Problem gives:
Input:
3
1 2 3

Expected Output:
6

// Test:
./solution < sample1.txt
// Compare with expected
\`\`\`

**If samples pass but submission fails → need more tests!**

**Level 2: Manual Edge Cases**

**Edge Case Categories:**

**1. Boundary Values**
\`\`\`cpp
// If constraint is 1 ≤ n ≤ 100000:

// Test n = 1 (minimum)
1
5

// Test n = 100000 (maximum)
100000
1 2 3 ... 100000

// Test n = 2 (just above minimum)
2
1 2
\`\`\`

**2. Special Values**
\`\`\`cpp
// Zero
1
0

// Negative (if allowed)
2
-5 10

// All same
3
5 5 5

// All different
3
1 2 3

// Sorted ascending
4
1 2 3 4

// Sorted descending
4
4 3 2 1

// Already optimal
3
3 2 1
\`\`\`

**3. Empty/Minimal Cases**
\`\`\`cpp
// Minimum possible input
1
1

// Empty array (if n can be 0)
0

// Single element
1
42
\`\`\`

**4. Extreme Cases**
\`\`\`cpp
// All elements maximum value
3
1000000000 1000000000 1000000000

// Mix of large and small
4
1 1000000000 1 1000000000
\`\`\`

**Level 3: Automated Test Generation**

**Generator Script: gen.py**
\`\`\`python
import random
import sys

def generate_test(n, max_val):
    print(n)
    for _ in range(n):
        print(random.randint(1, max_val), end=' ')
    print()

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    max_val = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    generate_test(n, max_val)
\`\`\`

**Usage:**
\`\`\`bash
python gen.py 10 100 > test1.txt      # 10 elements, max value 100
python gen.py 100000 1000000 > test2.txt  # Stress test
\`\`\`

**Level 4: Stress Testing**

**Compare two solutions:**

\`\`\`bash
#!/bin/bash
# stress.sh

# Compile both solutions
g++ solution.cpp -o solution  # Your optimized solution
g++ brute.cpp -o brute        # Simple brute force

for i in {1..1000}; do
    # Generate random test
    python gen.py 10 100 > input.txt
    
    # Run both
    ./solution < input.txt > output1.txt
    ./brute < input.txt > output2.txt
    
    # Compare
    if ! diff output1.txt output2.txt > /dev/null; then
        echo "Mismatch found on test $i!"
        echo "Input:"
        cat input.txt
        echo "Your output:"
        cat output1.txt
        echo "Expected output:"
        cat output2.txt
        exit 1
    fi
    
    echo "Test $i passed"
done

echo "All tests passed!"
\`\`\`

**Level 5: Targeted Test Cases**

**For specific bug types:**

**Testing for Overflow:**
\`\`\`cpp
// Generate large values
2
1000000000 1000000000
// Product will overflow int!
\`\`\`

**Testing for TLE:**
\`\`\`bash
# Generate maximum input
python gen.py 100000 1000000 > big_test.txt
time ./solution < big_test.txt
# Should complete in < 2 seconds
\`\`\`

**Testing for Corner Cases:**
\`\`\`cpp
// All elements equal
5
7 7 7 7 7

// Alternating pattern
6
1 2 1 2 1 2

// Palindrome
5
1 2 3 2 1
\`\`\`

**Advanced: Custom Validators**

**Validator to check output format:**
\`\`\`python
# validate.py
import sys

def validate_output(output):
    lines = output.strip().split('\\n')
    
    # Check: Single integer on one line
    if len(lines) != 1:
        return False, "Expected 1 line, got " + str(len(lines))
    
    # Check: Valid integer
    try:
        value = int(lines[0])
    except:
        return False, "Not a valid integer"
    
    # Check: Non-negative
    if value < 0:
        return False, "Output is negative"
    
    return True, "Valid"

if __name__ == "__main__":
    output = sys.stdin.read()
    valid, message = validate_output(output)
    if valid:
        print("✓ Valid:", message)
    else:
        print("✗ Invalid:", message)
        sys.exit(1)
\`\`\`

**Complete Testing System:**

\`\`\`bash
#!/bin/bash
# test_all.sh

echo "Running all tests..."

# Test 1: Samples
echo "1. Sample tests..."
./solution < sample1.txt | diff - sample1_output.txt || exit 1
./solution < sample2.txt | diff - sample2_output.txt || exit 1
echo "✓ Samples passed"

# Test 2: Edge cases
echo "2. Edge cases..."
for test in edge_*.txt; do
    output=$(echo $test | sed 's/\\.txt/_output.txt/')
    ./solution < $test | diff - $output || exit 1
done
echo "✓ Edge cases passed"

# Test 3: Stress test
echo "3. Stress testing..."
./stress.sh || exit 1
echo "✓ Stress tests passed"

# Test 4: Performance
echo "4. Performance test..."
python gen.py 100000 1000000 > perf_test.txt
timeout 2s ./solution < perf_test.txt > /dev/null
if [ $? -eq 124 ]; then
    echo "✗ TLE on large input!"
    exit 1
fi
echo "✓ Performance OK"

echo "All tests passed! Ready to submit."
\`\`\`

**Test Case Best Practices:**

**1. Start Small**
\`\`\`
Don't test n=100000 first!
Start with n=1, n=2, n=3
Gradually increase
\`\`\`

**2. One Bug at a Time**
\`\`\`
If multiple tests fail
Fix simplest case first
Retest all
\`\`\`

**3. Keep Failing Tests**
\`\`\`
When you find bug
Save that test case
Add to regression suite
\`\`\`

**4. Name Tests Descriptively**
\`\`\`
edge_minimum.txt
edge_maximum.txt  
edge_all_equal.txt
edge_sorted.txt
random_medium_001.txt
\`\`\`

**Common Testing Mistakes:**

**Mistake 1: Only testing samples**
- Samples are often too simple
- Hidden tests have edge cases

**Mistake 2: Not testing boundaries**
- n = 1 and n = max often break solutions

**Mistake 3: Not testing performance**
- Solution might TLE on large input

**Mistake 4: Ignoring overflow**
- Test with large numbers!

**Quick Test Generation Tricks:**

**C++ Generator:**
\`\`\`cpp
// gen.cpp
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    int n = argc > 1 ? atoi(argv[1]) : 10;
    int max_val = argc > 2 ? atoi(argv[2]) : 100;
    
    cout << n << endl;
    for(int i = 0; i < n; i++) {
        cout << (rand() % max_val) + 1 << " ";
    }
    cout << endl;
    
    return 0;
}
\`\`\`

**Compile once, use forever:**
\`\`\`bash
g++ gen.cpp -o gen
./gen 10 100 > test1.txt
./gen 1000 1000000 > test2.txt
\`\`\`

**Testing Checklist:**

Before submitting:
\`\`\`
✓ All samples pass
✓ Tested n = 1 (minimum)
✓ Tested n = max (maximum)
✓ Tested all same values
✓ Tested large numbers (overflow)
✓ Tested sorted input
✓ Tested reverse sorted input
✓ Run stress test (if time)
✓ Performance test on max input
\`\`\`

**Time Budget:**

For 20-minute problem:
- Writing solution: 12 minutes
- Testing samples: 2 minutes
- Edge cases: 3 minutes
- Stress test (if time): 2 minutes
- Submit: 1 minute

**Real Example:**

Problem: Find maximum subarray sum

**My test suite:**
\`\`\`bash
# Sample
3
-1 2 3
# Output: 5

# Edge: Single element
1
5
# Output: 5

# Edge: All negative
3
-1 -2 -3
# Output: -1

# Edge: All positive
4
1 2 3 4
# Output: 10

# Edge: Alternating
6
-1 2 -1 2 -1 2
# Output: 4

# Stress: Random 1000 elements
# (generated)

# Performance: 100000 elements
# (generated)
\`\`\`

**If all pass → confident submission!**

**Bottom Line:**

Effective testing:
- ✅ Always test samples first
- ✅ Create edge cases for boundaries
- ✅ Use generators for stress tests
- ✅ Compare with brute force when possible
- ✅ Test performance on large inputs
- ✅ Keep tests for future problems

**Good tests = fewer wrong submissions = better rank!**`,
    },
    {
      question:
        'Many debugging techniques involve trade-offs between debug information and code cleanliness. Discuss how to structure debug code so it can be easily enabled/disabled without cluttering the solution.',
      answer: `Clean debug infrastructure is key to fast, maintainable competitive programming code. Here's the complete approach:

**The Problem:**

Messy debugging:
\`\`\`cpp
void solve() {
    int n; cin >> n;
    // cout << "n = " << n << endl;  // Commented out
    vector<int> arr(n);
    for(int i = 0; i < n; i++) {
        cin >> arr[i];
        // cout << "arr[" << i << "] = " << arr[i] << endl;  // Commented
    }
    
    int result = process(arr);
    cout << result << endl;
    // cerr << "Debug: result = " << result << endl;  // Another comment
}
\`\`\`

**Problems:**
- ❌ Commented code is ugly
- ❌ Easy to forget to uncomment/comment
- ❌ Risk of submitting with debug output
- ❌ Hard to enable/disable selectively

**Solution: Preprocessor-Based System**

**Level 1: Basic LOCAL Flag**

\`\`\`cpp
#ifdef LOCAL
#define debug(x) cerr << #x << " = " << x << endl
#else
#define debug(x)
#endif

void solve() {
    int n; cin >> n;
    debug(n);  // Clean! No comments
    
    vector<int> arr(n);
    for(int i = 0; i < n; i++) {
        cin >> arr[i];
        debug(arr[i]);
    }
    
    int result = process(arr);
    debug(result);
    cout << result << endl;
}
\`\`\`

**Compile:**
\`\`\`bash
g++ -DLOCAL solution.cpp  # Debug enabled
g++ solution.cpp          # Debug disabled (for submit)
\`\`\`

**Level 2: Multi-Variable Debug**

\`\`\`cpp
#ifdef LOCAL
#define debug(x) cerr << #x << " = " << x << endl
#define debug2(x, y) cerr << #x << " = " << x << ", " << #y << " = " << y << endl
#define debug3(x, y, z) cerr << #x << " = " << x << ", " << #y << " = " << y << ", " << #z << " = " << z << endl
#define debug4(x, y, z, w) cerr << #x << " = " << x << ", " << #y << " = " << y << ", " << #z << " = " << z << ", " << #w << " = " << w << endl
#else
#define debug(x)
#define debug2(x, y)
#define debug3(x, y, z)
#define debug4(x, y, z, w)
#endif

// Usage:
debug2(i, sum);      // i = 5, sum = 15
debug3(a, b, c);     // a = 1, b = 2, c = 3
debug4(x, y, z, w);  // x = 1, y = 2, z = 3, w = 4
\`\`\`

**Level 3: Container Debug**

\`\`\`cpp
#ifdef LOCAL
template<typename T>
void _debug(const char* name, const vector<T>& v) {
    cerr << name << " = [";
    for(size_t i = 0; i < v.size(); i++) {
        if(i > 0) cerr << ", ";
        cerr << v[i];
    }
    cerr << "]" << endl;
}

template<typename T>
void _debug(const char* name, const set<T>& s) {
    cerr << name << " = {";
    bool first = true;
    for(const auto& x : s) {
        if(!first) cerr << ", ";
        cerr << x;
        first = false;
    }
    cerr << "}" << endl;
}

template<typename K, typename V>
void _debug(const char* name, const map<K, V>& m) {
    cerr << name << " = {";
    bool first = true;
    for(const auto& [k, v] : m) {
        if(!first) cerr << ", ";
        cerr << k << ": " << v;
        first = false;
    }
    cerr << "}" << endl;
}

#define debugv(v) _debug(#v, v)
#else
#define debugv(v)
#endif

// Usage:
vector<int> arr = {1, 2, 3};
debugv(arr);  // arr = [1, 2, 3]

set<int> s = {3, 1, 4};
debugv(s);  // s = {1, 3, 4}

map<int, string> m = {{1, "a"}, {2, "b"}};
debugv(m);  // m = {1: a, 2: b}
\`\`\`

**Level 4: Advanced Debug Macros**

\`\`\`cpp
#ifdef LOCAL
// Trace function entry/exit
#define TRACE_FUNC cerr << "→ " << __FUNCTION__ << endl

// Separator for visual clarity
#define SEP cerr << "==================" << endl

// Conditional debug
#define debug_if(cond, x) if(cond) cerr << #x << " = " << x << endl

// Line number
#define debug_line cerr << "Line " << __LINE__ << endl

// Multiple values at once
template<typename T>
void _print(const T& t) { cerr << t; }

template<typename T, typename... Args>
void _print(const T& first, const Args&... args) {
    cerr << first << " ";
    _print(args...);
}

#define print(...) cerr << "[" << #__VA_ARGS__ << "] = "; _print(__VA_ARGS__); cerr << endl

#else
#define TRACE_FUNC
#define SEP
#define debug_if(cond, x)
#define debug_line
#define print(...)
#endif

// Usage:
void solve() {
    TRACE_FUNC;  // → solve
    
    int x = 5, y = 10, z = 15;
    print(x, y, z);  // [x, y, z] = 5 10 15
    
    SEP;  // ==================
    
    debug_if(x > 3, x);  // x = 5 (only if x > 3)
    
    debug_line;  // Line 42
}
\`\`\`

**Level 5: Complete Debug System**

**Put this in your template:**
\`\`\`cpp
#ifdef LOCAL
// Basic debug
#define debug(x) cerr << #x << " = " << x << endl
#define debug2(x, y) cerr << #x << " = " << x << ", " << #y << " = " << y << endl

// Container debug
template<typename T>
ostream& operator<<(ostream& os, const vector<T>& v) {
    os << "[";
    for(size_t i = 0; i < v.size(); i++) {
        if(i > 0) os << ", ";
        os << v[i];
    }
    return os << "]";
}

template<typename T>
ostream& operator<<(ostream& os, const set<T>& s) {
    os << "{";
    bool first = true;
    for(const auto& x : s) {
        if(!first) os << ", ";
        os << x;
        first = false;
    }
    return os << "}";
}

template<typename T1, typename T2>
ostream& operator<<(ostream& os, const pair<T1, T2>& p) {
    return os << "(" << p.first << ", " << p.second << ")";
}

// Advanced
#define TRACE cerr << "→ " << __FUNCTION__ << ":" << __LINE__ << endl
#define SEP cerr << "==================" << endl

#else
#define debug(x)
#define debug2(x, y)
#define TRACE
#define SEP
#endif
\`\`\`

**Now you can:**
\`\`\`cpp
vector<int> arr = {1, 2, 3};
debug(arr);  // arr = [1, 2, 3]

vector<pair<int, int>> vp = {{1, 2}, {3, 4}};
debug(vp);  // vp = [(1, 2), (3, 4)]

set<int> s = {3, 1, 4};
debug(s);  // s = {1, 3, 4}
\`\`\`

**Compile Command Aliases:**

\`\`\`bash
# Add to ~/.bashrc or ~/.zshrc

# Local testing (debug enabled)
alias cpl='g++ -std=c++17 -O2 -Wall -Wextra -DLOCAL'

# Submission (debug disabled)
alias cps='g++ -std=c++17 -O2 -Wall -Wextra'

# Compile and run
alias run='cpl solution.cpp && ./a.out < input.txt'
\`\`\`

**Usage:**
\`\`\`bash
run  # Compile with debug and run
cps solution.cpp  # Compile for submission (no debug)
\`\`\`

**Organizational Best Practices:**

**1. Debug Macros at Top**
\`\`\`cpp
// ============ DEBUG MACROS ============
#ifdef LOCAL
// ... all debug macros here ...
#endif
// ====================================

// ============ SOLUTION ============
void solve() {
    // Your code
}
// ==================================
\`\`\`

**2. Separate Debug Output**
\`\`\`cpp
// Use cerr for debug (goes to stderr)
cerr << "Debug info" << endl;

// Use cout for solution output (goes to stdout)
cout << result << endl;
\`\`\`

**Why?** Can redirect separately:
\`\`\`bash
./solution < input.txt > output.txt 2> debug.txt
# output.txt has solution output
# debug.txt has debug info
\`\`\`

**3. Visual Separators**
\`\`\`cpp
void solve() {
    SEP;
    cerr << "Test case starting" << endl;
    // ... code ...
    cerr << "Result: " << result << endl;
    SEP;
}
\`\`\`

Output:
\`\`\`
==================
Test case starting
... debug info ...
Result: 42
==================
\`\`\`

**Advanced: Selective Debug Levels**

\`\`\`cpp
#define DEBUG_LEVEL 2  // 0 = off, 1 = basic, 2 = verbose, 3 = trace

#if DEBUG_LEVEL >= 1
#define debug(x) cerr << #x << " = " << x << endl
#else
#define debug(x)
#endif

#if DEBUG_LEVEL >= 2
#define debug_verbose(x) cerr << "[V] " << #x << " = " << x << endl
#else
#define debug_verbose(x)
#endif

#if DEBUG_LEVEL >= 3
#define debug_trace(x) cerr << "[T] " << __LINE__ << ": " << #x << " = " << x << endl
#else
#define debug_trace(x)
#endif

// Usage:
debug(x);           // Level 1: Always shown
debug_verbose(arr); // Level 2: Only if verbose
debug_trace(i);     // Level 3: With line numbers
\`\`\`

**IDE Integration:**

**VS Code: settings.json**
\`\`\`json
{
    "code-runner.executorMap": {
        "cpp": "cd $dir && g++ -std=c++17 -O2 -DLOCAL $fileName -o $fileNameWithoutExt && $dir$fileNameWithoutExt < input.txt"
    }
}
\`\`\`

Now Ctrl+Alt+N compiles with debug enabled!

**Common Pitfalls:**

**Pitfall 1: Forgetting to disable**
✅ Solution: Use #ifdef, not comments

**Pitfall 2: Debug affects performance**
✅ Solution: Only debug small cases locally

**Pitfall 3: Debug output matches solution output**
✅ Solution: Use cerr, not cout

**Pitfall 4: Too much debug output**
✅ Solution: Use debug levels or conditional debug

**My Complete Template Header:**

\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

#ifdef LOCAL
#include "debug.h"  // Separate file with all debug macros
#else
#define debug(...)
#define debugv(...)
#define TRACE
#define SEP
#endif

// Your solution here...
\`\`\`

**debug.h file:**
\`\`\`cpp
// All debug infrastructure in separate file
// Clean separation!
\`\`\`

**Bottom Line:**

Clean debug infrastructure:
- ✅ Use #ifdef LOCAL, not comments
- ✅ Debug macros at top of file
- ✅ Use cerr for debug, cout for output
- ✅ Overload operator<< for containers
- ✅ Compile with -DLOCAL for testing
- ✅ Submit without -DLOCAL

**Result: Clean code + powerful debugging!**`,
    },
  ],
} as const;
