export const commonCompilationErrorsSection = {
    id: 'cp-m1-s12',
    title: 'Common Compilation Errors',
    content: `

# Common Compilation Errors

## Introduction

**Compilation errors** are the bane of every competitive programmer's existence. You've got a brilliant algorithm, you're racing against the clock, you hit compile... and BAM! A wall of cryptic error messages fills your screen.

The difference between a top competitor and a struggling one often isn't algorithmic knowledge—it's the ability to **diagnose and fix compilation errors in seconds**. While beginners can waste 10-15 minutes debugging a simple syntax error, experts recognize patterns instantly and fix them in under a minute.

In this comprehensive section, we'll explore the most common C++ compilation errors in competitive programming, learn to read compiler messages effectively, understand the root causes, and master rapid-fire fixes. We'll cover syntax errors, type errors, template errors, linker errors, and more.

**Goal**: Master rapid diagnosis and fixing of C++ compilation errors to save precious time in contests.

---

## Understanding Compiler Error Messages

### Anatomy of a Compiler Error

\`\`\`
solution.cpp:15:23: error: expected ';' before 'return'
   15 |     int x = 5
      |                       ^
      |                       ;
   16 |     return x;
      |     ~~~~~~
\`\`\`

**Parts of the error:**

1. **Filename**: \`solution.cpp\`
2. **Line number**: \`15\` (where error is detected)
3. **Column number**: \`23\` (character position)
4. **Error type**: \`error\` (vs \`warning\` or \`note\`)
5. **Message**: \`expected ';' before 'return'\`
6. **Context**: Shows code with \`^ \` pointing to error
7. **Suggestion**: Shows fix with \`; \`

### Reading Error Messages: The Golden Rule

**Read errors from TOP to BOTTOM:**

\`\`\`
solution.cpp:10: error: 'vector' was not declared in this scope
solution.cpp:10: note: suggested alternative: 'vectors'
solution.cpp:15: error: 'v' was not declared in this scope
solution.cpp:20: error: 'v' was not declared in this scope
solution.cpp:25: error: 'v' was not declared in this scope
\`\`\`

**The FIRST error** (line 10) is the root cause. All others are **cascading errors** caused by the first one.

**Fix the FIRST error, then recompile!**

---

## Category 1: Syntax Errors

These are the most common errors in competitive programming.

### Error 1: Missing Semicolon

**Error message:**
\`\`\`
error: expected ';' before 'return'
\`\`\`

**Bad code:**
\`\`\`cpp
int main() {
    int x = 5  // Missing semicolon
    return x;
}
\`\`\`

**Fix:**
\`\`\`cpp
int main() {
    int x = 5;  // Added semicolon
    return x;
}
\`\`\`

**Common places:**
- After variable declarations
- After function calls
- After return statements
- **NOT** after \`}\` closing braces (common misconception!)

### Error 2: Missing Brace/Parenthesis

**Error message:**
\`\`\`
error: expected '}' at end of input
error: expected ')' before ';' token
\`\`\`

**Bad code:**
\`\`\`cpp
int main() {
    if (x > 5) {
        cout << "yes" << endl;
    // Missing closing brace
    return 0;
}
\`\`\`

**Fix:**
\`\`\`cpp
int main() {
    if (x > 5) {
        cout << "yes" << endl;
    }  // Added closing brace
    return 0;
}
\`\`\`

**Pro tip:** Use an editor with **bracket matching** to highlight matching pairs!

### Error 3: Missing Template Arguments

**Error message:**
\`\`\`
error: missing template arguments before 'v'
\`\`\`

**Bad code:**
\`\`\`cpp
int main() {
    vector v;  // Missing template argument
}
\`\`\`

**Fix:**
\`\`\`cpp
int main() {
    vector<int> v;  // Added template argument
}
\`\`\`

**Common with:**
- \`vector\`, \`set\`, \`map\`, \`priority_queue\`
- Must specify type: \`vector<int>\`, \`set<string>\`, etc.

### Error 4: Assignment in Condition

**Error message:**
\`\`\`
warning: suggest parentheses around assignment used as truth value
\`\`\`

**Bad code:**
\`\`\`cpp
if (x = 5) {  // Assignment, not comparison!
    cout << "x is 5" << endl;
}
\`\`\`

**Fix:**
\`\`\`cpp
if (x == 5) {  // Comparison
    cout << "x is 5" << endl;
}
\`\`\`

**Note:** This is often a **warning**, but it's almost always a bug!

### Error 5: Incorrect Comparison Operators

**Error message:**
\`\`\`
error: invalid operands to binary expression
\`\`\`

**Bad code:**
\`\`\`cpp
if (5 < x < 10) {  // Doesn't work in C++!
    // ...
}
\`\`\`

**Why it fails:**
- Evaluates as \`(5 < x) < 10\`
- \`(5 < x)\` returns \`true\` (1) or \`false\` (0)
- Then compares \`1 < 10\` or \`0 < 10\` (always true!)

**Fix:**
\`\`\`cpp
if (5 < x && x < 10) {  // Correct
    // ...
}
\`\`\`

---

## Category 2: Type Errors

### Error 6: Type Mismatch

**Error message:**
\`\`\`
error: cannot convert 'string' to 'int' in assignment
\`\`\`

**Bad code:**
\`\`\`cpp
int x = "hello";  // Can't assign string to int
\`\`\`

**Fix:**
\`\`\`cpp
string x = "hello";  // Correct type
// Or:
int x = 42;  // If you want an int
\`\`\`

### Error 7: Integer Division Surprise

**Not a compilation error, but a logic bug:**

\`\`\`cpp
double average = (5 + 10) / 2;  // Result: 7 (not 7.5!)
// Because: int / int = int (truncated)
\`\`\`

**Fix:**
\`\`\`cpp
double average = (5 + 10) / 2.0;  // Result: 7.5
// Or:
double average = (5.0 + 10.0) / 2.0;
// Or:
double average = (double)(5 + 10) / 2;
\`\`\`

### Error 8: Overflow Not Detected

**Not a compilation error:**

\`\`\`cpp
int x = 1000000;
int y = x * x;  // Overflow! Wrong result
\`\`\`

**Fix:**
\`\`\`cpp
long long x = 1000000;
long long y = x * x;  // Correct
// Or:
long long y = (long long)x * x;
\`\`\`

### Error 9: Comparing Signed and Unsigned

**Error message:**
\`\`\`
warning: comparison of integer expressions of different signedness
\`\`\`

**Bad code:**
\`\`\`cpp
int i = -1;
vector<int> v = {1, 2, 3};
if (i < v.size()) {  // Problem: int vs size_t (unsigned)
    // ...
}
\`\`\`

**Why it's dangerous:**
- \`v.size()\` returns \`size_t\` (unsigned)
- \`- 1\` converted to unsigned becomes a huge number!
- Comparison may be wrong

**Fix:**
\`\`\`cpp
if (i >= 0 && i < (int)v.size()) {  // Cast to int
    // ...
}
// Or:
if (i >= 0 && (size_t)i < v.size()) {  // Cast i to size_t
    // ...
}
\`\`\`

---

## Category 3: Undeclared Identifiers

### Error 10: Variable Not Declared

**Error message:**
\`\`\`
error: 'x' was not declared in this scope
\`\`\`

**Bad code:**
\`\`\`cpp
int main() {
    cout << x << endl;  // x never declared
}
\`\`\`

**Fix:**
\`\`\`cpp
int main() {
    int x = 5;  // Declare x first
    cout << x << endl;
}
\`\`\`

**Common causes:**
- Typo in variable name
- Variable declared in different scope
- Forgot to declare

### Error 11: Function Not Declared

**Error message:**
\`\`\`
error: 'solve' was not declared in this scope
\`\`\`

**Bad code:**
\`\`\`cpp
int main() {
    solve();  // solve() defined later
}

void solve() {
    // ...
}
\`\`\`

**Fix Option 1: Forward declaration**
\`\`\`cpp
void solve();  // Forward declaration

int main() {
    solve();  // OK now
}

void solve() {
    // ...
}
\`\`\`

**Fix Option 2: Define before use**
\`\`\`cpp
void solve() {
    // ...
}

int main() {
    solve();  // OK now
}
\`\`\`

**In CP:** Usually put helper functions before \`main()\`.

### Error 12: Missing Header

**Error message:**
\`\`\`
error: 'vector' is not a member of 'std'
error: 'cout' was not declared in this scope
\`\`\`

**Bad code:**
\`\`\`cpp
// Missing #include

int main() {
    vector<int> v;  // Error!
    cout << "hello" << endl;  // Error!
}
\`\`\`

**Fix:**
\`\`\`cpp
#include <bits/stdc++.h>  // Include everything (CP style)
using namespace std;

int main() {
    vector<int> v;  // OK
    cout << "hello" << endl;  // OK
}
\`\`\`

**Note:** \`#include < bits / stdc++.h > \` is a g++ extension that includes all standard headers. It's perfect for CP but shouldn't be used in production code!

---

## Category 4: Array and Vector Errors

### Error 13: Array Index Out of Bounds

**NO compilation error** (runtime error):

\`\`\`cpp
int arr[10];
arr[15] = 5;  // Out of bounds! Undefined behavior
\`\`\`

**Detection:**
- Won't get compilation error
- May crash at runtime
- May give wrong answer (WA)
- May work locally but fail on judge

**Fix:**
\`\`\`cpp
int arr[10];
// Only access arr[0] to arr[9]
if (index >= 0 && index < 10) {
    arr[index] = 5;  // Safe
}
\`\`\`

**Pro tip:** Use \`vector\` with \`.at()\` for bounds checking in local testing:
\`\`\`cpp
vector<int> v(10);
v.at(15) = 5;  // Throws exception (easier to debug)
\`\`\`

### Error 14: Array Size Not Constant

**Error message:**
\`\`\`
error: array bound is not an integer constant before ']' token
\`\`\`

**Bad code:**
\`\`\`cpp
int n;
cin >> n;
int arr[n];  // Variable-length array (not standard C++)
\`\`\`

**Fix:**
\`\`\`cpp
int n;
cin >> n;
vector<int> arr(n);  // Use vector for dynamic size
\`\`\`

**Or use global array:**
\`\`\`cpp
const int MAXN = 100000;
int arr[MAXN];

int main() {
    int n;
    cin >> n;
    // Use arr[0] to arr[n-1]
}
\`\`\`

### Error 15: Vector Initialization Error

**Error message:**
\`\`\`
error: too many initializers for 'std::vector<int>'
\`\`\`

**Bad code:**
\`\`\`cpp
vector<int> v(10, 0, 0);  // Wrong! Too many arguments
\`\`\`

**Correct usage:**
\`\`\`cpp
vector<int> v1;              // Empty vector
vector<int> v2(10);          // 10 elements, default value (0)
vector<int> v3(10, 5);       // 10 elements, all 5
vector<int> v4 = {1, 2, 3};  // Initialize with list
vector<int> v5{1, 2, 3};     // Same as above (C++11)
\`\`\`

### Error 16: 2D Vector Confusion

**Error message:**
\`\`\`
error: invalid types 'int[int]' for array subscript
\`\`\`

**Bad code:**
\`\`\`cpp
vector<int> grid(10, 10);  // WRONG! Creates vector of 10 elements, all value 10
grid[0][0] = 1;  // Error! grid[0] is int, not vector
\`\`\`

**Fix:**
\`\`\`cpp
// Correct 2D vector initialization
vector<vector<int>> grid(10, vector<int>(10, 0));
// 10 rows, each with 10 columns, all 0

grid[0][0] = 1;  // OK now
\`\`\`

---

## Category 5: Template Errors

### Error 17: Template Type Deduction Failure

**Error message:**
\`\`\`
error: no matching function for call to 'max(int, double)'
\`\`\`

**Bad code:**
\`\`\`cpp
cout << max(5, 3.14) << endl;  // Different types!
\`\`\`

**Fix:**
\`\`\`cpp
cout << max(5.0, 3.14) << endl;  // Both double
// Or:
cout << max<double>(5, 3.14) << endl;  // Explicit type
\`\`\`

### Error 18: Priority Queue Template Error

**Error message:**
\`\`\`
error: template argument 2 is invalid
\`\`\`

**Bad code:**
\`\`\`cpp
priority_queue<int, greater<int>> pq;  // Missing middle argument!
\`\`\`

**Fix:**
\`\`\`cpp
priority_queue<int, vector<int>, greater<int>> pq;  // Correct
// Type, Container, Comparator
\`\`\`

**Common patterns:**
\`\`\`cpp
// Max heap (default)
priority_queue<int> pq1;

// Min heap
priority_queue<int, vector<int>, greater<int>> pq2;

// Custom comparator
priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq3;
\`\`\`

---

## Category 6: Iterator Errors

### Error 19: Iterator Invalidation

**Runtime error (no compilation error):**

\`\`\`cpp
vector<int> v = {1, 2, 3, 4, 5};
for (auto it = v.begin(); it != v.end(); it++) {
    if (*it == 3) {
        v.erase(it);  // Invalidates iterator!
        // Using it after this is undefined behavior
    }
}
\`\`\`

**Fix:**
\`\`\`cpp
vector<int> v = {1, 2, 3, 4, 5};
for (auto it = v.begin(); it != v.end(); ) {
    if (*it == 3) {
        it = v.erase(it);  // erase returns next valid iterator
    } else {
        ++it;
    }
}
\`\`\`

**Or use remove-erase idiom:**
\`\`\`cpp
v.erase(remove(v.begin(), v.end(), 3), v.end());
\`\`\`

### Error 20: Dereferencing Invalid Iterator

**Error message:**
\`\`\`
error: no match for 'operator*' (operand type is 'std::vector<int>')
\`\`\`

**Bad code:**
\`\`\`cpp
vector<int> v = {1, 2, 3};
auto it = find(v.begin(), v.end(), 5);
cout << *it << endl;  // BUG! Element not found, it == v.end()
\`\`\`

**Fix:**
\`\`\`cpp
vector<int> v = {1, 2, 3};
auto it = find(v.begin(), v.end(), 5);
if (it != v.end()) {  // Check before dereferencing
    cout << *it << endl;
} else {
    cout << "Not found" << endl;
}
\`\`\`

---

## Category 7: String Errors

### Error 21: Character vs String Literal

**Error message:**
\`\`\`
error: conversion from 'const char*' to 'char' loses precision
\`\`\`

**Bad code:**
\`\`\`cpp
char c = "a";  // WRONG! "a" is string, not char
\`\`\`

**Fix:**
\`\`\`cpp
char c = 'a';  // Single quotes for char
string s = "a";  // Double quotes for string
\`\`\`

**Remember:**
- \`'a'\` = character (single quotes)
- \`"a"\` = string literal (double quotes)

### Error 22: String Comparison

**Not an error, but a common mistake:**

\`\`\`cpp
string s = "hello";
if (s == "hello") {  // OK in C++
    // ...
}

// But in C:
char s[] = "hello";
if (s == "hello") {  // WRONG! Compares pointers, not content
    // ...
}
// Should use strcmp(s, "hello") == 0
\`\`\`

**In C++:** String comparison with \`== \` works correctly!

### Error 23: String Concatenation Type Error

**Error message:**
\`\`\`
error: invalid operands of types 'const char [6]' and 'const char [7]' to binary 'operator+'
\`\`\`

**Bad code:**
\`\`\`cpp
string s = "hello" + "world";  // Can't add two C-strings directly
\`\`\`

**Fix:**
\`\`\`cpp
string s = string("hello") + "world";  // Convert first one
// Or:
string s = "hello";
s += "world";
\`\`\`

---

## Category 8: STL Container Errors

### Error 24: Using [] on Empty Map

**Runtime issue (no compilation error):**

\`\`\`cpp
map<string, int> m;
cout << m["key"] << endl;  // Creates entry if doesn't exist!
\`\`\`

**Problem:** Using \`[]\` on a map creates the key if it doesn't exist (with default value 0 for int).

**Better:**
\`\`\`cpp
map<string, int> m;
if (m.count("key")) {
    cout << m["key"] << endl;
} else {
    cout << "Key not found" << endl;
}
\`\`\`

### Error 25: Set/Map Element Modification

**Error message:**
\`\`\`
error: assignment of read-only reference
\`\`\`

**Bad code:**
\`\`\`cpp
set<int> s = {1, 2, 3};
for (auto& x : s) {
    x = x * 2;  // ERROR! Can't modify set elements directly
}
\`\`\`

**Why:** Sets/maps maintain sorted order. Modifying elements would break invariants.

**Fix:**
\`\`\`cpp
set<int> s = {1, 2, 3};
set<int> newSet;
for (auto x : s) {
    newSet.insert(x * 2);
}
s = newSet;
\`\`\`

---

## Category 9: Macro Errors

### Error 26: Macro Side Effects

**Not a compilation error, but dangerous:**

\`\`\`cpp
#define SQUARE(x) x * x

int a = 5;
cout << SQUARE(a + 1) << endl;  // Expects 36, gets 11!
// Expands to: a + 1 * a + 1 = 5 + 1 * 5 + 1 = 11
\`\`\`

**Fix:**
\`\`\`cpp
#define SQUARE(x) ((x) * (x))  // Parentheses!

cout << SQUARE(a + 1) << endl;  // Now correctly: 36
\`\`\`

### Error 27: Macro Multiple Evaluation

\`\`\`cpp
#define MAX(a, b) ((a) > (b) ? (a) : (b))

int x = 5, y = 10;
cout << MAX(x++, y++) << endl;  // BUG! x or y incremented twice
\`\`\`

**Better:** Use \`max()\` function instead of macro!

---

## Category 10: Linker Errors

### Error 28: Undefined Reference

**Error message:**
\`\`\`
undefined reference to \`solve()\`
\`\`\`

**Cause:** Function declared but not defined.

**Bad code:**
\`\`\`cpp
void solve();  // Declaration

int main() {
    solve();  // Error! No definition
}
// Missing definition of solve()
\`\`\`

**Fix:**
\`\`\`cpp
void solve();  // Declaration

int main() {
    solve();
}

void solve() {  // Definition
    // Implementation here
}
\`\`\`

### Error 29: Multiple Definition

**Error message:**
\`\`\`
multiple definition of \`function\`
\`\`\`

**Cause:** Function defined in header and included multiple times.

**Fix:** Use \`inline\` or define in .cpp file only.

**In CP:** Usually not an issue (single file).

---

## Quick Reference: Error Message Patterns

| Error Message | Meaning | Quick Fix |
|---------------|---------|-----------|
| \`expected ';\` | Missing semicolon | Add \`; \` |
| \`expected '}\` | Missing closing brace | Add \`}\` |
| \`was not declared\` | Undeclared variable/function | Declare it first |
| \`missing template arguments\` | Forgot template type | Add \`<type>\` |
| \`cannot convert\` | Type mismatch | Cast or fix type |
| \`invalid operands\` | Operation not supported | Check types |
| \`no matching function\` | Wrong function arguments | Check parameters |
| \`undefined reference\` | Function not defined | Add definition |
| \`array bound is not constant\` | Variable array size | Use vector |

---

## Debugging Strategy

### Step-by-Step Debugging Process

**When you get compilation errors:**

1. **Read FIRST error only** (ignore rest initially)
2. **Check line number** (error might be on previous line!)
3. **Look for patterns** (semicolon, brace, type, etc.)
4. **Fix the first error**
5. **Recompile**
6. **Repeat**

**Example:**
\`\`\`
error: 'vector' was not declared in this scope
error: 'v' was not declared in this scope
error: 'v' was not declared in this scope
...100 more errors...
\`\`\`

**Don't panic!** Fix the FIRST error (missing header), and all others will disappear.

---

## Tools to Prevent Errors

### 1. IDE/Editor Features

**Use an IDE with:**
- Syntax highlighting
- Bracket matching
- Auto-completion
- Error highlighting
- Auto-formatting

**Recommended:**
- VS Code with C++ extension
- CLion
- CodeBlocks
- Sublime Text with plugins

### 2. Compiler Flags

\`\`\`bash
g++ -std=c++17 -Wall -Wextra -O2 solution.cpp -o solution
\`\`\`

**Flags:**
- \`- std=c++17\`: Use C++17 standard
- \`- Wall\`: Enable all warnings
- \`- Wextra\`: Enable extra warnings
- \`- O2\`: Optimization level 2

### 3. Static Analysis

\`\`\`bash
# Clang-tidy
clang-tidy solution.cpp -- -std=c++17

# Cppcheck
cppcheck solution.cpp
\`\`\`

---

## Summary

**Most Common Errors:**

1. Missing semicolon (\`; \`\)
2. Missing/extra braces (\`{ } \`\)
3. Missing template arguments (\`vector\` → \`vector<int>\`)
4. Undeclared variables/functions
5. Type mismatches
6. Array/vector access issues
7. Template type deduction failures

**Golden Rules:**

✅ **Read FIRST error only**
✅ **Check line BEFORE error too**
✅ **Look for patterns**
✅ **Fix one error at a time**
✅ **Recompile after each fix**
✅ **Use proper editor/IDE**
✅ **Enable compiler warnings**

**Time-Saving Tips:**

- Use \`#include < bits / stdc++.h > \` (CP only!)
- Use \`using namespace std; \`
- Use proper IDE with error highlighting
- Learn to recognize error patterns
- Build muscle memory for common fixes

---

## Next Steps

Now let's learn about **Debugging in Competitive Environment** - how to debug your code efficiently when it compiles but gives wrong answers!

**Key Takeaway**: Compilation errors are inevitable, but they shouldn't slow you down. Master pattern recognition, read compiler messages effectively, and fix errors systematically. Most errors have simple, instant fixes once you recognize the pattern. Build a mental database of error-pattern-fix triplets through practice!
`,
    quizId: 'cp-m1-s12-quiz',
        discussionId: 'cp-m1-s12-discussion',
} as const ;
