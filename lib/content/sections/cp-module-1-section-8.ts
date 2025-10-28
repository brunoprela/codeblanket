export const macrosPreprocessorTricksSection = {
  id: 'cp-m1-s8',
  title: 'Macros & Preprocessor Tricks',
  content: `

# Macros & Preprocessor Tricks

## Introduction

**Macros** are one of the most controversial topics in competitive programming. Love them or hate them, they're used extensively by top coders to save time during contests. A well-designed macro library can save you 30+ seconds per problem—which adds up to 3-5 minutes in a 5-problem contest!

In this section, we'll explore useful macros, when to use them, potential pitfalls, and how to build your personal macro library.

**Warning**: Macros can be dangerous if misused. We'll show you both the power and the perils.

---

## What Are Macros?

**Macros** are text replacements performed by the preprocessor before compilation. They're defined using \`#define\`.

### Basic Macro

\`\`\`cpp
#define PI 3.14159265358979323846

double area = PI * r * r;
// Preprocessor replaces PI with 3.14159...
\`\`\`

### Function-Like Macro

\`\`\`cpp
#define SQUARE(x) ((x) * (x))

int result = SQUARE(5);  // Becomes: ((5) * (5))
\`\`\`

**Note the parentheses!** They're crucial for preventing bugs.

---

## Essential CP Macros

Let's start with the most useful macros that appear in almost every competitive programmer's template.

### 1. Loop Macros

\`\`\`cpp
#define FOR(i, a, b) for (int i = (a); i < (b); i++)
#define RFOR(i, a, b) for (int i = (a); i > (b); i--)
#define REP(i, n) for (int i = 0; i < (n); i++)
#define RREP(i, n) for (int i = (n) - 1; i >= 0; i--)

// Usage:
REP(i, n) {
    cout << i << " ";
}

FOR(i, 1, n + 1) {
    // i goes from 1 to n
}
\`\`\`

**Saves:** ~20 characters per loop, 5-10 seconds of typing

### 2. Container Iteration

\`\`\`cpp
#define ALL(v) (v).begin(), (v).end()
#define RALL(v) (v).rbegin(), (v).rend()
#define SIZE(v) (int)(v).size()

// Usage:
vector<int> v = {5, 2, 8, 1, 9};

sort(ALL(v));              // Instead of: sort(v.begin(), v.end())
reverse(RALL(v));          // Instead of: reverse(v.rbegin(), v.rend())
int n = SIZE(v);           // Instead of: int n = (int)v.size()
\`\`\`

**Why cast size to int?** \`v.size()\` returns \`size_t\` (unsigned), which can cause bugs when subtracting.

### 3. Common Operations

\`\`\`cpp
#define PB push_back
#define MP make_pair
#define F first
#define S second

// Usage:
vector<int> v;
v.PB(5);
v.PB(10);

pair<int, int> p = MP(3, 4);
cout << p.F << " " << p.S << endl;
\`\`\```

### 4. Min/Max for Multiple Values

\`\`\`cpp
#define MAX3(a, b, c) max((a), max((b), (c)))
#define MIN3(a, b, c) min((a), min((b), (c)))
#define MAX4(a, b, c, d) max(max((a), (b)), max((c), (d)))

// Usage:
int maximum = MAX3(5, 10, 3);  // 10
\`\`\`

### 5. Debug Macro

\`\`\`cpp
#ifdef LOCAL
#define DEBUG(x) cerr << #x << " = " << (x) << endl
#define DEBUGV(v) cerr << #v << " = "; for (auto x : v) cerr << x << " "; cerr << endl
#else
#define DEBUG(x)
#define DEBUGV(v)
#endif

// Usage:
int x = 42;
DEBUG(x);  // Prints: x = 42 (only if LOCAL defined)

vector<int> v = {1, 2, 3};
DEBUGV(v);  // Prints: v = 1 2 3
\`\`\`

**How to use:**
\`\`\`bash
# Compile for debugging
g++ -DLOCAL solution.cpp -o solution

# Compile for submission (no debug output)
g++ solution.cpp -o solution
\`\`\`

---

## Type Alias Macros

### Common Type Aliases

\`\`\`cpp
typedef long long ll;
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;
typedef vector<int> vi;
typedef vector<ll> vll;
typedef vector<pii> vpii;
typedef vector<string> vs;

// Modern equivalent (using):
using ll = long long;
using pii = pair<int, int>;
using vi = vector<int>;
\`\`\`

### Shorter Names

\`\`\`cpp
#define ll long long
#define ull unsigned long long
#define ld long double
#define pii pair<int, int>
#define pll pair<long long, long long>
#define vi vector<int>
#define vll vector<long long>

// Usage:
ll n = 1000000000000LL;
vi v(n);
pii p = {3, 4};
\`\`\`

---

## Advanced Macro Techniques

### Variadic Macros (C++11)

\`\`\`cpp
// Debug multiple variables at once
#define DBG(...) cerr << "[" << #__VA_ARGS__ << "]: "; debug_out(__VA_ARGS__)

template<typename T>
void debug_out(T t) {
    cerr << t << endl;
}

template<typename T, typename... Args>
void debug_out(T t, Args... args) {
    cerr << t << ", ";
    debug_out(args...);
}

// Usage:
int a = 5, b = 10, c = 15;
DBG(a, b, c);  // [a, b, c]: 5, 10, 15
\`\`\`

### Macro for Multiple Test Cases

\`\`\`cpp
#define MULTI_TEST true

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int t = 1;
    if (MULTI_TEST) cin >> t;
    
    while (t--) {
        solve();
    }
    
    return 0;
}
\`\`\`

### Grid Direction Macros

\`\`\`cpp
#define DIR4 {{0,1}, {1,0}, {0,-1}, {-1,0}}
#define DIR8 {{0,1}, {1,1}, {1,0}, {1,-1}, {0,-1}, {-1,-1}, {-1,0}, {-1,1}}

// Usage:
int dx[] = {0, 1, 0, -1};
int dy[] = {1, 0, -1, 0};

// Or:
vector<pii> directions = DIR4;
for (auto [dx, dy] : directions) {
    int nx = x + dx;
    int ny = y + dy;
}
\`\`\`

---

## Dangerous Macro Pitfalls

### Pitfall 1: Missing Parentheses

❌ **WRONG:**
\`\`\`cpp
#define SQUARE(x) x * x

int result = SQUARE(3 + 2);
// Expands to: 3 + 2 * 3 + 2 = 3 + 6 + 2 = 11 (WRONG!)
\`\`\`

✅ **CORRECT:**
\`\`\`cpp
#define SQUARE(x) ((x) * (x))

int result = SQUARE(3 + 2);
// Expands to: ((3 + 2) * (3 + 2)) = 25 (CORRECT!)
\`\`\`

**Rule:** Always use parentheses around parameters and the entire expression!

### Pitfall 2: Side Effects

❌ **WRONG:**
\`\`\`cpp
#define MAX(a, b) ((a) > (b) ? (a) : (b))

int x = 5;
int result = MAX(x++, 10);
// Expands to: ((x++) > (10) ? (x++) : (10))
// x gets incremented TWICE if x++ > 10!
\`\`\`

✅ **CORRECT:** Use \`std::max\` for values with side effects
\`\`\`cpp
int result = max(x++, 10);  // x incremented exactly once
\`\`\`

### Pitfall 3: No Type Checking

\`\`\`cpp
#define SQUARE(x) ((x) * (x))

string s = "hello";
SQUARE(s);  // Compiles! Tries to multiply strings (error)
\`\`\`

**Macros don't check types!** Compiler errors can be cryptic.

### Pitfall 4: Debugging Nightmare

\`\`\`cpp
#define FOR(i, n) for (int i = 0; i < n; i++)

FOR(i, 10) {
    FOR(i, 5) {  // BUG! Nested loop redefines i
        cout << i << " ";
    }
}
\`\`\`

**Macros are text replacement—watch out for variable name collisions!**

---

## Building Your Macro Library

Here's a comprehensive macro library that balances usefulness and safety:

\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

// Type aliases
using ll = long long;
using ull = unsigned long long;
using ld = long double;
using pii = pair<int, int>;
using pll = pair<ll, ll>;
using vi = vector<int>;
using vll = vector<ll>;
using vpii = vector<pii>;

// Constants
const int MOD = 1e9 + 7;
const int INF = 1e9;
const ll LINF = 1e18;
const ld EPS = 1e-9;

// Loops
#define FOR(i, a, b) for (int i = (a); i < (b); i++)
#define RFOR(i, a, b) for (int i = (a); i > (b); i--)
#define REP(i, n) for (int i = 0; i < (n); i++)
#define RREP(i, n) for (int i = (n) - 1; i >= 0; i--)

// Container shortcuts
#define ALL(v) (v).begin(), (v).end()
#define RALL(v) (v).rbegin(), (v).rend()
#define SIZE(v) (int)(v).size()
#define PB push_back
#define MP make_pair
#define F first
#define S second

// Min/Max
#define MAX3(a, b, c) max({a, b, c})
#define MIN3(a, b, c) min({a, b, c})

// Debug (only active when LOCAL is defined)
#ifdef LOCAL
#define DEBUG(x) cerr << #x << " = " << (x) << endl
#define DEBUGV(v) cerr << #v << " = "; for (auto x : v) cerr << x << " "; cerr << endl
#else
#define DEBUG(x)
#define DEBUGV(v)
#endif

// Fast I/O
#define FASTIO ios_base::sync_with_stdio(false); cin.tie(nullptr); cout.tie(nullptr)

int main() {
    FASTIO;
    
    // Your code here
    
    return 0;
}
\`\`\`

---

## When to Use vs Avoid Macros

### ✅ Use Macros For:

1. **Frequently typed code**
   - \`ALL(v)\` instead of \`v.begin(), v.end()\`
   - \`REP(i, n)\` for simple loops

2. **Conditional compilation**
   - Debug macros
   - Platform-specific code

3. **Constants**
   - \`MOD = 1e9 + 7\`
   - \`INF = 1e9\`

4. **Type aliases**
   - \`ll\` for \`long long\`
   - \`pii\` for \`pair<int, int>\`

### ❌ Avoid Macros For:

1. **Complex logic**
   - Use inline functions instead

2. **When type safety matters**
   - Use templates or \`constexpr\` functions

3. **Operations with side effects**
   - Anything involving \`++\`, \`--\`, function calls

4. **When debugging**
   - Macro errors are harder to understand

---

## Modern Alternatives to Macros

Many macros can be replaced with modern C++ features:

### Alternative 1: constexpr Instead of #define

\`\`\`cpp
// Old:
#define PI 3.14159265358979323846

// Modern:
constexpr double PI = 3.14159265358979323846;
\`\`\`

**Advantage:** Type-safe, debuggable

### Alternative 2: inline Functions Instead of Macros

\`\`\`cpp
// Old:
#define SQUARE(x) ((x) * (x))

// Modern:
inline int square(int x) {
    return x * x;
}

// Or template:
template<typename T>
inline T square(T x) {
    return x * x;
}
\`\`\`

**Advantage:** Type-safe, no side-effect issues

### Alternative 3: using Instead of typedef

\`\`\`cpp
// Old:
typedef long long ll;

// Modern:
using ll = long long;
\`\`\`

**Advantage:** Clearer syntax, works with templates

---

## Conditional Compilation

\`\`\`cpp
#ifdef LOCAL
    // Code for local testing
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif

#ifndef ONLINE_JUDGE
    // Alternative check
    freopen("input.txt", "r", stdin);
#endif
\`\`\`

### Compile with Flag

\`\`\`bash
# For local testing (with file I/O)
g++ -DLOCAL solution.cpp -o solution

# For submission (no file I/O)
g++ solution.cpp -o solution
\`\`\`

---

## Macro Library Examples from Top Coders

### Tourist's Template (Simplified)

\`\`\`cpp
using ll = long long;
using pii = pair<int, int>;
#define all(x) (x).begin(), (x).end()
#define sz(x) (int)(x).size()
\`\`\`

**Observation:** Tourist uses minimal macros, prefers modern C++!

### Benq's Template

\`\`\`cpp
#define F0R(i,a) for (int i=0; i<(a); i++)
#define FOR(i,a,b) for (int i=(a); i<=(b); i++)
#define R0F(i,a) for (int i=(a)-1; i>=0; i--)
#define ROF(i,a,b) for (int i=(a); i>=(b); i--)
#define trav(a,x) for (auto& a: x)
\`\`\`

### Jiangly's Approach

Minimal macros, heavy use of modern C++ features and templates.

**Takeaway:** Top coders use different styles. Find what works for YOU!

---

## The bits/stdc++.h Header

\`bits/stdc++.h\` is a non-standard header that includes everything.

\`\`\`cpp
#include <bits/stdc++.h>
\`\`\`

**Includes:** iostream, vector, algorithm, map, set, queue, stack, etc.

### Pros:
✅ Saves time (don't list each header)
✅ Never forget to include something
✅ Standard in competitive programming

### Cons:
❌ Non-standard (GCC only)
❌ Slower compilation
❌ Bad practice for production code

**In CP:** Almost everyone uses it. Go ahead!

**For production:** Use specific headers.

---

## Precompiling bits/stdc++.h

To speed up compilation:

\`\`\`bash
# Locate bits/stdc++.h
find /usr/include -name stdc++.h

# Precompile it
sudo g++ -std=c++17 /usr/include/x86_64-linux-gnu/c++/11/bits/stdc++.h

# Now compilation is faster!
\`\`\`

**Result:** Compilation time: 2s → 0.3s

---

## Your Personal Template

Here's a balanced template with useful macros:

\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

// Type aliases
using ll = long long;
using pii = pair<int, int>;
using vi = vector<int>;

// Constants
const int MOD = 1e9 + 7;
const int INF = 1e9;

// Shortcuts
#define ALL(v) (v).begin(), (v).end()
#define SIZE(v) (int)(v).size()
#define REP(i, n) for (int i = 0; i < (n); i++)

// Debug (compile with -DLOCAL for debug output)
#ifdef LOCAL
#define DEBUG(x) cerr << #x << " = " << (x) << endl
#else
#define DEBUG(x)
#endif

void solve() {
    // Your solution here
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int t = 1;
    // cin >> t;  // Uncomment for multiple test cases
    
    while (t--) {
        solve();
    }
    
    return 0;
}
\`\`\`

**Customize this to your preferences!**

---

## Building Your Macro Habits

### Week 1: Start Simple
Use only:
- \`ALL(v)\`
- \`REP(i, n)\`
- \`ll\` for long long

### Week 2-3: Add More
- \`DEBUG(x)\`
- \`SIZE(v)\`
- \`pii\`, \`vi\`

### Week 4+: Finalize
- Add macros that YOU use frequently
- Remove macros you never use
- **Your template should evolve with you!**

---

## Macro Best Practices

1. **Be Consistent**
   - If you use \`ALL(v)\`, always use it
   - Don't mix \`v.begin(), v.end()\` and \`ALL(v)\`

2. **Use Descriptive Names**
   - \`REP\` is clear
   - \`R\` alone is not

3. **Parenthesize Everything**
   - \`#define SQUARE(x) ((x) * (x))\`
   - Not: \`#define SQUARE(x) x * x\`

4. **Avoid Macros for Complex Logic**
   - Use functions or templates

5. **Comment Your Macros**
   - Future you will thank you

6. **Test Your Macros**
   - Before using in contest
   - With edge cases

---

## Common Macro Mistakes

### 1. Forgetting Semicolon

\`\`\`cpp
#define PRINT(x) cout << x << endl  // No semicolon!

PRINT(5);  // OK
PRINT(5)   // Compile error!
\`\`\`

**Solution:** Let users add semicolon themselves.

### 2. Macro Shadowing Variables

\`\`\`cpp
#define MIN 0

int MIN = 5;  // Error! MIN is already defined
\`\`\`

### 3. Using Macro Name in String

\`\`\`cpp
#define N 100

string s = "The value of N is ...";  // N replaced with 100!
// Becomes: "The value of 100 is ..."
\`\`\`

---

## Summary

**Essential Macros to Use:**

✅ \`using ll = long long;\`
✅ \`#define ALL(v) (v).begin(), (v).end()\`
✅ \`#define REP(i, n) for (int i = 0; i < (n); i++)\`
✅ \`#define DEBUG(x) ...\` (with \`#ifdef LOCAL\`)

**Macros to Avoid:**

❌ Complex function-like macros (use inline functions)
❌ Macros with side effects
❌ Too many obscure abbreviations

**Golden Rule:**

> "Use macros to save time, but not at the cost of clarity and correctness."

**Your Action:**

1. Start with the minimal template provided
2. Add macros as you find yourself typing the same thing repeatedly
3. Test your template on 10-20 problems
4. Refine based on what you actually use

---

## Next Steps

Now that you have a powerful macro library, let's dive into **Bits, Bytes & Bitwise Operations**—essential for many CP problems!

**Key Takeaway**: Macros are tools. Used wisely, they save time. Used carelessly, they cause bugs. Build your personal macro library thoughtfully and iteratively.
`,
  quizId: 'cp-m1-s8-quiz',
  discussionId: 'cp-m1-s8-discussion',
} as const;
