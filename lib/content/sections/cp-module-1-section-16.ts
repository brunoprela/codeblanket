export const buildingRobustCpTemplateSection = {
  id: 'cp-m1-s16',
  title: 'Building Your Starter Template',
  content: `

# Building Your Starter Template

## Introduction

Every competitive programmer has their secret weapon: **the template**. That magical file they copy at the start of every contest, filled with pre-written code, macros, and utility functions. It's like a Swiss Army knife for competitive programming—always ready, always useful.

But here's the thing: **your template should be YOURS**. Not copied blindly from someone else. Built gradually, tested thoroughly, and customized to YOUR coding style and needs. A good template saves 2-5 minutes per problem. In a 2-hour contest with 5 problems, that's 10-25 minutes saved—enough time to solve an extra problem!

In this final section of Module 1, we'll build a comprehensive starter template from scratch, understand each component, explore customization options, and learn best practices for template management.

**Goal**: Create a personalized, battle-tested competitive programming template that saves time and reduces errors.

---

## Template Philosophy

### What Makes a Good Template?

**DO include:**
✅ Fast I/O setup
✅ Common macros you actually use
✅ Utility functions you need frequently
✅ Proper structure (main, solve, etc.)
✅ Easy-to-modify components

**DON'T include:**
❌ Code you don't understand
❌ Macros you never use
❌ Complex data structures you haven't tested
❌ Functions for every possible scenario
❌ Hundreds of lines of "just in case" code

**Philosophy:** Start minimal, add gradually as you need things.

### Template Evolution

**Stage 1: Beginner (20-30 lines)**
- Basic includes
- Main function
- Fast I/O

**Stage 2: Intermediate (50-100 lines)**
- Common macros
- Basic utility functions
- Multiple test case handling

**Stage 3: Advanced (100-200 lines)**
- Tested utility functions
- Custom data structures
- Problem-specific variants

**Don't rush to Stage 3!** Build your template gradually through practice.

---

## Core Template Structure

### Minimal Template (Stage 1)

\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    // Your solution here
    
    return 0;
}
\`\`\`

**This works!** Start here and build up.

### Basic Template (Stage 2)

\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

#define ll long long
#define ld long double
#define all(v) v.begin(), v.end()
#define rep(i,n) for(int i = 0; i < (n); i++)

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

**This is practical!** Most problems need this structure.

---

## Component 1: Includes and Namespaces

### The Standard Header

\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;
\`\`\`

**What is \`bits/ stdc++.h\`?**
- g++ extension (not standard C++)
- Includes ALL standard library headers
- Perfect for competitive programming
- DON'T use in production code

**Alternative (if bits/stdc++.h not available):**
\`\`\`cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <map>
#include <set>
#include <queue>
#include <stack>
#include <cmath>
#include <cstring>
#include <iomanip>
using namespace std;
\`\`\`

**In CP:** Just use \`bits / stdc++.h\`!

### Using namespace std

\`\`\`cpp
using namespace std;
\`\`\`

**Why?**
- Saves typing \`std:: \` everywhere
- Standard in competitive programming
- Don't worry about namespace pollution in CP

**In production:** DON'T use \`using namespace std; \`!

---

## Component 2: Type Definitions

### Common Type Shortcuts

\`\`\`cpp
// Integer types
#define ll long long
#define ull unsigned long long
#define ld long double

// Containers
#define vi vector<int>
#define vll vector<long long>
#define vvi vector<vector<int>>
#define pii pair<int, int>
#define pll pair<long long, long long>
#define vpii vector<pair<int, int>>

// Map and set
#define mii map<int, int>
#define si set<int>
#define usi unordered_set<int>
#define umii unordered_map<int, int>
\`\`\`

**Customize based on what YOU use!**

**Minimal version (recommended):**
\`\`\`cpp
#define ll long long
#define ld long double
#define pii pair<int, int>
#define vi vector<int>
\`\`\`

---

## Component 3: Utility Macros

### Container Operations

\`\`\`cpp
#define all(v) v.begin(), v.end()
#define rall(v) v.rbegin(), v.rend()
#define sz(v) (int)v.size()
#define pb push_back
#define mp make_pair
#define F first
#define S second
\`\`\`

**Usage:**
\`\`\`cpp
vector<int> v = {3, 1, 4, 1, 5};

// Without macros:
sort(v.begin(), v.end());
int size = (int)v.size();

// With macros:
sort(all(v));
int size = sz(v);
\`\`\`

### Loop Macros

\`\`\`cpp
#define rep(i, n) for(int i = 0; i < (n); i++)
#define rep1(i, n) for(int i = 1; i <= (n); i++)
#define repr(i, n) for(int i = (n)-1; i >= 0; i--)
#define FOR(i, a, b) for(int i = (a); i < (b); i++)
\`\`\`

**Usage:**
\`\`\`cpp
// Without macros:
for(int i = 0; i < n; i++) {
    cout << arr[i] << " ";
}

// With macros:
rep(i, n) {
    cout << arr[i] << " ";
}
\`\`\`

**Warning:** Some people find loop macros confusing. Use only if comfortable!

### Constants

\`\`\`cpp
#define INF 1e9
#define LINF 1e18
#define MOD 1000000007
#define MOD2 998244353
#define PI 3.14159265358979323846
#define EPS 1e-9
\`\`\`

**Usage:**
\`\`\`cpp
const ll MOD = 1e9 + 7;
ll ans = (ans + x) % MOD;

int maxDist = INF;
if(dist < maxDist) { ... }
\`\`\`

---

## Component 4: Utility Functions

### Mathematical Functions

\`\`\`cpp
// GCD
template<typename T>
T gcd(T a, T b) {
    return b ? gcd(b, a % b) : a;
}

// LCM
template<typename T>
T lcm(T a, T b) {
    return a / gcd(a, b) * b;
}

// Power (with mod)
ll power(ll a, ll b, ll mod = MOD) {
    ll res = 1;
    a %= mod;
    while(b > 0) {
        if(b & 1) res = (res * a) % mod;
        a = (a * a) % mod;
        b >>= 1;
    }
    return res;
}

// Modular inverse
ll modInv(ll a, ll mod = MOD) {
    return power(a, mod - 2, mod);
}
\`\`\`

### Container Functions

\`\`\`cpp
// Read array
template<typename T>
vector<T> readArray(int n) {
    vector<T> v(n);
    for(auto& x : v) cin >> x;
    return v;
}

// Print array
template<typename T>
void printArray(const vector<T>& v) {
    for(const auto& x : v) cout << x << " ";
    cout << "\\n";
}

// Create 2D vector
template<typename T>
vector<vector<T>> create2D(int n, int m, T val = T()) {
    return vector<vector<T>>(n, vector<T>(m, val));
}

// Sum of vector
template<typename T>
T sum(const vector<T>& v) {
    return accumulate(all(v), (T)0);
}

// Max element
template<typename T>
T maxElement(const vector<T>& v) {
    return *max_element(all(v));
}

// Min element
template<typename T>
T minElement(const vector<T>& v) {
    return *min_element(all(v));
}
\`\`\`

### Debugging Functions

\`\`\`cpp
// Debug macro (only works when LOCAL is defined)
#ifdef LOCAL
#define debug(x) cerr << #x << " = " << (x) << endl
#define debugArr(a, n) cerr << #a << " = "; for(int i=0;i<(n);i++) cerr<<a[i]<<" "; cerr<<endl
#define debugVec(v) cerr << #v << " = "; for(auto x : v) cerr << x << " "; cerr << endl
#else
#define debug(x)
#define debugArr(a, n)
#define debugVec(v)
#endif

// Compile with: g++ -DLOCAL solution.cpp
// Debug prints will only show locally!
\`\`\`

**Usage:**
\`\`\`cpp
int x = 42;
vector<int> v = {1, 2, 3};

debug(x);       // x = 42
debugVec(v);    // v = 1 2 3
\`\`\`

---

## Component 5: Main Function Structure

### Single Test Case

\`\`\`cpp
void solve() {
    // Solution here
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    solve();
    
    return 0;
}
\`\`\`

### Multiple Test Cases

\`\`\`cpp
void solve() {
    // Solution here
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int t;
    cin >> t;
    
    while(t--) {
        solve();
    }
    
    return 0;
}
\`\`\`

### Flexible (Choose at Runtime)

\`\`\`cpp
void solve() {
    // Solution here
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int t = 1;
    // cin >> t;  // Uncomment if multiple test cases
    
    while(t--) {
        solve();
    }
    
    return 0;
}
\`\`\`

**This is the most flexible!** Just uncomment one line for multiple tests.

---

## Complete Template Examples

### Minimal Template (Recommended for Beginners)

\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

#define ll long long
#define all(v) v.begin(), v.end()

void solve() {
    // Solution here
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int t = 1;
    // cin >> t;
    while(t--) solve();
    
    return 0;
}
\`\`\`

### Intermediate Template (Recommended for Most)

\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

// Type definitions
#define ll long long
#define ld long double
#define pii pair<int, int>
#define vi vector<int>
#define vll vector<long long>

// Macros
#define all(v) v.begin(), v.end()
#define rall(v) v.rbegin(), v.rend()
#define rep(i, n) for(int i = 0; i < (n); i++)
#define pb push_back
#define mp make_pair
#define F first
#define S second

// Constants
const ll MOD = 1e9 + 7;
const ll INF = 1e18;
const ld PI = 3.14159265358979323846;

// Utility functions
template<typename T>
T gcd(T a, T b) { return b ? gcd(b, a%b) : a; }

ll power(ll a, ll b, ll mod = MOD) {
    ll res = 1;
    a %= mod;
    while(b > 0) {
        if(b & 1) res = (res * a) % mod;
        a = (a * a) % mod;
        b >>= 1;
    }
    return res;
}

// Debug (only in LOCAL mode)
#ifdef LOCAL
#define debug(x) cerr << #x << " = " << (x) << endl
#else
#define debug(x)
#endif

void solve() {
    // Solution here
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

### Advanced Template (For Experienced Programmers)

\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

// ==================== TYPE DEFINITIONS ====================
#define ll long long
#define ull unsigned long long
#define ld long double
#define vi vector<int>
#define vll vector<long long>
#define vvi vector<vector<int>>
#define pii pair<int, int>
#define pll pair<long long, long long>
#define vpii vector<pair<int, int>>

// ==================== MACROS ====================
#define all(v) v.begin(), v.end()
#define rall(v) v.rbegin(), v.rend()
#define sz(v) (int)v.size()
#define pb push_back
#define mp make_pair
#define F first
#define S second
#define rep(i, n) for(int i = 0; i < (n); i++)
#define rep1(i, n) for(int i = 1; i <= (n); i++)
#define repr(i, n) for(int i = (n)-1; i >= 0; i--)
#define FOR(i, a, b) for(int i = (a); i < (b); i++)

// ==================== CONSTANTS ====================
const ll MOD = 1e9 + 7;
const ll MOD2 = 998244353;
const ll INF = 1e18;
const int MAXN = 2e5 + 5;
const ld PI = 3.14159265358979323846;
const ld EPS = 1e-9;

// ==================== UTILITY FUNCTIONS ====================

// Math
template<typename T> T gcd(T a, T b) { return b ? gcd(b, a%b) : a; }
template<typename T> T lcm(T a, T b) { return a / gcd(a, b) * b; }

ll power(ll a, ll b, ll mod = MOD) {
    ll res = 1; a %= mod;
    while(b > 0) {
        if(b & 1) res = (res * a) % mod;
        a = (a * a) % mod; b >>= 1;
    }
    return res;
}

ll modInv(ll a, ll mod = MOD) { return power(a, mod - 2, mod); }

// Containers
template<typename T>
vector<T> readArray(int n) {
    vector<T> v(n);
    for(auto& x : v) cin >> x;
    return v;
}

template<typename T>
void printArray(const vector<T>& v, string sep = " ") {
    for(int i = 0; i < sz(v); i++) {
        if(i > 0) cout << sep;
        cout << v[i];
    }
    cout << "\\n";
}

template<typename T>
vector<vector<T>> create2D(int n, int m, T val = T()) {
    return vector<vector<T>>(n, vector<T>(m, val));
}

// Debug
#ifdef LOCAL
#define debug(x) cerr << #x << " = " << (x) << endl
#define debugVec(v) cerr << #v << " = "; for(auto x : v) cerr << x << " "; cerr << endl
#define debugPair(p) cerr << #p << " = (" << p.F << ", " << p.S << ")" << endl
#else
#define debug(x)
#define debugVec(v)
#define debugPair(p)
#endif

// ==================== SOLUTION ====================

void solve() {
    // Solution here
}

// ==================== MAIN ====================

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    #ifdef LOCAL
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
    #endif
    
    int t = 1;
    // cin >> t;  // Uncomment for multiple test cases
    
    while(t--) {
        solve();
    }
    
    return 0;
}
\`\`\`

---

## Customization Guide

### What to Include?

**Include if you use it in 50%+ of problems:**
- Fast I/O: YES (almost every problem)
- \`ll\` typedef: YES (very common)
- \`all()\` macro: YES (used often)
- GCD function: MAYBE (depends on problem types)
- Graph algorithms: NO (problem-specific)

**Track your usage** for a few contests, then decide!

### Personal Preferences

**Some people prefer:**
\`\`\`cpp
#define int long long  // Make int = long long by default
// WARNING: Can cause issues with certain problems!
\`\`\`

**Others prefer:**
\`\`\`cpp
typedef long long ll;  // Instead of #define
typedef pair<int, int> pii;
\`\`\`

**Choose what works for YOU!**

---

## Template Management

### Organizing Templates

**Option 1: Single template**
- One \`template.cpp\`
- Modify for each problem

**Option 2: Multiple variants**
- \`template_simple.cpp\`
- \`template_graph.cpp\`
- \`template_math.cpp\`

**Option 3: Snippets**
- Use IDE snippets
- Type keyword → expands to code

**Recommended:** Start with Option 1, evolve to Option 2 or 3.

### Testing Your Template

**Important:** Test EVERY function in your template!

\`\`\`cpp
// test_template.cpp
#include "template.cpp"

void testGCD() {
    assert(gcd(12, 8) == 4);
    assert(gcd(7, 3) == 1);
    cout << "GCD tests passed\\n";
}

void testPower() {
    assert(power(2, 10, MOD) == 1024);
    assert(power(2, 0, MOD) == 1);
    cout << "Power tests passed\\n";
}

int main() {
    testGCD();
    testPower();
    cout << "All tests passed!\\n";
}
\`\`\`

**Never include untested code in your template!**

---

## Best Practices

### Do's and Don'ts

**✅ DO:**
- Start minimal, build gradually
- Test every function
- Understand every line
- Keep it organized
- Update after contests
- Comment complex parts

**❌ DON'T:**
- Copy blindly from others
- Include unused code
- Make it too complex
- Use without testing
- Add untested functions
- Forget to update

### Version Control

**Track your template evolution:**

\`\`\`
template_v1.cpp  # Minimal
template_v2.cpp  # Added macros
template_v3.cpp  # Added functions
template_current.cpp  # Your active template
\`\`\`

**Or use git:**
\`\`\`bash
git init
git add template.cpp
git commit -m "Initial template"
# Update as you go
\`\`\`

---

## Summary

**Key Principles:**

✅ **Start minimal** - Don't overcomplicate
✅ **Add gradually** - Build through practice
✅ **Test thoroughly** - Never trust untested code
✅ **Customize for you** - Not someone else's template
✅ **Keep organized** - Sections, comments
✅ **Update regularly** - Learn and improve

**Essential Components:**

1. **Includes**: \`#include < bits / stdc++.h > \`
2. **Fast I/O**: \`ios_base:: sync_with_stdio(false)\`
3. **Type shortcuts**: \`ll\`, \`pii\`, etc.
4. **Common macros**: \`all(v)\`, \`rep(i, n)\`
5. **Utility functions**: Only tested ones
6. **Main structure**: Handle single/multiple tests

**Template Levels:**

- **Minimal (20-30 lines)**: Perfect for beginners
- **Intermediate (50-100 lines)**: Most practical
- **Advanced (100-200 lines)**: For specific needs

**Build YOUR template:**
1. Start with minimal
2. Add macros you use often
3. Add functions you need repeatedly
4. Test everything
5. Refine through contests
6. Update and improve

---

## Conclusion: Module 1 Complete!

**🎉 Congratulations!** You've completed Module 1: C++ Setup & Fundamentals!

**What you've learned:**

✅ Algorithmic intuition and problem-solving mindset
✅ Why C++ is ideal for competitive programming
✅ Complete environment setup and compilation
✅ Modern CP tool ecosystem
✅ Fast input/output techniques
✅ C++ basics and modern features (C++11/14/17/20)
✅ Macros and preprocessor tricks
✅ Bits, bytes, and bitwise operations
✅ Memory management for CP
✅ Template metaprogramming basics
✅ Common compilation errors and fixes
✅ Debugging in competitive environment
✅ Reading other people's code
✅ Contest-day tips and strategies
✅ Building your personal template

**You now have:**
- Solid C++ foundation for competitive programming
- Understanding of the CP environment
- Debugging and problem-solving skills
- Your own tested template
- Contest-day strategies

**Next Steps:**

Module 2 will cover **Data Structures & STL**, where you'll master:
- vectors, arrays, strings
- stacks, queues, deques
- sets, maps, and their variants
- priority queues (heaps)
- pairs, tuples
- iterators and algorithms
- And much more!

**Keep practicing, keep learning, and most importantly—enjoy the journey!**

**Happy coding! 🚀**
`,
  quizId: 'cp-m1-s16-quiz',
  discussionId: 'cp-m1-s16-discussion',
} as const;
