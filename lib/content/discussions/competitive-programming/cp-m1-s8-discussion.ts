export default {
    id: 'cp-m1-s8-discussion',
    title: 'Macros & Preprocessor Tricks - Discussion Questions',
    questions: [
        {
            question: 'Macros can save significant typing time in competitive programming but can also lead to hard-to-debug errors. Discuss the trade-offs and explain which macros are worth using and which should be avoided.',
            answer: `Macros are controversial in competitive programming. Here's the balanced perspective:

**What Macros Are:**

Text replacement before compilation:
\`\`\`cpp
#define MAXN 100000
#define ll long long
#define pb push_back
#define all(v) v.begin(), v.end()
#define rep(i,n) for(int i=0; i<(n); i++)
\`\`\`

Preprocessor replaces text literally before compiling.

**Advantages:**

✅ **Saves Typing:** \`rep(i, n)\` vs \`for(int i = 0; i<n; i++)\`
✅ **Less Error-Prone:** No typos in loop syntax
✅ **Readable (Once Learned):** \`all(v)\` clearer than \`v.begin(), v.end()\`
✅ **Flexible:** Can create complex patterns
✅ **Fast to Write:** Critical in timed contests

**Disadvantages:**

❌ **Hard to Debug:** Errors show expanded macro, not original
❌ **No Type Safety:** Just text replacement
❌ **Side Effects:** Can evaluate arguments multiple times
❌ **Namespace Pollution:** Macros don't respect scope
❌ **IDE Confusion:** Autocomplete doesn't work well

**Safe, Recommended Macros:**

**1. Type Shortcuts:**
\`\`\`cpp
#define ll long long
#define ull unsigned long long
#define ld long double

// Usage:
ll sum = 0;  // Clear, no side effects
\`\`\`

**2. STL Shortcuts:**
\`\`\`cpp
#define pb push_back
#define mp make_pair
#define all(v) v.begin(), v.end()
#define rall(v) v.rbegin(), v.rend()
#define sz(v) (int)v.size()

// Usage:
v.pb(5);
sort(all(v));
\`\`\`

**3. Constants:**
\`\`\`cpp
#define INF 1e9
#define LINF 1e18
#define MOD 1000000007
#define EPS 1e-9

// Usage:
int maxVal = INF;
ll result = (result + x) % MOD;
\`\`\`

**Questionable Macros (Use with Caution):**

**1. Loop Macros:**
\`\`\`cpp
#define rep(i,n) for(int i=0; i<(n); i++)
#define rep1(i,n) for(int i=1; i<=(n); i++)
#define rrep(i,n) for(int i=(n)-1; i>=0; i--)

// Pros: Fast to write
rep(i, n) {
    // ...
}

// Cons: Less clear to beginners, harder to debug
\`\`\`

**2. Pair Shortcuts:**
\`\`\`cpp
#define F first
#define S second

// Usage:
pair<int, int> p;
cout << p.F << " " << p.S;

// Con: Extremely short, can be confusing
\`\`\`

**Dangerous Macros (Avoid):**

**1. Multiple Evaluation:**
\`\`\`cpp
#define SQUARE(x) x * x

int a = 5;
cout << SQUARE(a + 1);  // Expands to: a + 1 * a + 1 = 5 + 5 + 1 = 11 (WRONG!)

// Should be (a+1) * (a+1) = 36

// Fix with parentheses:
#define SQUARE(x) ((x) * (x))
cout << SQUARE(a + 1);  // Now: ((a+1) * (a+1)) = 36 (CORRECT)

// But still dangerous:
cout << SQUARE(a++);  // a++ evaluated twice! Undefined behavior
\`\`\`

**2. Hidden Control Flow:**
\`\`\`cpp
#define FAIL_IF(cond) if(cond) return -1

int process() {
    FAIL_IF(x < 0);  // Hidden return!
    // ...
}

// Con: Not obvious function can return here
\`\`\`

**3. Type-Unsafe Operations:**
\`\`\`cpp
#define MAX(a,b) ((a) > (b) ? (a) : (b))

int x = 5, y = 10;
cout << MAX(x++, y++);  // Both incremented! Undefined behavior

// Better: Use std::max which is type-safe
\`\`\`

**Debugging Issues:**

**Problem: Macro Expansion Errors**
\`\`\`cpp
#define rep(i,n) for(int i=0; i<(n); i++)

rep(i, n) {
    // Error here
}

// Compiler shows:
// Error on line X: for(int i=0; i<(n); i++) {
// Not clear which macro caused it
\`\`\`

**Solution:**
- Comment out macros during debugging
- Use -E flag to see expanded code: \`g++ - E solution.cpp\`

**Best Practices:**

**1. Always Use Parentheses:**
\`\`\`cpp
// WRONG:
#define DOUBLE(x) x * 2

// RIGHT:
#define DOUBLE(x) ((x) * 2)
\`\`\`

**2. Use Uppercase for Macro Names:**
\`\`\`cpp
// Clear it's a macro:
#define MAXN 100000
#define MOD 1000000007

// Avoid (looks like variable):
#define maxn 100000
\`\`\`

**3. Keep Macros Simple:**
\`\`\`cpp
// GOOD: Simple, clear
#define all(v) v.begin(), v.end()

// BAD: Complex, error-prone
#define DO_COMPLEX_THING(x, y, z) /* complicated multi-line macro */
\`\`\`

**4. Document Unusual Macros:**
\`\`\`cpp
// Iterate from i=0 to i<n
#define rep(i,n) for(int i=0; i<(n); i++)
\`\`\`

**5. Use const/constexpr for Constants:**
\`\`\`cpp
// Macro:
#define MAXN 100000

// Better (type-safe):
const int MAXN = 100000;
constexpr int MAXN = 100000;

// Both work in array declarations:
int arr[MAXN];
\`\`\`

**When Macros Make Sense:**

✅ Contest environment (speed matters)
✅ Frequently used patterns
✅ Non-critical code quality
✅ Personal templates (you know the risks)

**When to Avoid Macros:**

❌ Production code
❌ Team projects
❌ Complex logic
❌ When alternatives exist (templates, inline functions)

**Alternatives to Macros:**

**Instead of MAX macro:**
\`\`\`cpp
// Macro:
#define MAX(a,b) ((a) > (b) ? (a) : (b))

// Better: Use std::max
int m = max(a, b);

// Or template:
template<typename T>
T maximum(T a, T b) { return (a > b) ? a : b; }
\`\`\`

**Instead of loop macro:**
\`\`\`cpp
// Macro:
#define rep(i,n) for(int i=0; i<(n); i++)
rep(i, n) { ... }

// Alternative: Just write the loop
for(int i = 0; i < n; i++) { ... }

// Or C++20 ranges (when available):
for(int i : views::iota(0, n)) { ... }
\`\`\`

**My Recommended Macro Set:**

Minimal but useful:
\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

// Type shortcuts
#define ll long long
#define ld long double

// STL helpers
#define all(v) v.begin(), v.end()
#define sz(v) (int)v.size()

// Constants
const int MOD = 1e9 + 7;
const ll INF = 1e18;

// Optional (if you like them):
#define pb push_back
#define mp make_pair
#define F first
#define S second
\`\`\`

**Controversial Opinion:**

Some top competitive programmers use extensive macros:
\`\`\`cpp
#define rep(i,n) for(int i=0;i<(n);i++)
#define rep1(i,n) for(int i=1;i<=(n);i++)
#define rrep(i,n) for(int i=(n)-1;i>=0;i--)
#define FOR(i,a,b) for(int i=(a);i<(b);i++)
// etc.
\`\`\`

**My take:**
- If you're comfortable with them: Use them
- If you're learning: Start minimal, add gradually
- If you're unsure: Avoid complex macros

**Bottom Line:**

**Worth Using:**
- Type shortcuts (ll, ld)
- STL helpers (all, sz, pb)
- Constants (MOD, INF)

**Use Carefully:**
- Loop macros (if you like them)
- Pair shortcuts (F, S)

**Avoid:**
- Complex function-like macros
- Macros with side effects
- Macros that hide control flow

**The Right Balance:**
Use macros that save time without sacrificing debuggability. When in doubt, write explicit code—it's always safer!`,
    },
{
    question: 'Conditional compilation with #ifdef and #ifndef can be useful for debugging. Explain how to set up a robust debug system using preprocessor directives that works both locally and on online judges.',
        answer: `Conditional compilation is a powerful debugging tool. Here's how to use it effectively:

**The Problem:**

\`\`\`cpp
// Debug prints in code:
cout << "Debug: x = " << x << endl;

// Submit to judge:
// Wrong Answer (debug output messes up actual output!)

// Must manually remove/comment out debug prints
// Time-consuming and error-prone
\`\`\`

**The Solution: Conditional Compilation**

\`\`\`cpp
#ifdef LOCAL
    // This code only compiles when LOCAL is defined
    cerr << "Debug: x = " << x << endl;
#endif
\`\`\`

**Basic Setup:**

**1. Define LOCAL when compiling locally:**
\`\`\`bash
g++ -DLOCAL -std=c++17 -O2 solution.cpp -o solution
\`\`\`

**2. Don't define LOCAL when submitting:**
Just submit the code as-is. Online judges won't define LOCAL.

**Debug Macro System:**

\`\`\`cpp
#ifdef LOCAL
    #define debug(x) cerr << #x << " = " << (x) << endl
    #define debugArr(a, n) cerr << #a << " = "; for(int i=0;i<(n);i++) cerr << a[i] << " "; cerr << endl
    #define debugVec(v) cerr << #v << " = "; for(auto x : v) cerr << x << " "; cerr << endl
    #define debugPair(p) cerr << #p << " = (" << p.first << ", " << p.second << ")" << endl
#else
    #define debug(x)
    #define debugArr(a, n)
    #define debugVec(v)
    #define debugPair(p)
#endif

int main() {
    int x = 42;
    vector<int> v = {1, 2, 3, 4, 5};
    
    debug(x);      // Shows locally: "x = 42"
    debugVec(v);   // Shows locally: "v = 1 2 3 4 5"
    
    // Submitted to judge: These do nothing!
    
    cout << x << endl;  // Actual output
    return 0;
}
\`\`\`

**Why cerr vs cout:**

\`\`\`cpp
#ifdef LOCAL
    #define debug(x) cerr << #x << " = " << x << endl  // cerr!
#endif

// cerr goes to stderr (error stream)
// cout goes to stdout (standard output)
// Judge only sees stdout, not stderr
// Can leave debug statements even on judge!
\`\`\`

**Advanced Debug System:**

\`\`\`cpp
#ifdef LOCAL
    // Timestamp
    #include <chrono>
    auto start_time = chrono::high_resolution_clock::now();
    
    #define debug(x) cerr << "[" << __LINE__ << "] " << #x << " = " << (x) << endl
    #define debugTime() cerr << "Time: " << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_time).count() << "ms" << endl
    
    // Print with line number
    #define debugVec(v) cerr << "[" << __LINE__ << "] " << #v << " = "; \\
                        for(auto x : v) cerr << x << " "; cerr << endl
#else
    #define debug(x)
    #define debugTime()
    #define debugVec(v)
#endif

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int n;
    cin >> n;
    debug(n);  // Shows line number: [23] n = 100
    
    vector<int> v(n);
    for(auto& x : v) cin >> x;
    debugVec(v);  // Shows: [28] v = 1 2 3 4 5
    
    // Process...
    debugTime();  // Shows: Time: 15ms
    
    return 0;
}
\`\`\`

**File I/O for Local Testing:**

\`\`\`cpp
int main() {
    #ifdef LOCAL
        freopen("input.txt", "r", stdin);
        freopen("output.txt", "w", stdout);
        freopen("error.txt", "w", stderr);
    #endif
    
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    // Your solution...
    // Reads from input.txt locally
    // Reads from stdin on judge
    
    return 0;
}
\`\`\`

**Compilation Script:**

Create \`compile.sh\`:
\`\`\`bash
#!/bin/bash

# Local compilation with debug
g++ -DLOCAL -std=c++17 -O2 -Wall -Wextra -Wshadow solution.cpp -o solution

# Or for judge (no LOCAL)
# g++ -std=c++17 -O2 solution.cpp -o solution
\`\`\`

**Makefile:**

\`\`\`makefile
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

# Local build (with debug)
local: solution.cpp
	$(CXX) -DLOCAL $(CXXFLAGS) solution.cpp -o solution

# Judge build (no debug)
judge: solution.cpp
	$(CXX) $(CXXFLAGS) solution.cpp -o solution

# Run with input
run: local
	./solution < input.txt

# Clean
clean:
	rm -f solution
\`\`\`

Usage: \`make local && make run\`

**VS Code Configuration:**

tasks.json:
\`\`\`json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Compile (Debug)",
            "type": "shell",
            "command": "g++",
            "args": [
                "-DLOCAL",
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "\${file}",
                "-o",
                "\${fileDirname}/solution"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            }
        },
        {
            "label": "Run",
            "type": "shell",
            "command": "\${fileDirname}/solution < \${fileDirname}/input.txt",
            "dependsOn": ["Compile (Debug)"]
        }
    ]
}
\`\`\`

**Comprehensive Template:**

\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

// ==================== DEBUG SYSTEM ====================
#ifdef LOCAL
    #define debug(x) cerr << #x << " = " << (x) << endl
    #define debugVec(v) cerr << #v << " = ["; \\
                        for(size_t i=0; i<v.size(); i++) { \\
                            if(i>0) cerr << ", "; \\
                            cerr << v[i]; \\
                        } cerr << "]" << endl
    #define debugPair(p) cerr << #p << " = (" << p.first << ", " << p.second << ")" << endl
    #define debugMap(m) cerr << #m << " = {"; \\
                        bool first=true; \\
                        for(auto& [k,v] : m) { \\
                            if(!first) cerr << ", "; \\
                            cerr << k << ":" << v; \\
                            first=false; \\
                        } cerr << "}" << endl
    #define trace() cerr << "Line " << __LINE__ << " executed" << endl
#else
    #define debug(x)
    #define debugVec(v)
    #define debugPair(p)
    #define debugMap(m)
    #define trace()
#endif

// ==================== TYPE DEFINITIONS ====================
#define ll long long
#define ld long double
#define all(v) v.begin(), v.end()

// ==================== SOLUTION ====================
void solve() {
    int n;
    cin >> n;
    debug(n);  // Only shows locally
    
    vector<int> arr(n);
    for(auto& x : arr) cin >> x;
    debugVec(arr);  // Only shows locally
    
    // Your solution...
    
    cout << "answer" << endl;  // Goes to judge
}

int main() {
    #ifdef LOCAL
        freopen("input.txt", "r", stdin);
        freopen("output.txt", "w", stdout);
        freopen("error.txt", "w", stderr);
    #endif
    
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    solve();
    
    return 0;
}
\`\`\`

**Assertion System:**

\`\`\`cpp
#ifdef LOCAL
    #define ASSERT(condition, message) \\
        if(!(condition)) { \\
            cerr << "Assertion failed at line " << __LINE__ << ": " << message << endl; \\
            exit(1); \\
        }
#else
    #define ASSERT(condition, message)
#endif

int main() {
    int n;
    cin >> n;
    ASSERT(n > 0, "n must be positive");  // Checks locally only
    ASSERT(n <= 100000, "n too large");
    
    // Process...
}
\`\`\`

**Multiple Debug Levels:**

\`\`\`cpp
#ifdef LOCAL
    #define DEBUG_LEVEL 2  // 0=none, 1=basic, 2=verbose
#else
    #define DEBUG_LEVEL 0
#endif

#if DEBUG_LEVEL >= 1
    #define debug1(x) cerr << #x << " = " << (x) << endl
#else
    #define debug1(x)
#endif

#if DEBUG_LEVEL >= 2
    #define debug2(x) cerr << "VERBOSE: " << #x << " = " << (x) << endl
#else
    #define debug2(x)
#endif

int main() {
    int x = 5;
    debug1(x);  // Shows if DEBUG_LEVEL >= 1
    debug2(x);  // Shows only if DEBUG_LEVEL >= 2
}
\`\`\`

**Best Practices:**

1. **Always use cerr for debug**, not cout
2. **Keep debug macros simple**
3. **Test without LOCAL flag** before submitting
4. **Use consistent naming** (debug, debugVec, etc.)
5. **Document your system** in template

**Common Pitfalls:**

**Pitfall 1: Using cout for debug**
\`\`\`cpp
#ifdef LOCAL
    cout << "Debug: " << x << endl;  // WRONG! Goes to stdout
#endif
\`\`\`

**Pitfall 2: Forgetting to undefine LOCAL**
\`\`\`cpp
#define LOCAL  // In submitted code - judge sees debug output!
\`\`\`

Solution: Never define LOCAL in code, only via compiler flag

**Pitfall 3: Macro side effects**
\`\`\`cpp
#define debug(x) cerr << #x << " = " << x++ << endl
debug(i);  // i incremented! Behavior different with/without LOCAL
\`\`\`

Solution: Never modify in debug macros

**Bottom Line:**

A good debug system:
- ✅ Works locally (shows debug info)
- ✅ Works on judge (no debug output)
- ✅ Uses cerr (separate from actual output)
- ✅ Easy to use (simple macros)
- ✅ Doesn't affect logic (no side effects)

Set it up once in your template and use it in every contest!`,
    },
{
    question: 'Some competitive programmers create extensive macro libraries with dozens of shortcuts. Others use minimal macros. Discuss your philosophy on macro usage and what belongs in a competitive programming template.',
        answer: `The macro philosophy debate is one of the most contentious in competitive programming. Here's my comprehensive take:

**The Spectrum:**

**Minimalist (No/Few Macros):**
\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    // Write everything explicitly
    for(int i = 0; i < n; i++) {
        // ...
    }
    sort(v.begin(), v.end());
    
    return 0;
}
\`\`\`

**Moderate (Selective Macros):**
\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

#define ll long long
#define all(v) v.begin(), v.end()

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    ll sum = 0;
    vector<int> v;
    sort(all(v));
    
    return 0;
}
\`\`\`

**Maximalist (Heavy Macros):**
\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

#define ll long long
#define ull unsigned long long
#define ld long double
#define vi vector<int>
#define vll vector<ll>
#define pii pair<int,int>
#define pll pair<ll,ll>
#define pb push_back
#define mp make_pair
#define all(v) v.begin(),v.end()
#define rall(v) v.rbegin(),v.rend()
#define sz(v) (int)v.size()
#define F first
#define S second
#define rep(i,n) for(int i=0;i<(n);i++)
#define rep1(i,n) for(int i=1;i<=(n);i++)
#define rrep(i,n) for(int i=(n)-1;i>=0;i--)
#define FOR(i,a,b) for(int i=(a);i<(b);i++)
#define endl '\\n'
// ... 50 more macros

int main(){
    ios_base::sync_with_stdio(false);cin.tie(0);
    int n;cin>>n;
    vi v(n);rep(i,n)cin>>v[i];
    sort(all(v));
    rep(i,n)cout<<v[i]<<" ";cout<<endl;
    return 0;
}
\`\`\`

**Arguments For Heavy Macros:**

✅ **Speed:** Save 2-3 seconds per problem
✅ **Fewer Typos:** Less chance of syntax errors
✅ **Muscle Memory:** Once learned, very fast
✅ **Top Coders Use Them:** Many red coders have extensive macros
✅ **Contest Efficiency:** Every second counts

**Arguments Against Heavy Macros:**

❌ **Hard to Read:** Especially for others or your future self
❌ **Debugging Difficulty:** Errors show expanded code
❌ **Learning Curve:** Must memorize many shortcuts
❌ **Bad Habits:** Doesn't translate to real programming
❌ **Maintenance:** More to keep track of
❌ **Confusion:** Easy to forget what a macro does

**My Philosophy: The Balanced Approach**

Start minimal, add only what you actually use frequently:

**Tier 1: Essential (Everyone Should Have)**
\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

// Type shortcuts
#define ll long long
#define ld long double

// STL helpers
#define all(v) v.begin(), v.end()

// Constants
const int MOD = 1e9 + 7;
const ll INF = 1e18;
\`\`\`

**Tier 2: Common (Add If You Use Often)**
\`\`\`cpp
// More STL
#define sz(v) (int)v.size()
#define pb push_back

// Pairs
#define F first
#define S second
\`\`\`

**Tier 3: Personal Preference (Optional)**
\`\`\`cpp
// Loop macros (if you like them)
#define rep(i,n) for(int i=0;i<(n);i++)

// Debug
#ifdef LOCAL
#define debug(x) cerr << #x << " = " << x << endl
#else
#define debug(x)
#endif
\`\`\`

**What Should ALWAYS Be in Your Template:**

**1. Fast I/O:**
\`\`\`cpp
ios_base::sync_with_stdio(false);
cin.tie(nullptr);
\`\`\`

**2. Main Structure:**
\`\`\`cpp
void solve() {
    // Solution here
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int t = 1;
    // cin >> t;  // Uncomment for multiple test cases
    while(t--) solve();
    
    return 0;
}
\`\`\`

**3. A Few Key Macros:**
\`\`\`cpp
#define ll long long
#define all(v) v.begin(), v.end()
\`\`\`

**What's Debatable:**

**Loop Macros:**
\`\`\`cpp
#define rep(i,n) for(int i=0;i<(n);i++)

// Pro: rep(i,n) faster than for(int i=0;i<n;i++)
// Con: Less readable, especially for beginners
\`\`\`

My take: If you're comfortable, use them. If not, don't.

**Pair Shortcuts:**
\`\`\`cpp
#define F first
#define S second

// Pro: p.F shorter than p.first
// Con: Very cryptic, especially F and S
\`\`\`

My take: Borderline too short. I prefer writing .first/.second

**What Should NEVER Be in Your Template:**

**1. Untested Code:**
Never include functions/macros you haven't tested!

**2. Hundreds of Lines:**
If template is >100 lines, it's probably too much

**3. Complex Logic:**
\`\`\`cpp
// DON'T:
#define DO_COMPLICATED_THING(args...) /* 20 lines of code */

// This belongs in a function, not a macro
\`\`\`

**4. Macros You Don't Remember:**
If you can't recall what it does, don't include it

**Evolution Strategy:**

**Week 1: Minimal**
\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

#define ll long long

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    return 0;
}
\`\`\`

**Week 5: Add What You Need**
\`\`\`cpp
// Started using all(v) frequently?
#define all(v) v.begin(), v.end()

// Using pairs a lot?
#define F first
#define S second
\`\`\`

**Month 3: Stable Template**
\`\`\`cpp
// Now you have 10-15 macros you actually use
// Comfortable and efficient
\`\`\`

**Real Example: My Actual Template**

\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

// ==================== MACROS ====================
#define ll long long
#define ld long double
#define all(v) v.begin(), v.end()
#define sz(v) (int)v.size()
#define pb push_back
#define F first
#define S second

// ==================== CONSTANTS ====================
const int MOD = 1e9 + 7;
const ll INF = 1e18;

// ==================== DEBUG ====================
#ifdef LOCAL
#define debug(x) cerr << #x << " = " << x << endl
#define debugVec(v) cerr << #v << " = "; for(auto x:v) cerr << x << " "; cerr << endl
#else
#define debug(x)
#define debugVec(v)
#endif

// ==================== UTILITY FUNCTIONS ====================
template<typename T>
T gcd(T a, T b) { return b ? gcd(b, a%b) : a; }

ll power(ll a, ll b, ll mod = MOD) {
    ll res = 1; a %= mod;
    while(b > 0) {
        if(b & 1) res = (res * a) % mod;
        a = (a * a) % mod;
        b >>= 1;
    }
    return res;
}

// ==================== SOLUTION ====================
void solve() {
    // Solution here
}

// ==================== MAIN ====================
int main() {
    #ifdef LOCAL
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
    #endif
    
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int t = 1;
    // cin >> t;
    while(t--) solve();
    
    return 0;
}
\`\`\`

**Size: ~50 lines**
**Memorized: Yes**
**Tested: Yes**
**Useful: Every contest**

**Recommendation by Level:**

**Beginner (< 6 months CP):**
- Minimal macros
- Focus on learning algorithms
- Template should be <30 lines

**Intermediate (6-18 months):**
- Add macros you use frequently
- 30-50 line template
- Balance convenience and clarity

**Advanced (18+ months):**
- Use whatever works for you
- You know the trade-offs
- Probably 40-70 lines

**Bottom Line:**

**My Philosophy:**
1. **Start minimal** - don't cargo-cult others' templates
2. **Add incrementally** - only what you actually use
3. **Keep it maintainable** - should fit on one screen
4. **Test everything** - never untested code in template
5. **Be consistent** - use same template in practice and contests

**The goal:** Fast enough to save time, simple enough to not cause bugs.

**Remember:** The best template is one YOU understand completely and can modify quickly when needed!`,
    },
  ],
} as const ;

