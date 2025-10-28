export default {
  id: 'cp-m1-s11-discussion',
  title: 'Template Metaprogramming Basics - Discussion Questions',
  questions: [
    {
      question:
        'Templates enable generic programming but can make code harder to debug. Explain when templates are worth using in competitive programming versus when explicit types are better, with specific examples.',
      answer: `Templates are powerful but should be used judiciously in CP. Here's when each approach makes sense:

**When Templates Are Worth It:**

**1. Generic Utility Functions**

Problem: Need same function for multiple types

Without templates:
\`\`\`cpp
int maxInt(int a, int b) { return (a > b) ? a : b; }
long long maxLL(long long a, long long b) { return (a > b) ? a : b; }
double maxDouble(double a, double b) { return (a > b) ? a : b; }
\`\`\`

With templates:
\`\`\`cpp
template<typename T>
T maximum(T a, T b) {
    return (a > b) ? a : b;
}

// Works for all types
cout << maximum(5, 10) << endl;           // int
cout << maximum(3.14, 2.71) << endl;      // double
cout << maximum(5LL, 10LL) << endl;       // long long
\`\`\`

**Value:** DRY (Don't Repeat Yourself), fewer bugs from copy-paste

**2. Container-Agnostic Operations**

\`\`\`cpp
template<typename Container>
void print(const Container& c) {
    for(const auto& x : c) {
        cout << x << " ";
    }
    cout << "\\n";
}

// Works with any container
vector<int> v = {1, 2, 3};
set<int> s = {3, 1, 4};
list<string> l = {"a", "b"};

print(v);  // All work!
print(s);
print(l);
\`\`\`

**3. Type-Safe Comparators**

\`\`\`cpp
template<typename T>
struct Greater {
    bool operator()(const T& a, const T& b) const {
        return a > b;
    }
};

// Min heap using Greater
priority_queue<int, vector<int>, Greater<int>> pq;

// Works for any comparable type
priority_queue<double, vector<double>, Greater<double>> pq2;
\`\`\`

**4. Reusable Data Structure Code**

\`\`\`cpp
template<typename T>
class SegmentTree {
    vector<T> tree;
    int n;
    
public:
    SegmentTree(int size) : n(size) {
        tree.resize(4 * n);
    }
    
    void update(int pos, T value) { /* ... */ }
    T query(int l, int r) { /* ... */ }
};

// Works for different types
SegmentTree<int> st1(n);
SegmentTree<long long> st2(n);
\`\`\`

**When Explicit Types Are Better:**

**1. Simple Problem-Specific Code**

Don't template:
\`\`\`cpp
// Overkill:
template<typename T>
T solve(T n) {
    return n * (n + 1) / 2;
}

// Better:
long long solve(int n) {
    return (long long)n * (n + 1) / 2;
}
\`\`\`

**Reason:** Only used once, template adds complexity without benefit

**2. When Type Matters for Correctness**

\`\`\`cpp
// Template hides important type choice
template<typename T>
T computeSum(vector<T>& arr) {
    T sum = 0;
    for(T x : arr) sum += x;
    return sum;
}

// Explicit shows intent
long long computeSum(vector<int>& arr) {
    long long sum = 0;  // Prevent overflow
    for(int x : arr) sum += x;
    return sum;
}
\`\`\`

**3. When Debugging Is Priority**

Template error:
\`\`\`
error: no match for 'operator<' (operand types are 'std::pair<int, int>' and 'std::pair<int, int>')
note: candidate: template<class T> bool operator<(const T&, const T&)
note:   template argument deduction/substitution failed:
[100 more lines of template errors]
\`\`\`

Explicit error:
\`\`\`
error: no match for 'operator<' (operand types are 'Point' and 'Point')
\`\`\`

**Much clearer!**

**4. Performance-Critical Inner Loops**

\`\`\`cpp
// Template might not inline perfectly
template<typename T>
void process(vector<T>& v) {
    for(auto& x : v) {
        // Critical hot path
    }
}

// Explicit guarantees compiler knows exact type
void process(vector<int>& v) {
    for(int& x : v) {
        // Compiler can optimize better
    }
}
\`\`\`

**Hybrid Approach (Recommended):**

\`\`\`cpp
// Template library functions
template<typename T>
vector<T> readArray(int n) {
    vector<T> v(n);
    for(auto& x : v) cin >> x;
    return v;
}

template<typename T>
T gcd(T a, T b) {
    return b ? gcd(b, a % b) : a;
}

// Explicit problem solution
void solve() {
    int n;
    cin >> n;
    
    // Use templates for utilities
    auto arr = readArray<int>(n);
    
    // Explicit types for problem logic
    long long sum = 0;
    for(int x : arr) {
        sum += gcd(x, n);  // Template function
    }
    
    cout << sum << endl;
}
\`\`\`

**Practical Decision Matrix:**

| Situation | Use Template? | Reason |
|-----------|---------------|---------|
| Utility function used often | ✅ Yes | Reusable, DRY |
| Problem-specific logic | ❌ No | One-time use |
| Works with any container | ✅ Yes | Flexibility |
| Type affects correctness | ❌ No | Explicit is safer |
| Custom comparator | ✅ Yes | STL compatibility |
| Performance critical | ❌ No | Explicit optimizes better |
| Shared across problems | ✅ Yes | Build library |
| Complex algorithm | ❌ No | Debugging easier |

**Common Template Use Cases in CP:**

**Use Case 1: Reading Input**
\`\`\`cpp
template<typename T>
vector<T> read(int n) {
    vector<T> v(n);
    for(auto& x : v) cin >> x;
    return v;
}

auto arr = read<int>(n);
auto coords = read<long long>(m);
\`\`\`

**Use Case 2: Printing Output**
\`\`\`cpp
template<typename T>
void print(const vector<T>& v) {
    for(const auto& x : v) cout << x << " ";
    cout << "\\n";
}
\`\`\`

**Use Case 3: Math Functions**
\`\`\`cpp
template<typename T>
T gcd(T a, T b) { return b ? gcd(b, a%b) : a; }

template<typename T>
T lcm(T a, T b) { return a / gcd(a, b) * b; }
\`\`\`

**Use Case 4: Min/Max of Many Values**
\`\`\`cpp
template<typename T, typename... Args>
T minimum(T first, Args... args) {
    if constexpr(sizeof...(args) == 0) {
        return first;
    } else {
        return min(first, minimum(args...));
    }
}

cout << minimum(5, 3, 8, 1, 9) << endl;  // 1
\`\`\`

**Debugging Template Code:**

**Tip 1: Instantiate Explicitly During Debug**
\`\`\`cpp
// During debugging:
#ifdef LOCAL
// Explicit instantiation helps debugging
template int solve<int>(int);
template long long solve<long long>(long long);
#endif
\`\`\`

**Tip 2: Use -ftemplate-backtrace-limit**
\`\`\`bash
g++ -ftemplate-backtrace-limit=0 solution.cpp
# Shows full template error (helpful for finding root cause)
\`\`\`

**Tip 3: Simplify for Debugging**
\`\`\`cpp
// If template version has bugs
template<typename T>
T buggyFunction(T x) { /* ... */ }

// Debug with explicit version
int debugFunction(int x) { /* same logic */ }

// Once fixed, switch back to template
\`\`\`

**Performance Considerations:**

**Myth:** Templates are slow
**Reality:** Templates have zero runtime overhead!

\`\`\`cpp
// Template compiles to same code as explicit
template<typename T>
T add(T a, T b) { return a + b; }

int add_int(int a, int b) { return a + b; }

// Both compile to identical assembly
\`\`\`

**True cost:** Compilation time (templates generate code for each type)

**Common Mistakes:**

**Mistake 1: Over-templating**
\`\`\`cpp
// Too much:
template<typename T, typename U, typename V>
auto solve(T a, U b, V c) -> decltype(/* complex */) {
    // This is overkill for CP!
}

// Better:
long long solve(int a, int b, int c) {
    // Simple and clear
}
\`\`\`

**Mistake 2: Template When Simple Function Works**
\`\`\`cpp
// Unnecessary:
template<typename T>
T square(T x) { return x * x; }

// Just use:
int square(int x) { return x * x; }
// Or even:
#define SQUARE(x) ((x) * (x))
\`\`\`

**My Personal Template Library:**

\`\`\`cpp
// Minimal but useful
template<typename T>
vector<T> read(int n) {
    vector<T> v(n);
    for(auto& x : v) cin >> x;
    return v;
}

template<typename T>
void print(const vector<T>& v) {
    for(const auto& x : v) cout << x << " ";
    cout << "\\n";
}

template<typename T>
T gcd(T a, T b) { return b ? gcd(b, a%b) : a; }

template<typename T>
T power(T base, long long exp) {
    T result = 1;
    while(exp > 0) {
        if(exp & 1) result *= base;
        base *= base;
        exp >>= 1;
    }
    return result;
}
\`\`\`

**Bottom Line:**

**Use templates for:**
✅ Reusable utilities (read, print, gcd)
✅ Generic algorithms
✅ Custom comparators
✅ Building your CP library

**Use explicit types for:**
✅ Problem-specific logic
✅ One-time use code
✅ When type matters for correctness
✅ When debugging is difficult

**The sweet spot:** Templates for utilities, explicit for solutions. This gives you speed without sacrificing clarity!`,
    },
    {
      question:
        'Variadic templates and fold expressions (C++17) enable elegant solutions for certain problems. Provide examples of when these advanced features are useful in competitive programming.',
      answer: `Variadic templates and fold expressions are powerful but rarely essential in CP. Here's when they shine:

**Variadic Templates - Functions with Variable Arguments**

Basic syntax:
\`\`\`cpp
template<typename... Args>
void function(Args... args) {
    // Process variable number of arguments
}
\`\`\`

**Use Case 1: Min/Max of Multiple Values**

Without variadic:
\`\`\`cpp
int minimum2(int a, int b) { return min(a, b); }
int minimum3(int a, int b, int c) { return min(a, min(b, c)); }
int minimum4(int a, int b, int c, int d) { return min(min(a, b), min(c, d)); }
// Need different function for each count!
\`\`\`

With variadic:
\`\`\`cpp
template<typename T>
T minimum(T value) {
    return value;  // Base case
}

template<typename T, typename... Args>
T minimum(T first, Args... args) {
    return min(first, minimum(args...));  // Recursive
}

// Usage:
cout << minimum(5, 3, 8, 1, 9, 2) << endl;  // 1
cout << minimum(10, 20) << endl;            // 10
cout << minimum(42) << endl;                // 42
\`\`\`

**Use Case 2: Generic Debug Print**

\`\`\`cpp
#ifdef LOCAL
template<typename T>
void debug_print(const T& t) {
    cerr << t;
}

template<typename T, typename... Args>
void debug_print(const T& first, const Args&... args) {
    debug_print(first);
    cerr << ", ";
    debug_print(args...);
}

#define DEBUG(...) cerr << #__VA_ARGS__ << " = "; debug_print(__VA_ARGS__); cerr << endl
#else
#define DEBUG(...)
#endif

// Usage:
int x = 5, y = 10;
string s = "hello";
DEBUG(x, y, s);  // Prints: x, y, s = 5, 10, hello
\`\`\`

**Fold Expressions (C++17) - Simplify Variadic Operations**

Syntax:
\`\`\`cpp
(... op args)     // Left fold
(args op ...)     // Right fold
(init op ... op args)  // Left fold with init
\`\`\`

**Use Case 1: Sum of All Arguments**

Without fold:
\`\`\`cpp
template<typename T>
T sum(T value) {
    return value;
}

template<typename T, typename... Args>
T sum(T first, Args... args) {
    return first + sum(args...);
}
\`\`\`

With fold expression:
\`\`\`cpp
template<typename... Args>
auto sum(Args... args) {
    return (... + args);  // Left fold: ((a1 + a2) + a3) + ...
}

// Usage:
cout << sum(1, 2, 3, 4, 5) << endl;  // 15
cout << sum(1.5, 2.5, 3.0) << endl;  // 7.0
\`\`\`

**Use Case 2: Print Multiple Values**

\`\`\`cpp
template<typename... Args>
void print(Args... args) {
    (cout << ... << args) << endl;  // Left fold
}

// Usage:
print(1, 2, 3);  // Prints: 123
print("Hello", " ", "World");  // Prints: Hello World
\`\`\`

**Use Case 3: Check All Conditions**

\`\`\`cpp
template<typename... Args>
bool all_positive(Args... args) {
    return ((args > 0) && ...);  // Fold with &&
}

// Usage:
cout << all_positive(1, 2, 3) << endl;     // true
cout << all_positive(1, -2, 3) << endl;    // false
\`\`\`

**Use Case 4: Generic Function Application**

\`\`\`cpp
template<typename Func, typename... Args>
void apply_to_all(Func f, Args&&... args) {
    (f(std::forward<Args>(args)), ...);  // Apply f to each arg
}

// Usage:
apply_to_all([](int x) { cout << x << " "; }, 1, 2, 3, 4, 5);
// Prints: 1 2 3 4 5
\`\`\`

**Practical CP Examples:**

**Example 1: Update Multiple Variables**

\`\`\`cpp
template<typename... Args>
void maximize(Args&... args) {
    auto max_val = max({args...});
    ((args = max_val), ...);
}

int a = 5, b = 10, c = 3;
maximize(a, b, c);
// Now a = b = c = 10
\`\`\`

**Example 2: Hash Combination**

\`\`\`cpp
template<typename T>
size_t hash_combine(size_t seed, const T& v) {
    return seed ^ (std::hash<T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

template<typename... Args>
size_t hash_all(const Args&... args) {
    size_t seed = 0;
    ((seed = hash_combine(seed, args)), ...);
    return seed;
}

// Usage: Hash multiple values together
auto h = hash_all(x, y, z);
\`\`\`

**Example 3: Tuple-Like Operations**

\`\`\`cpp
template<typename... Args>
auto make_sorted_tuple(Args... args) {
    std::array arr = {args...};
    std::sort(arr.begin(), arr.end());
    return arr;
}

auto sorted = make_sorted_tuple(5, 2, 8, 1, 9);
// sorted = [1, 2, 5, 8, 9]
\`\`\`

**When NOT to Use:**

**1. Simple Cases**
\`\`\`cpp
// Overkill:
template<typename... Args>
auto product(Args... args) {
    return (args * ...);
}

// Just use:
int product(int a, int b, int c) {
    return a * b * c;
}
\`\`\`

**2. Performance-Critical Code**
\`\`\`cpp
// Recursive variadic can be slow
// Better: Use explicit loop or array
\`\`\`

**3. When Debugging is Priority**
\`\`\`cpp
// Template errors are cryptic
// Explicit functions easier to debug
\`\`\`

**Limitations in CP:**

1. **Compiler Support**: Need C++17 for fold expressions
   - Check if judge supports it
   - Fallback to C++11 variadic if needed

2. **Compilation Time**: More template instantiations = slower compile

3. **Error Messages**: Still cryptic, harder to debug

**Practical Template Library:**

\`\`\`cpp
// Minimal variadic utilities for CP

// Min/max of multiple values
template<typename T>
T minimum(T value) { return value; }

template<typename T, typename... Args>
T minimum(T first, Args... args) {
    return min(first, minimum(args...));
}

template<typename T>
T maximum(T value) { return value; }

template<typename T, typename... Args>
T maximum(T first, Args... args) {
    return max(first, maximum(args...));
}

// With C++17 fold (cleaner):
template<typename... Args>
auto minimum_fold(Args... args) {
    return std::min({args...});
}

// Debug print
#ifdef LOCAL
template<typename T>
void dbg(const T& t) { cerr << t; }

template<typename T, typename... Args>
void dbg(const T& first, const Args&... args) {
    dbg(first);
    cerr << ", ";
    dbg(args...);
}
#define DEBUG(...) cerr << #__VA_ARGS__ << " = "; dbg(__VA_ARGS__); cerr << endl
#else
#define DEBUG(...)
#endif
\`\`\`

**My Recommendation:**

**Learn it:** Yes, useful to know
**Use frequently:** No, rarely needed
**Include in template:** Maybe 1-2 functions

**Most useful:**
- \`minimum()/ maximum()\` for multiple values
- Generic debug print
- That's about it for CP

**Bottom Line:**

Variadic templates and fold expressions are:
- ✅ Elegant for generic utilities
- ✅ Useful for debug printing
- ❌ Overkill for most problems
- ❌ Not essential for competitive success

Learn them for your toolkit, but don't force them. Simple explicit code often better in contests!`,
    },
    {
      question:
        'Template specialization allows different implementations for specific types. Discuss when this is useful in competitive programming and provide examples where it significantly simplifies code.',
      answer: `Template specialization allows custom implementations for specific types. Here's when it's valuable in CP:

**What is Template Specialization?**

Generic template:
\`\`\`cpp
template<typename T>
struct Handler {
    void process(T value) {
        // Generic implementation
    }
};
\`\`\`

Specialized template:
\`\`\`cpp
template<>
struct Handler<int> {
    void process(int value) {
        // Special implementation for int
    }
};
\`\`\`

**Use Case 1: Type-Specific Optimization**

Problem: Hash function for pairs

Generic (doesn't work):
\`\`\`cpp
template<typename T>
struct Hash {
    size_t operator()(const T& value) const {
        return std::hash<T>{}(value);
    }
};
\`\`\`

Specialized for pairs:
\`\`\`cpp
template<typename T1, typename T2>
struct Hash<pair<T1, T2>> {
    size_t operator()(const pair<T1, T2>& p) const {
        size_t h1 = std::hash<T1>{}(p.first);
        size_t h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

// Usage:
unordered_map<pair<int, int>, int, Hash<pair<int, int>>> m;
\`\`\`

**Use Case 2: Custom Comparators**

Generic ascending:
\`\`\`cpp
template<typename T>
struct Compare {
    bool operator()(const T& a, const T& b) const {
        return a < b;
    }
};
\`\`\`

Specialized for pairs (compare by second, then first):
\`\`\`cpp
template<typename T1, typename T2>
struct Compare<pair<T1, T2>> {
    bool operator()(const pair<T1, T2>& a, const pair<T1, T2>& b) const {
        if(a.second != b.second) return a.second < b.second;
        return a.first < b.first;
    }
};

// Usage:
set<pair<int, int>, Compare<pair<int, int>>> s;
\`\`\`

**Use Case 3: Output Formatting**

Generic print:
\`\`\`cpp
template<typename T>
void print(const T& value) {
    cout << value;
}
\`\`\`

Specialized for vectors:
\`\`\`cpp
template<typename T>
void print(const vector<T>& v) {
    cout << "[";
    for(size_t i = 0; i < v.size(); i++) {
        if(i > 0) cout << ", ";
        print(v[i]);  // Recursive for nested vectors
    }
    cout << "]";
}
\`\`\`

Specialized for pairs:
\`\`\`cpp
template<typename T1, typename T2>
void print(const pair<T1, T2>& p) {
    cout << "(" << p.first << ", " << p.second << ")";
}
\`\`\`

Usage:
\`\`\`cpp
vector<int> v = {1, 2, 3};
print(v);  // [1, 2, 3]

pair<int, string> p = {1, "hello"};
print(p);  // (1, hello)

vector<pair<int, int>> vp = {{1,2}, {3,4}};
print(vp);  // [(1, 2), (3, 4)]
\`\`\`

**Use Case 4: Default Values**

Generic default:
\`\`\`cpp
template<typename T>
T getDefault() {
    return T();  // Default constructor
}
\`\`\`

Specialized for common types:
\`\`\`cpp
template<>
int getDefault<int>() { return 0; }

template<>
double getDefault<double>() { return 0.0; }

template<>
string getDefault<string>() { return ""; }
\`\`\`

**Use Case 5: Infinity Values**

Generic:
\`\`\`cpp
template<typename T>
T infinity() {
    return std::numeric_limits<T>::max();
}
\`\`\`

Specialized for specific needs:
\`\`\`cpp
template<>
int infinity<int>() { return 1e9; }  // More practical than INT_MAX

template<>
long long infinity<long long>() { return 1e18; }

template<>
double infinity<double>() { return 1e18; }  // Avoid overflow
\`\`\`

**Use Case 6: Segment Tree for Different Operations**

Generic template:
\`\`\`cpp
template<typename T, typename Operation>
class SegmentTree {
    vector<T> tree;
    T identity;
    Operation op;
    
public:
    SegmentTree(int n, T id, Operation operation) 
        : identity(id), op(operation) {
        tree.resize(4 * n, identity);
    }
    
    void update(int pos, T value) { /* ... */ }
    T query(int l, int r) { /* ... */ }
};
\`\`\`

Specialized constructors:
\`\`\`cpp
// Sum segment tree
auto sumTree = SegmentTree<int, plus<int>>(n, 0, plus<int>());

// Min segment tree  
auto minTree = SegmentTree<int, function<int(int,int)>>(
    n, INT_MAX, [](int a, int b) { return min(a, b); }
);

// GCD segment tree
auto gcdTree = SegmentTree<int, function<int(int,int)>>(
    n, 0, [](int a, int b) { return __gcd(a, b); }
);
\`\`\`

**Practical CP Example: Complete Debug System**

\`\`\`cpp
#ifdef LOCAL

// Base template
template<typename T>
void debug_print(const T& value) {
    cerr << value;
}

// Specialization for pairs
template<typename T1, typename T2>
void debug_print(const pair<T1, T2>& p) {
    cerr << "(";
    debug_print(p.first);
    cerr << ", ";
    debug_print(p.second);
    cerr << ")";
}

// Specialization for vectors
template<typename T>
void debug_print(const vector<T>& v) {
    cerr << "[";
    for(size_t i = 0; i < v.size(); i++) {
        if(i > 0) cerr << ", ";
        debug_print(v[i]);
    }
    cerr << "]";
}

// Specialization for sets
template<typename T>
void debug_print(const set<T>& s) {
    cerr << "{";
    bool first = true;
    for(const auto& x : s) {
        if(!first) cerr << ", ";
        debug_print(x);
        first = false;
    }
    cerr << "}";
}

// Specialization for maps
template<typename K, typename V>
void debug_print(const map<K, V>& m) {
    cerr << "{";
    bool first = true;
    for(const auto& [k, v] : m) {
        if(!first) cerr << ", ";
        debug_print(k);
        cerr << ": ";
        debug_print(v);
        first = false;
    }
    cerr << "}";
}

#define DEBUG(x) cerr << #x << " = "; debug_print(x); cerr << endl

#else
#define DEBUG(x)
#endif

// Usage:
vector<int> v = {1, 2, 3};
DEBUG(v);  // v = [1, 2, 3]

vector<pair<int, int>> vp = {{1,2}, {3,4}};
DEBUG(vp);  // vp = [(1, 2), (3, 4)]

map<int, vector<int>> m = {{1, {2,3}}, {4, {5,6}}};
DEBUG(m);  // m = {1: [2, 3], 4: [5, 6]}
\`\`\`

**When Specialization Simplifies Code:**

**Before (No Specialization):**
\`\`\`cpp
void printInt(int x) { cout << x; }
void printString(string s) { cout << s; }
void printPairIntInt(pair<int,int> p) { cout << p.first << " " << p.second; }
void printVectorInt(vector<int> v) { /* ... */ }
// Need separate function for each type!
\`\`\`

**After (With Specialization):**
\`\`\`cpp
template<typename T> void print(T value);  // Generic

template<> void print<int>(int x) { cout << x; }
template<> void print<string>(string s) { cout << s; }
template<> void print<pair<int,int>>(pair<int,int> p) { /* ... */ }
// One function name, multiple implementations
\`\`\`

**When NOT to Use:**

**1. Simple Cases:**
\`\`\`cpp
// Don't specialize for this:
template<typename T>
T square(T x) { return x * x; }

template<>
int square<int>(int x) { return x * x; }  // Same code!

// Just use one template
\`\`\`

**2. When Overloading Works:**
\`\`\`cpp
// Instead of specialization:
template<typename T>
void process(T value);

template<>
void process<int>(int value) { /* ... */ }

// Better: Function overloading
void process(int value) { /* ... */ }
void process(double value) { /* ... */ }
\`\`\`

**3. Over-Engineering:**
\`\`\`cpp
// Too much for CP:
template<typename T, int N>
struct Array;

template<typename T>
struct Array<T, 0> { /* special case */ };

template<typename T, int N>
struct Array { /* general case */ };

// Just use vector!
\`\`\`

**Limitations:**

1. **Must specialize all template parameters**
2. **Partial specialization not for functions**
3. **Can make code harder to navigate**
4. **Compilation time increases**

**My Recommendation:**

**Use specialization for:**
✅ Debug print system (big win!)
✅ Custom hash for unordered containers
✅ Type-specific optimizations

**Don't use for:**
❌ Simple generic functions
❌ When function overloading works
❌ Over-engineering solutions

**Practical Template with Specialization:**

\`\`\`cpp
// In your template file:

// Generic print
template<typename T>
void print(const T& value) {
    cout << value;
}

// Specialized for vector
template<typename T>
void print(const vector<T>& v) {
    for(size_t i = 0; i < v.size(); i++) {
        if(i > 0) cout << " ";
        print(v[i]);
    }
}

// Specialized for pair
template<typename T1, typename T2>
void print(const pair<T1, T2>& p) {
    print(p.first);
    cout << " ";
    print(p.second);
}

// Now you can print anything!
\`\`\`

**Bottom Line:**

Template specialization:
- ✅ Powerful for type-specific behavior
- ✅ Essential for debug systems
- ✅ Useful for custom hashing
- ❌ Often overkill in simple cases
- ❌ Adds complexity

Use it for your debug/utility library, but don't force it into problem solutions!`,
    },
  ],
} as const;
