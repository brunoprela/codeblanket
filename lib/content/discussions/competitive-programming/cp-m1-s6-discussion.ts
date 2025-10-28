export default {
    id: 'cp-m1-s6-discussion',
    title: 'C++ Basics Review - Discussion Questions',
    questions: [
        {
            question: 'Many competitive programmers use both arrays and vectors. Explain when you would choose a C-style array over a vector, and vice versa. What are the performance implications of each choice?',
            answer: `The array vs vector choice affects both performance and code quality. Here's the complete analysis:

**C-Style Arrays:**

Declaration:
\`\`\`cpp
int arr[100000];           // Stack (if local, dangerous if too large!)
int global_arr[100000];    // Data segment (safe, any size)
\`\`\`

**Advantages:**
✅ Slightly faster (no bounds checking overhead)
✅ No indirection (direct memory access)
✅ Compile-time size known
✅ Can be global easily
✅ Zero initialization for global arrays
✅ Familiar C syntax

**Disadvantages:**
❌ Fixed size at compile time
❌ No bounds checking (segfault on overflow)
❌ Can't be returned from function
❌ Stack overflow if local and large
❌ No size() method
❌ Harder to pass to functions
❌ Can't resize

**Vectors:**

Declaration:
\`\`\`cpp
vector<int> v(100000);  // Heap allocation
vector<int> v;          // Empty, will grow
\`\`\`

**Advantages:**
✅ Dynamic size (can grow/shrink)
✅ Bounds checking with .at()
✅ Can be returned from function
✅ Size known (.size())
✅ Easy to pass to functions
✅ Many STL algorithms work
✅ RAII (automatic cleanup)
✅ Can use push_back, pop_back, insert, erase

**Disadvantages:**
❌ Slightly slower (function call overhead)
❌ Heap allocation (slower than stack)
❌ Indirection (pointer to data)
❌ Initialization might be slower

**Performance Comparison:**

**Access Speed:**
\`\`\`cpp
// Array: ~same speed
int arr[1000000];
for(int i = 0; i < 1000000; i++) {
    arr[i] = i;  // Direct memory write
}

// Vector: ~same speed with optimization
vector<int> v(1000000);
for(int i = 0; i < 1000000; i++) {
    v[i] = i;  // Also direct with [] operator
}
\`\`\`

With -O2 optimization: **No significant difference!**

**Initialization:**
\`\`\`cpp
// Array: Instant for global (BSS segment)
int global_arr[1000000];  // Zero-initialized at load time

// Vector: Must allocate and initialize
vector<int> v(1000000);  // Allocates + zeros out memory
\`\`\`

Vector initialization is slower (~10ms for 10^6 elements).

**When to Use Arrays:**

1. **Global/Static Arrays:**
\`\`\`cpp
int dp[1000][1000];  // Global DP table
int visited[100000]; // Global visited array

int main() {
    // All zeros by default
    // No initialization overhead
}
\`\`\`

2. **Small Fixed-Size Data:**
\`\`\`cpp
int dx[] = {0, 1, 0, -1};  // Direction vectors
int dy[] = {1, 0, -1, 0};
\`\`\`

3. **Performance-Critical Inner Loops:**
\`\`\`cpp
// If every nanosecond counts
int temp[100];  // Local, small, fast
for(int i = 0; i < n; i++) {
    // Process using temp
}
\`\`\`

4. **When Size is Known at Compile Time:**
\`\`\`cpp
const int MAXN = 100000;
int arr[MAXN];
\`\`\`

**When to Use Vectors:**

1. **Dynamic Size:**
\`\`\`cpp
int n;
cin >> n;
vector<int> arr(n);  // Size from input
\`\`\`

2. **Need to Resize:**
\`\`\`cpp
vector<int> v;
while(cin >> x) {
    v.push_back(x);  // Grows as needed
}
\`\`\`

3. **Need to Return from Function:**
\`\`\`cpp
vector<int> getData() {
    vector<int> result(n);
    // Fill result...
    return result;  // Can return vector
}
\`\`\`

4. **Working with STL:**
\`\`\`cpp
vector<int> v = {5, 2, 8, 1, 9};
sort(v.begin(), v.end());  // STL algorithms
\`\`\`

5. **2D Dynamic Arrays:**
\`\`\`cpp
// Can't do this with arrays easily:
int n, m;
cin >> n >> m;
vector<vector<int>> grid(n, vector<int>(m));
\`\`\`

6. **Default Choice:**
When in doubt, use vector - safer and more flexible.

**Hybrid Approach:**

\`\`\`cpp
const int MAXN = 100005;
int arr[MAXN];  // Global array for max size

int main() {
    int n;
    cin >> n;
    // Use first n elements of arr
    for(int i = 0; i < n; i++) {
        cin >> arr[i];
    }
}
\`\`\`

**Common Mistakes:**

**Mistake 1: Large Local Array**
\`\`\`cpp
int main() {
    int arr[1000000];  // Stack overflow!
}
\`\`\`

Fix: Use global array or vector

**Mistake 2: Returning Array**
\`\`\`cpp
int* getArray() {
    int arr[100];
    return arr;  // Undefined behavior!
}
\`\`\`

Fix: Use vector or dynamic allocation

**Mistake 3: Array Size from Input**
\`\`\`cpp
int n;
cin >> n;
int arr[n];  // Not standard C++ (VLA)
\`\`\`

Fix: Use vector or global array with MAXN

**Myth-Busting:**

**Myth: Vectors are much slower**
Reality: With -O2, negligible difference in most cases

**Myth: Always use vectors**
Reality: Global arrays have their place

**Myth: Arrays are faster for everything**
Reality: Vectors provide more features with minimal cost

**Recommendation by Scenario:**

**Beginner:** Use vectors everywhere
- Safer (bounds checking with .at())
- More forgiving
- Learn one tool well

**Intermediate:** Mix based on needs
- Vectors for dynamic size
- Global arrays for fixed DP tables
- Whatever is cleaner

**Advanced:** Optimize when needed
- Profile to find bottlenecks
- Use arrays only where proven necessary
- Default to vectors unless proven slow

**Bottom Line:**

**Use vectors when:**
- Size not known at compile time
- Need dynamic resizing
- Working with STL
- Want safety

**Use arrays when:**
- Global fixed-size data
- Maximum performance critical
- Size known at compile time

**In practice:** Both work fine! Choose based on convenience, not premature optimization.`,
        },
        {
            question: 'Explain the differences between auto, decltype, and explicit type declarations. When should you use each in competitive programming?',
            answer: `Modern C++ type features can simplify code significantly. Here's when and how to use each:

**auto - Type Deduction from Initializer:**

\`\`\`cpp
auto x = 42;              // int
auto y = 3.14;            // double
auto s = string("hello"); // string
auto v = vector<int>(10); // vector<int>
\`\`\`

**How it works:**
Compiler deduces type from right-hand side.

**Advantages:**
✅ Saves typing long type names
✅ Easier to refactor
✅ Cleaner code with complex types
✅ Reduces errors from type mismatches

**When to use in CP:**

1. **Iterator Types:**
\`\`\`cpp
// Old way:
vector<int>::iterator it = v.begin();
map<string, int>::iterator mit = m.begin();

// New way:
auto it = v.begin();
auto mit = m.begin();
\`\`\`

2. **Complex STL Types:**
\`\`\`cpp
// Old:
vector<pair<int, int>> vp = {{1,2}, {3,4}};

// New:
auto vp = vector<pair<int, int>>{{1,2}, {3,4}};
\`\`\`

3. **Range-Based For:**
\`\`\`cpp
vector<int> v = {1, 2, 3};
for(auto x : v) {  // Copy
    cout << x << " ";
}

for(auto& x : v) {  // Reference (modify)
    x *= 2;
}

for(const auto& x : v) {  // Const reference (read-only, efficient)
    cout << x << " ";
}
\`\`\`

4. **Function Return Types (when obvious):**
\`\`\`cpp
auto result = find(v.begin(), v.end(), 5);  // iterator type
auto m = max(a, b);  // type of a/b
\`\`\`

**When NOT to use auto:**

1. **Unclear Types:**
\`\`\`cpp
auto x = someFunction();  // What type is x?
\`\`\`

2. **Integer Literals (when size matters):**
\`\`\`cpp
auto x = 1000000;  // int (32-bit)
// But if you need long long:
long long x = 1000000;  // Explicit
auto x = 1000000LL;     // Or use LL suffix
\`\`\`

3. **When Readability Suffers:**
\`\`\`cpp
auto result = calculateSomething();  // Not obvious what type
int result = calculateSomething();   // Clear it returns int
\`\`\`

**decltype - Type of Expression:**

\`\`\`cpp
int x = 5;
decltype(x) y = 10;  // y is int (type of x)

vector<int> v;
decltype(v) v2;  // v2 is vector<int>
\`\`\`

**How it works:**
Gets type of an expression without evaluating it.

**When to use in CP:**

1. **Template Programming:**
\`\`\`cpp
template<typename T, typename U>
auto add(T a, U b) -> decltype(a + b) {
    return a + b;
}
\`\`\`

2. **Matching Types:**
\`\`\`cpp
vector<int> v;
decltype(v)::value_type x;  // x is int
\`\`\`

3. **When auto isn't enough:**
\`\`\`cpp
int x = 5;
auto y = x;          // y is int
decltype(x) z = x;   // z is int
decltype((x)) w = x; // w is int& (reference!)
\`\`\`

**When NOT to use decltype (in CP):**

Rarely needed! Use auto instead for simplicity.

**Explicit Type Declarations:**

\`\`\`cpp
int x = 42;
double y = 3.14;
string s = "hello";
vector<int> v;
\`\`\`

**When to use:**

1. **When Type Matters:**
\`\`\`cpp
// Force long long for large numbers:
long long x = 1000000;  // Might overflow as int later

// vs
auto x = 1000000;  // int (dangerous!)
\`\`\`

2. **Zero-Initialization:**
\`\`\`cpp
int sum = 0;  // Clear intent
auto sum = 0; // Less clear
\`\`\`

3. **Documentation:**
\`\`\`cpp
int numberOfNodes = 100;  // Clear
auto numberOfNodes = 100; // Less clear
\`\`\`

4. **Primitive Types in Simple Code:**
\`\`\`cpp
int n, m, k;  // Standard
cin >> n >> m >> k;
\`\`\`

**Comparison Table:**

| Feature | auto | decltype | Explicit |
|---------|------|----------|----------|
| Saves typing | ✅ | ❌ | ❌ |
| Type clear | ❌ | ❌ | ✅ |
| Refactoring easy | ✅ | ✅ | ❌ |
| Beginner-friendly | ✅ | ❌ | ✅ |
| Control | ❌ | ✅ | ✅ |

**Practical Examples:**

**Example 1: Reading pairs**
\`\`\`cpp
// Old way:
vector<pair<int, int>> pairs;
for(int i = 0; i < n; i++) {
    pair<int, int> p;
    cin >> p.first >> p.second;
    pairs.push_back(p);
}

// Modern way:
vector<pair<int, int>> pairs;
for(int i = 0; i < n; i++) {
    auto p = make_pair(0, 0);
    cin >> p.first >> p.second;
    pairs.push_back(p);
}

// Even better (C++17):
for(auto& [a, b] : pairs) {
    cin >> a >> b;
}
\`\`\`

**Example 2: Map iteration**
\`\`\`cpp
map<string, int> m;

// Old:
for(map<string, int>::iterator it = m.begin(); it != m.end(); it++) {
    cout << it->first << ": " << it->second << endl;
}

// Modern:
for(auto it = m.begin(); it != m.end(); it++) {
    cout << it->first << ": " << it->second << endl;
}

// Even better:
for(const auto& [key, value] : m) {  // C++17
    cout << key << ": " << value << endl;
}
\`\`\`

**Example 3: Binary search**
\`\`\`cpp
vector<int> v = {1, 3, 5, 7, 9};

// Old:
vector<int>::iterator it = lower_bound(v.begin(), v.end(), 5);

// Modern:
auto it = lower_bound(v.begin(), v.end(), 5);

// Both work the same!
\`\`\`

**Common Pitfalls:**

**Pitfall 1: auto with initializer lists**
\`\`\`cpp
auto v = {1, 2, 3};  // v is initializer_list<int>, NOT vector!

// Fix:
vector<int> v = {1, 2, 3};  // Explicit
\`\`\`

**Pitfall 2: auto removes const and references**
\`\`\`cpp
const int& r = x;
auto y = r;  // y is int, not const int&!

// Fix:
auto& y = r;        // Reference
const auto& y = r;  // Const reference
\`\`\`

**Pitfall 3: auto with integer literals**
\`\`\`cpp
auto x = 1e18;  // x is double!
// Should be:
long long x = 1e18;  // Explicit
\`\`\`

**Recommendations by Level:**

**Beginner:**
- Use explicit types
- Learn what types you're using
- Use auto only for iterators

**Intermediate:**
- Use auto for iterators and complex types
- Explicit for primitives (int, long long, double)
- Balance readability and convenience

**Advanced:**
- Use auto liberally where type is obvious
- Explicit when type affects correctness (long long vs int)
- Use const auto& in range-for for efficiency

**My Template Style:**

\`\`\`cpp
int main() {
    // Primitives: explicit
    int n, m;
    long long sum = 0;
    
    // Containers: explicit declaration, auto for iteration
    vector<int> v(n);
    for(auto& x : v) cin >> x;
    
    // STL algorithms: auto
    auto it = find(v.begin(), v.end(), target);
    auto maxElem = max_element(v.begin(), v.end());
    
    // Complex types: auto
    auto pairs = vector<pair<int, int>>(n);
}
\`\`\`

**Bottom Line:**

**Use auto:**
- Iterators
- Complex types
- Range-based for loops
- When type is obvious from context

**Use explicit types:**
- Primitives (int, long long, double)
- When type affects correctness
- When clarity improves readability

**Use decltype:**
- Rarely in CP (mostly for templates)

The goal: Balance convenience with clarity!`,
        },
        {
            question: 'Pointers and references are fundamental C++ concepts. Explain when you would use each in competitive programming, and describe common bugs that arise from misusing them.',
            answer: `Pointers and references are powerful but can cause subtle bugs. Here's the complete picture:

**References - Alias to Existing Variable:**

\`\`\`cpp
int x = 5;
int& ref = x;  // ref is another name for x
ref = 10;      // Now x is 10
\`\`\`

**Key Properties:**
- Must be initialized
- Can't be null
- Can't be reassigned
- No overhead (same as original variable)
- Safer than pointers

**When to Use References in CP:**

1. **Passing to Functions (Avoid Copies):**
\`\`\`cpp
// BAD: Copies entire vector
void process(vector<int> v) {
    // ...
}

// GOOD: Passes reference (no copy)
void process(const vector<int>& v) {
    // ...
}

// If you need to modify:
void modify(vector<int>& v) {
    v[0] = 100;
}
\`\`\`

2. **Range-Based For (Modifying Elements):**
\`\`\`cpp
vector<int> v = {1, 2, 3};

// Copy (inefficient):
for(auto x : v) {
    x *= 2;  // Doesn't modify v!
}

// Reference (efficient):
for(auto& x : v) {
    x *= 2;  // Modifies v!
}

// Const reference (read-only, efficient):
for(const auto& x : v) {
    cout << x << " ";
}
\`\`\`

3. **Avoiding Copies of Large Objects:**
\`\`\`cpp
map<string, vector<int>> m;

// BAD: Copies entire vector
vector<int> v = m["key"];

// GOOD: References original
const auto& v = m["key"];
\`\`\`

4. **Structured Bindings (C++17):**
\`\`\`cpp
map<string, int> m = {{"Alice", 25}, {"Bob", 30}};

// Reference to avoid copying
for(const auto& [name, age] : m) {
    cout << name << ": " << age << endl;
}
\`\`\`

**Pointers - Address of Variable:**

\`\`\`cpp
int x = 5;
int* ptr = &x;  // ptr holds address of x
*ptr = 10;      // Dereference and modify (x is now 10)
\`\`\`

**Key Properties:**
- Can be null
- Can be reassigned
- Can do pointer arithmetic
- More flexible but more dangerous
- Overhead (stores address)

**When to Use Pointers in CP:**

1. **Dynamic Memory:**
\`\`\`cpp
int* arr = new int[n];  // Dynamic array
// ... use arr
delete[] arr;  // Must free!

// But usually better to use vector!
vector<int> v(n);  // Automatic memory management
\`\`\`

2. **Tree/Graph Nodes:**
\`\`\`cpp
struct Node {
    int value;
    Node* left;
    Node* right;
    
    Node(int v) : value(v), left(nullptr), right(nullptr) {}
};

// Usage:
Node* root = new Node(5);
root->left = new Node(3);
root->right = new Node(7);
\`\`\`

3. **Linked Lists:**
\`\`\`cpp
struct ListNode {
    int val;
    ListNode* next;
    
    ListNode(int x) : val(x), next(nullptr) {}
};
\`\`\`

4. **Optional Values (C++11 better: std::optional):**
\`\`\`cpp
int* findElement(vector<int>& v, int target) {
    for(auto& x : v) {
        if(x == target) return &x;
    }
    return nullptr;  // Not found
}

// Usage:
int* result = findElement(v, 5);
if(result) {
    cout << "Found: " << *result << endl;
}
\`\`\`

**References vs Pointers - When to Choose:**

**Use References when:**
- Passing parameters to functions
- Avoiding copies
- Aliasing existing variables
- You don't need null
- Safety is priority

**Use Pointers when:**
- Dynamic memory allocation
- Need null as a valid value
- Building linked structures (trees, graphs, lists)
- Need to reassign
- Pointer arithmetic needed

**Common Bugs:**

**Bug 1: Dangling Reference**
\`\`\`cpp
int& getRef() {
    int x = 5;
    return x;  // BUG: Returns reference to local variable
}

int main() {
    int& r = getRef();
    cout << r;  // Undefined behavior!
}
\`\`\`

Fix: Return by value or use static/global

**Bug 2: Null Pointer Dereference**
\`\`\`cpp
int* ptr = nullptr;
*ptr = 5;  // Segmentation fault!
\`\`\`

Fix: Always check before dereferencing
\`\`\`cpp
if(ptr) {
    *ptr = 5;
}
\`\`\`

**Bug 3: Memory Leak**
\`\`\`cpp
void function() {
    int* arr = new int[1000000];
    // ... use arr
    // Forgot to delete[]!
}  // Memory leaked
\`\`\`

Fix: Use RAII (vector) or delete[] when done

**Bug 4: Double Delete**
\`\`\`cpp
int* ptr = new int(5);
delete ptr;
delete ptr;  // BUG: Double delete!
\`\`\`

Fix: Set to nullptr after delete
\`\`\`cpp
int* ptr = new int(5);
delete ptr;
ptr = nullptr;  // Safe to delete again
\`\`\`

**Bug 5: Using Dangling Pointer**
\`\`\`cpp
int* ptr = new int(5);
delete ptr;
cout << *ptr;  // Undefined behavior!
\`\`\`

Fix: Don't use after delete

**Bug 6: Reference to Temporary**
\`\`\`cpp
const int& r = someFunction();  // If function returns by value
// r might be dangling after this line!
\`\`\`

Fix: Capture by value if unsure

**Bug 7: Modifying Through Const Reference**
\`\`\`cpp
void process(const vector<int>& v) {
    v[0] = 100;  // Compilation error! Can't modify
}
\`\`\`

Remove const if you need to modify

**Best Practices:**

**1. Prefer References for Parameters:**
\`\`\`cpp
// Good (no copy, can't be null)
void process(const vector<int>& v) { }

// Bad (copies entire vector)
void process(vector<int> v) { }
\`\`\`

**2. Use const When You Don't Modify:**
\`\`\`cpp
void readOnly(const vector<int>& v) {
    // Can't accidentally modify v
}
\`\`\`

**3. Check Pointers Before Use:**
\`\`\`cpp
if(ptr) {
    // Safe to use ptr
}
\`\`\`

**4. Use Smart Pointers (Not Common in CP):**
\`\`\`cpp
unique_ptr<int> ptr = make_unique<int>(5);
// Automatic deletion, no memory leak
\`\`\`

**5. Avoid new/delete When Possible:**
\`\`\`cpp
// Instead of:
int* arr = new int[n];
// Use:
vector<int> arr(n);
\`\`\`

**In Competitive Programming:**

**90% of the time:**
- Use references for function parameters
- Use vectors instead of pointers
- Avoid manual memory management

**10% of the time:**
- Pointers for trees/graphs
- Pointers for linked structures
- When explicitly needed by problem

**Quick Decision Guide:**

\`\`\`
Need to avoid copying large object?
→ Use const reference

Need to modify parameter?
→ Use non-const reference

Need nullable value?
→ Use pointer

Building tree/ graph ?
→ Use pointers for nodes

Everything else?
→ Use values or vectors!
    \`\`\`

**Memory Management Hierarchy (Best to Worst for CP):**

1. **Local variables** (automatic)
2. **std::vector** (RAII, automatic)
3. **References** (no ownership, safe)
4. **Pointers** (manual, error-prone)
5. **new/delete** (avoid in CP!)

**Bottom Line:**

- **References**: Use for parameters (const if read-only)
- **Pointers**: Only when building dynamic structures
- **Both**: Check for validity, understand lifetime
- **Best**: Use vectors and avoid manual memory management!`,
    },
  ],
} as const ;

