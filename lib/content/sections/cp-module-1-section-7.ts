export const cppModernFeaturesSection = {
  id: 'cp-m1-s7',
  title: 'C++11/14/17/20 Features for CP',
  content: `

# C++11/14/17/20 Features for CP

## Introduction

Modern C++ (C++11 and later) introduced powerful features that make competitive programming code cleaner, faster to write, and less error-prone. While you don't need to know every feature, mastering the most useful ones will give you a significant speed advantage in contests.

In this section, we'll focus on the features that actually matter for competitive programming, with practical examples for each.

**Goal**: Learn modern C++ features that save time and prevent bugs in contests.

---

## Why Modern C++ Matters for CP

### Before C++11 (Old Style)

\`\`\`cpp
vector<int> v;
for (vector<int>::iterator it = v.begin(); it != v.end(); ++it) {
    cout << *it << endl;
}

pair<int, int> p;
p = make_pair(3, 4);
int x = p.first;
int y = p.second;
\`\`\`

**Problems:**
- Verbose iterator syntax
- Type declarations are repetitive
- More opportunities for typos

### After C++11 (Modern Style)

\`\`\`cpp
vector<int> v;
for (auto x : v) {
    cout << x << endl;
}

auto p = make_pair(3, 4);
auto [x, y] = p;  // C++17
\`\`\`

**Benefits:**
- Less typing
- Fewer errors
- More readable
- **Faster contest coding!**

---

## auto Keyword: Type Inference

The \`auto\` keyword lets the compiler deduce the type automatically.

### Basic Usage

\`\`\`cpp
auto x = 5;           // int
auto y = 3.14;        // double
auto s = "hello";     // const char*
auto str = string("hello");  // string

vector<int> v = {1, 2, 3};
auto it = v.begin();  // vector<int>::iterator
\`\`\`

### When to Use auto in CP

**✅ Use auto for:**

**1. Iterators**
\`\`\`cpp
// Instead of:
map<string, vector<int>>::iterator it = m.begin();

// Write:
auto it = m.begin();
\`\`\`

**2. Pair/Tuple returns**
\`\`\`cpp
auto p = make_pair(3, 4);
auto t = make_tuple(1, 2, 3);
\`\`\`

**3. STL algorithm returns**
\`\`\`cpp
auto it = lower_bound(v.begin(), v.end(), target);
auto minIt = min_element(v.begin(), v.end());
\`\`\`

**❌ Avoid auto when:**

**1. Type is important for understanding**
\`\`\`cpp
// Unclear:
auto result = calculateSomething();

// Clear:
int result = calculateSomething();
\`\`\`

**2. You need a specific type**
\`\`\`cpp
// Might be int, but you want long long
auto sum = 0;  // int
long long sum = 0;  // long long
\`\`\`

### auto with References

\`\`\`cpp
vector<int> v = {1, 2, 3, 4, 5};

// Copy (slow)
for (auto x : v) {
    x *= 2;  // Modifies copy, not original
}

// Reference (fast, can modify)
for (auto& x : v) {
    x *= 2;  // Modifies original
}

// Const reference (fast, read-only)
for (const auto& x : v) {
    cout << x << " ";
}
\`\`\`

**Rule:** Use \`const auto&\` for large objects (vectors, strings, pairs)

---

## Range-Based For Loops

The **range-based for loop** is one of the most useful C++11 features for competitive programming.

### Basic Syntax

\`\`\`cpp
vector<int> v = {1, 2, 3, 4, 5};

// Old way
for (int i = 0; i < v.size(); i++) {
    cout << v[i] << " ";
}

// Modern way
for (int x : v) {
    cout << x << " ";
}
\`\`\`

### With Different Containers

\`\`\`cpp
// Vector
vector<int> v = {1, 2, 3};
for (auto x : v) { }

// Array
int arr[] = {1, 2, 3, 4, 5};
for (int x : arr) { }

// Set
set<int> s = {3, 1, 4, 1, 5};
for (int x : s) { }  // Iterates in sorted order

// Map
map<string, int> m;
for (auto [key, value] : m) {  // C++17 structured binding
    cout << key << ": " << value << endl;
}
\`\`\`

### Modifying Elements

\`\`\`cpp
vector<int> v = {1, 2, 3, 4, 5};

// To modify elements, use reference
for (auto& x : v) {
    x *= 2;
}
// Now v = {2, 4, 6, 8, 10}
\`\`\`

### Reading Input with Range-Based For

\`\`\`cpp
int n;
cin >> n;
vector<int> a(n);

// Read n integers
for (auto& x : a) {
    cin >> x;
}
\`\`\`

**This is THE standard way to read input in modern CP!**

---

## Lambda Functions

Lambda functions are anonymous functions you can define inline. Super useful for custom comparators!

### Basic Syntax

\`\`\`cpp
// Traditional function
bool compare(int a, int b) {
    return a > b;
}

// Lambda equivalent
auto compare = [](int a, int b) {
    return a > b;
};

sort(v.begin(), v.end(), compare);
\`\`\`

### Inline Lambda

\`\`\`cpp
vector<int> v = {5, 2, 8, 1, 9};

// Sort in descending order
sort(v.begin(), v.end(), [](int a, int b) {
    return a > b;
});
\`\`\`

### Lambda for Custom Sorting

\`\`\`cpp
vector<pair<int, int>> points = {{3, 4}, {1, 2}, {5, 1}};

// Sort by sum of coordinates
sort(points.begin(), points.end(), [](auto a, auto b) {
    return a.first + a.second < b.first + b.second;
});

// Sort by distance from origin
sort(points.begin(), points.end(), [](auto a, auto b) {
    return a.first*a.first + a.second*a.second < 
           b.first*b.first + b.second*b.second;
});
\`\`\`

### Capturing Variables

\`\`\`cpp
int k = 5;

// Capture k by value
auto checkSum = [k](int a, int b) {
    return a + b == k;
};

// Capture k by reference
auto increment = [&k]() {
    k++;
};

// Capture all by value
auto lambda1 = [=]() { return k; };

// Capture all by reference
auto lambda2 [&]() { k++; };
\`\`\`

### Practical CP Examples

**Find if any element satisfies condition:**
\`\`\`cpp
vector<int> v = {1, 2, 3, 4, 5};

bool hasEven = any_of(v.begin(), v.end(), [](int x) {
    return x % 2 == 0;
});
\`\`\`

**Count elements satisfying condition:**
\`\`\`cpp
int evenCount = count_if(v.begin(), v.end(), [](int x) {
    return x % 2 == 0;
});
\`\`\`

**Custom priority queue:**
\`\`\`cpp
auto cmp = [](pair<int,int> a, pair<int,int> b) {
    return a.first + a.second < b.first + b.second;
};

priority_queue<pair<int,int>, vector<pair<int,int>>, decltype(cmp)> pq(cmp);
\`\`\`

---

## Structured Bindings (C++17)

**Structured bindings** let you unpack pairs and tuples elegantly.

### With Pairs

\`\`\`cpp
pair<int, int> p = {3, 4};

// Old way
int x = p.first;
int y = p.second;

// C++17 way
auto [x, y] = p;

cout << x << " " << y << endl;  // 3 4
\`\`\`

### With Maps

\`\`\`cpp
map<string, int> m = {{"alice", 100}, {"bob", 200}};

// Old way
for (auto it : m) {
    string key = it.first;
    int value = it.second;
    cout << key << ": " << value << endl;
}

// C++17 way
for (auto [key, value] : m) {
    cout << key << ": " << value << endl;
}
\`\`\`

**This makes map iteration SO much cleaner!**

### With Tuples

\`\`\`cpp
tuple<int, string, double> t = {1, "hello", 3.14};

// Unpack all at once
auto [id, name, score] = t;

cout << id << " " << name << " " << score << endl;
\`\`\`

### Real CP Example

\`\`\`cpp
vector<tuple<int, int, int>> edges;  // {u, v, weight}

for (auto [u, v, w] : edges) {
    // Process edge from u to v with weight w
    adj[u].push_back({v, w});
}
\`\`\`

### With References (to modify)

\`\`\`cpp
map<int, int> freq;
for (auto& [key, count] : freq) {
    count++;  // Modify count
}
\`\`\`

---

## Tuple Improvements

C++11 introduced tuples for storing multiple values.

### Creating Tuples

\`\`\`cpp
// Old way
tuple<int, string, double> t = make_tuple(1, "hello", 3.14);

// C++11 way (still common)
auto t = make_tuple(1, "hello", 3.14);

// C++17 way
tuple t = {1, "hello", 3.14};  // Type deduction
\`\`\`

### Accessing Elements

\`\`\`cpp
tuple<int, string, double> t = {1, "hello", 3.14};

// By index
int x = get<0>(t);        // 1
string s = get<1>(t);     // "hello"
double d = get<2>(t);     // 3.14

// C++17: structured binding
auto [x, s, d] = t;
\`\`\`

### Why Use Tuples in CP?

**Returning multiple values:**
\`\`\`cpp
tuple<int, int, int> solve() {
    return {answer1, answer2, answer3};
}

auto [a, b, c] = solve();
\`\`\`

**Sorting with multiple criteria:**
\`\`\`cpp
vector<tuple<int, int, int>> students;  // {score, age, id}

sort(students.begin(), students.end());
// Sorts by score first, then age, then id automatically!
\`\`\`

### Tie for Multiple Assignment

\`\`\`cpp
int a, b, c;
tie(a, b, c) = make_tuple(1, 2, 3);

// Ignore some values with std::ignore
int x, z;
tie(x, ignore, z) = make_tuple(1, 2, 3);
// x = 1, z = 3
\`\`\`

---

## constexpr: Compile-Time Computation

\`constexpr\` tells the compiler to evaluate expressions at compile time.

### Basic Usage

\`\`\`cpp
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

constexpr int f10 = factorial(10);  // Computed at compile time!
\`\`\`

### Practical CP Uses

**1. Computing powers of 2:**
\`\`\`cpp
constexpr long long pow2(int n) {
    return (n == 0) ? 1 : 2 * pow2(n - 1);
}

constexpr long long MAX = pow2(20);  // 1048576
\`\`\`

**2. Array sizes:**
\`\`\`cpp
constexpr int MAXN = 100000;
int arr[MAXN];
\`\`\`

**3. Compile-time constants:**
\`\`\`cpp
constexpr int MOD = 1e9 + 7;
constexpr double PI = 3.14159265358979323846;
\`\`\`

**Benefits:**
- No runtime overhead
- Computed once at compile time
- Can use in array sizes

---

## nullptr Instead of NULL

C++11 introduced \`nullptr\` as a proper null pointer constant.

### Old vs New

\`\`\`cpp
// Old (C-style)
int* p = NULL;

// New (C++)
int* p = nullptr;
\`\`\`

### Why It Matters

\`\`\`cpp
void foo(int x) { cout << "int" << endl; }
void foo(int* p) { cout << "pointer" << endl; }

foo(NULL);     // Calls foo(int) - ambiguous!
foo(nullptr);  // Calls foo(int*) - correct!
\`\`\`

**In CP:** Use \`nullptr\` for pointer initialization and comparisons.

\`\`\`cpp
TreeNode* root = nullptr;

if (root == nullptr) {
    // ...
}
\`\`\`

---

## Initializer Lists

C++11 makes initialization much cleaner.

### Vector Initialization

\`\`\`cpp
// Old way
vector<int> v;
v.push_back(1);
v.push_back(2);
v.push_back(3);

// C++11 way
vector<int> v = {1, 2, 3, 4, 5};

// Or
vector<int> v{1, 2, 3, 4, 5};
\`\`\`

### Map Initialization

\`\`\`cpp
map<string, int> m = {
    {"alice", 100},
    {"bob", 200},
    {"charlie", 150}
};
\`\`\`

### Set Initialization

\`\`\`cpp
set<int> s = {3, 1, 4, 1, 5, 9, 2, 6};
// Automatically sorted and duplicates removed
\`\`\`

### Pair and Tuple

\`\`\`cpp
pair<int, int> p = {3, 4};

tuple<int, string, double> t = {1, "hello", 3.14};

vector<pair<int, int>> points = {
    {1, 2},
    {3, 4},
    {5, 6}
};
\`\`\`

---

## emplace vs push

C++11 introduced \`emplace\` methods for more efficient insertion.

### emplace_back vs push_back

\`\`\`cpp
vector<pair<int, int>> v;

// push_back: creates temporary, then copies
v.push_back(make_pair(3, 4));
v.push_back({3, 4});

// emplace_back: constructs in-place (faster)
v.emplace_back(3, 4);
\`\`\`

### Why emplace is Better

\`\`\`cpp
vector<string> v;

// push_back: creates temporary string, then copies
v.push_back(string("hello"));

// emplace_back: constructs string directly
v.emplace_back("hello");
\`\`\`

### For Simple Types: No Difference

\`\`\`cpp
vector<int> v;
v.push_back(5);    // Fine
v.emplace_back(5); // Also fine, same efficiency
\`\`\`

**In CP:** Use \`emplace_back\` as a habit for complex types; either is fine for simple types.

---

## Move Semantics (Advanced)

C++11 introduced move semantics for efficient resource transfer.

### Basic Concept

\`\`\`cpp
vector<int> createLargeVector() {
    vector<int> v(1000000, 1);
    return v;  // Move, not copy! (C++11)
}

vector<int> data = createLargeVector();  // Efficient!
\`\`\`

**Before C++11:** Would copy entire vector (slow)
**C++11 and later:** Moves vector ownership (fast)

### std::move

\`\`\`cpp
vector<int> v1 = {1, 2, 3, 4, 5};
vector<int> v2 = move(v1);  // v1 is now empty, v2 has data

// v1 should not be used after this!
\`\`\`

**In CP:** Usually automatic, but good to know it exists.

---

## Using Declarations

C++11 improved type aliases.

### Old typedef vs New using

\`\`\`cpp
// Old way
typedef long long ll;
typedef pair<int, int> pii;
typedef vector<int> vi;

// New way (clearer syntax)
using ll = long long;
using pii = pair<int, int>;
using vi = vector<int>;
\`\`\`

### Template Aliases

\`\`\`cpp
// Can't do this with typedef:
template<typename T>
using Vec = vector<T>;

Vec<int> v1;
Vec<string> v2;
\`\`\`

**In CP:** Both work, but \`using\` is more modern and flexible.

---

## Useful C++14/17/20 Features

### C++14: Binary Literals

\`\`\`cpp
int mask = 0b1010;  // 10 in decimal
int bits = 0b11111111;  // 255
\`\`\`

### C++14: Digit Separators

\`\`\`cpp
int million = 1'000'000;
long long big = 1'000'000'000'000LL;
\`\`\`

**Makes large numbers more readable!**

### C++17: if with Initializer

\`\`\`cpp
// Old way
auto it = m.find(key);
if (it != m.end()) {
    cout << it->second << endl;
}

// C++17 way
if (auto it = m.find(key); it != m.end()) {
    cout << it->second << endl;
}
// it is scoped to the if block
\`\`\`

### C++20: Ranges (Preview)

\`\`\`cpp
#include <ranges>

vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

// Filter even numbers and square them
auto result = v | views::filter([](int x) { return x % 2 == 0; })
                | views::transform([](int x) { return x * x; });

for (int x : result) {
    cout << x << " ";  // 4 16 36 64 100
}
\`\`\`

**Note:** Ranges are powerful but not widely supported yet in online judges.

---

## Practical Examples for CP

### Example 1: Reading and Sorting

\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int n;
    cin >> n;
    
    vector<int> a(n);
    for (auto& x : a) cin >> x;
    
    sort(a.begin(), a.end());
    
    for (auto x : a) cout << x << " ";
    cout << "\\n";
    
    return 0;
}
\`\`\`

### Example 2: Custom Sorting with Lambda

\`\`\`cpp
vector<pair<int, int>> intervals;
int n;
cin >> n;

for (int i = 0; i < n; i++) {
    int l, r;
    cin >> l >> r;
    intervals.emplace_back(l, r);
}

// Sort by left endpoint, then by right endpoint
sort(intervals.begin(), intervals.end(), [](auto a, auto b) {
    if (a.first != b.first) return a.first < b.first;
    return a.second < b.second;
});

for (auto [l, r] : intervals) {
    cout << l << " " << r << "\\n";
}
\`\`\`

### Example 3: Frequency Map

\`\`\`cpp
vector<int> arr = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4};

map<int, int> freq;
for (auto x : arr) freq[x]++;

for (auto [val, count] : freq) {
    cout << val << " appears " << count << " times\\n";
}
\`\`\`

---

## What to Actually Use in Contests

### Always Use:
✅ \`auto\` for iterators and obvious types
✅ Range-based for loops
✅ Structured bindings (C++17)
✅ Lambdas for sorting
✅ \`emplace_back\` for complex types
✅ Initializer lists
✅ \`nullptr\`

### Sometimes Use:
⚠️ \`constexpr\` for compile-time constants
⚠️ Tuples for multiple return values
⚠️ \`std::move\` in specific situations

### Rarely Needed:
❌ Advanced template metaprogramming
❌ C++20 features (not widely supported)
❌ Complex move semantics

---

## Summary

**Most Important Modern C++ Features for CP:**

1. **auto** - Type inference
2. **Range-based for** - Iterate containers easily
3. **Lambdas** - Custom comparators inline
4. **Structured bindings** - Unpack pairs/tuples (C++17)
5. **Initializer lists** - Clean initialization
6. **emplace_back** - Efficient insertion

**Quick Template with Modern Features:**

\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int n;
    cin >> n;
    
    vector<pair<int, int>> v(n);
    for (auto& [x, y] : v) cin >> x >> y;
    
    sort(v.begin(), v.end(), [](auto a, auto b) {
        return a.first < b.first;
    });
    
    for (const auto& [x, y] : v) {
        cout << x << " " << y << "\\n";
    }
    
    return 0;
}
\`\`\`

**This uses: auto, range-based for, structured bindings, lambdas, emplace_back - all the essential modern features!**

---

## Next Steps

Now that you know modern C++ features, let's learn about **Macros & Preprocessor Tricks** that can speed up your coding even more!

**Key Takeaway**: Modern C++ features make code shorter, clearer, and faster to write - perfect for competitive programming where every second counts!
`,
  quizId: 'cp-m1-s7-quiz',
  discussionId: 'cp-m1-s7-discussion',
} as const;
