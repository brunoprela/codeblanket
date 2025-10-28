export default {
    id: 'cp-m1-s7-discussion',
    title: 'C++11/14/17/20 Features for CP - Discussion Questions',
    questions: [
        {
            question: 'Modern C++ features like lambda functions, auto, and range-based for loops can significantly speed up code writing. Explain how to use these features effectively in competitive programming and when you might avoid them.',
            answer: `Modern C++ features are game-changers for competitive programming. Here's the comprehensive guide:

**Lambda Functions - Anonymous Functions:**

Basic syntax:
\`\`\`cpp
[capture](parameters) -> return_type { body }
\`\`\`

**Common Uses in CP:**

1. **Custom Sorting:**
\`\`\`cpp
vector<int> v = {5, 2, 8, 1, 9};

// Sort descending
sort(v.begin(), v.end(), [](int a, int b) {
    return a > b;
});

// Sort pairs by second element
vector<pair<int, int>> pairs = {{1,5}, {2,3}, {3,7}};
sort(pairs.begin(), pairs.end(), [](auto& a, auto& b) {
    return a.second < b.second;
});

// Complex sorting logic inline
sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
    if(a.priority != b.priority) return a.priority > b.priority;
    return a.timestamp < b.timestamp;
});
\`\`\`

2. **Custom Comparators for Priority Queue:**
\`\`\`cpp
// Min heap with custom comparison
auto cmp = [](int a, int b) { return a > b; };
priority_queue<int, vector<int>, decltype(cmp)> pq(cmp);
\`\`\`

3. **Inline Logic:**
\`\`\`cpp
// Check if all elements satisfy condition
bool allPositive = all_of(v.begin(), v.end(), [](int x) {
    return x > 0;
});

// Count elements matching condition
int evenCount = count_if(v.begin(), v.end(), [](int x) {
    return x % 2 == 0;
});

// Transform elements
transform(v.begin(), v.end(), v.begin(), [](int x) {
    return x * x;  // Square all elements
});
\`\`\`

4. **Capturing Variables:**
\`\`\`cpp
int threshold = 10;
int k = 5;

// Capture by value
auto greaterThanThreshold = [threshold](int x) {
    return x > threshold;
};

// Capture by reference
auto incrementByK = [&k](int x) {
    return x + k++;  // k changes!
};

// Capture all by value
auto lambda = [=](int x) {
    return x + threshold + k;
};

// Capture all by reference
auto lambda2 = [&](int x) {
    threshold++;  // Modifies original
    return x + threshold;
};
\`\`\`

**When to use lambdas:**
✅ Custom sorting (very common)
✅ One-time comparison logic
✅ STL algorithm predicates
✅ Keeping logic inline and readable

**When to avoid lambdas:**
❌ Reused logic (write named function instead)
❌ Very complex logic (hard to read)
❌ When simple function is clearer

**Range-Based For Loops:**

\`\`\`cpp
vector<int> v = {1, 2, 3, 4, 5};

// Read-only (copy)
for(auto x : v) {
    cout << x << " ";
}

// Modify elements (reference)
for(auto& x : v) {
    x *= 2;
}

// Efficient read-only (const reference)
for(const auto& x : v) {
    cout << x << " ";
}
\`\`\`

**Works with:**
- vector, set, map, array, string
- Any container with begin()/end()
- C-style arrays

**Performance tips:**
\`\`\`cpp
// SLOW: Copies each pair
for(auto p : mapOfLargeObjects) {
    // ...
}

// FAST: References each pair
for(const auto& p : mapOfLargeObjects) {
    // ...
}

// C++17: Structured bindings
for(const auto& [key, value] : map) {
    cout << key << ": " << value << endl;
}
\`\`\`

**auto - Type Deduction:**

Already covered in previous section, but key points:

\`\`\`cpp
// Iterators
auto it = find(v.begin(), v.end(), target);

// Complex types
auto result = complexFunction();

// Range-for
for(auto& x : container) { }

// Structured bindings (C++17)
auto [min_it, max_it] = minmax_element(v.begin(), v.end());
\`\`\`

**C++17 Structured Bindings:**

\`\`\`cpp
// Pairs
pair<int, int> p = {1, 2};
auto [x, y] = p;  // x=1, y=2

// Maps
map<string, int> m = {{"Alice", 25}, {"Bob", 30}};
for(auto& [name, age] : m) {
    cout << name << " is " << age << endl;
}

// Arrays/tuples
array<int, 3> arr = {1, 2, 3};
auto [a, b, c] = arr;

// Return multiple values
auto divmod(int a, int b) {
    return make_pair(a / b, a % b);
}
auto [quotient, remainder] = divmod(17, 5);
\`\`\`

**C++14 Generic Lambdas:**

\`\`\`cpp
// Works with any type
auto print = [](auto x) {
    cout << x << endl;
};

print(42);      // int
print(3.14);    // double
print("hello"); // const char*

// Generic sorting
auto sorter = [](auto& container) {
    sort(container.begin(), container.end());
};
\`\`\`

**C++11 Move Semantics:**

\`\`\`cpp
// Avoid copies
vector<int> v1(1000000);
vector<int> v2 = move(v1);  // Transfer ownership, no copy!
// v1 is now empty

// Useful in contests for performance
\`\`\`

**Initializer Lists:**

\`\`\`cpp
// Uniform initialization
vector<int> v = {1, 2, 3, 4, 5};
set<int> s = {3, 1, 4, 1, 5};  // Automatically sorted
map<string, int> m = {{"Alice", 25}, {"Bob", 30}};

// Arrays
int arr[] = {1, 2, 3};

// Pairs
pair<int, int> p = {1, 2};
\`\`\`

**emplace vs push:**

\`\`\`cpp
vector<pair<int, int>> v;

// Old way (creates temp, then copies)
v.push_back(make_pair(1, 2));
v.push_back({1, 2});  // C++11

// New way (constructs in place)
v.emplace_back(1, 2);  // Slightly faster
\`\`\`

**constexpr - Compile-Time Evaluation:**

\`\`\`cpp
constexpr int factorial(int n) {
    return n <= 1 ? 1 : n * factorial(n - 1);
}

constexpr int f10 = factorial(10);  // Computed at compile time!
\`\`\`

**Time Savings Examples:**

**Before modern C++:**
\`\`\`cpp
bool cmp(const pair<int, int>& a, const pair<int, int>& b) {
    return a.second < b.second;
}

sort(pairs.begin(), pairs.end(), cmp);
\`\`\`

**After:**
\`\`\`cpp
sort(pairs.begin(), pairs.end(), [](auto& a, auto& b) {
    return a.second < b.second;
});
\`\`\`

Saved: ~5 lines, kept logic inline

**Performance Considerations:**

1. **Lambdas:** Zero overhead when inlined (same as regular function)
2. **auto:** Zero overhead (just type deduction)
3. **Range-for:** Same as manual loop
4. **Structured bindings:** Zero overhead (compiler optimizes)
5. **Move:** Can be much faster than copy for large objects

**When to Avoid Modern Features:**

1. **Platform doesn't support:** Check judge C++ version
2. **Unclear code:** If lambda makes code harder to read
3. **Debugging:** Sometimes explicit types help debugging
4. **Compiler errors:** Template errors can be confusing

**Compiler Flags:**

\`\`\`bash
g++ -std=c++17 solution.cpp  # C++17 features
g++ -std=c++20 solution.cpp  # C++20 features (if available)
\`\`\`

**Bottom Line:**

Modern C++ features save time and make code cleaner:
- **Lambdas:** Custom sorting and STL algorithms
- **auto:** Less typing, especially with iterators
- **Range-for:** Cleaner iteration
- **Structured bindings:** Elegant pair/tuple handling

Use them! They're not just syntactic sugar—they make you faster.`,
        },
        {
            question: 'C++20 introduced ranges and concepts. While not universally available on judges yet, explain what these features offer and how they might change competitive programming in the future.',
            answer: `C++20 brings paradigm-shifting features to C++. Here's what the future holds:

**Ranges - Composable Algorithms:**

Traditional STL:
\`\`\`cpp
vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

// Filter even numbers, square them, take first 3
vector<int> result;
copy_if(v.begin(), v.end(), back_inserter(result), 
        [](int x) { return x % 2 == 0; });

for(auto& x : result) x = x * x;

result.resize(min(result.size(), 3ul));
\`\`\`

C++20 Ranges:
\`\`\`cpp
#include <ranges>
namespace views = std::ranges::views;

vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

// Composable pipeline
auto result = v 
    | views::filter([](int x) { return x % 2 == 0; })
    | views::transform([](int x) { return x * x; })
    | views::take(3);

// Result is a view (lazy evaluation)
for(int x : result) {
    cout << x << " ";  // 4 16 36
}
\`\`\`

**Key Advantages:**

1. **Composability:** Chain operations with |
2. **Readability:** Read left-to-right like a pipeline
3. **Lazy Evaluation:** No intermediate containers
4. **Performance:** Often faster due to fusion

**Common Range Views:**

\`\`\`cpp
#include <ranges>
namespace ranges = std::ranges;
namespace views = std::ranges::views;

vector<int> v = {1, 2, 3, 4, 5};

// Filter
auto evens = v | views::filter([](int x) { return x % 2 == 0; });

// Transform
auto squared = v | views::transform([](int x) { return x * x; });

// Take first n
auto first3 = v | views::take(3);

// Drop first n
auto rest = v | views::drop(2);

// Reverse
auto reversed = v | views::reverse;

// Enumerate (with index)
for(auto [i, val] : v | views::enumerate) {
    cout << i << ": " << val << endl;
}

// Zip (combine two ranges)
vector<int> a = {1, 2, 3};
vector<char> b = {'a', 'b', 'c'};
for(auto [num, ch] : views::zip(a, b)) {
    cout << num << ch << " ";  // 1a 2b 3c
}

// iota (generate sequence)
for(int i : views::iota(1, 11)) {  // 1 to 10
    cout << i << " ";
}
\`\`\`

**Practical CP Examples:**

**Example 1: Find kth even number:**

Traditional:
\`\`\`cpp
int findKthEven(vector<int>& v, int k) {
    vector<int> evens;
    for(int x : v) {
        if(x % 2 == 0) evens.push_back(x);
    }
    return evens[k-1];
}
\`\`\`

Ranges:
\`\`\`cpp
int findKthEven(vector<int>& v, int k) {
    auto evens = v | views::filter([](int x) { return x % 2 == 0; });
    return ranges::next(evens.begin(), k-1);
}
\`\`\`

**Example 2: Sum of squares of positive numbers:**

Traditional:
\`\`\`cpp
int sumSquaresPositive(vector<int>& v) {
    int sum = 0;
    for(int x : v) {
        if(x > 0) sum += x * x;
    }
    return sum;
}
\`\`\`

Ranges:
\`\`\`cpp
int sumSquaresPositive(vector<int>& v) {
    auto squares = v 
        | views::filter([](int x) { return x > 0; })
        | views::transform([](int x) { return x * x; });
    
    return ranges::fold_left(squares, 0, plus{});
}
\`\`\`

**Concepts - Constrained Templates:**

Traditional template (anything goes):
\`\`\`cpp
template<typename T>
T add(T a, T b) {
    return a + b;  // Works if T supports +
}

add(5, 10);        // OK
add("hi", "bye");  // Compiles but may not work as expected
\`\`\`

With Concepts:
\`\`\`cpp
#include <concepts>

template<typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template<Numeric T>
T add(T a, T b) {
    return a + b;
}

add(5, 10);        // OK
add(3.14, 2.71);   // OK
add("hi", "bye");  // Compilation error with clear message!
\`\`\`

**Benefits:**
1. **Better error messages:** Clear constraint violations
2. **Self-documenting:** Concepts describe requirements
3. **Overload resolution:** Better function selection
4. **Type safety:** Catch errors at template instantiation

**Common Standard Concepts:**

\`\`\`cpp
#include <concepts>

// Arithmetic types
template<std::integral T>  // int, long long, etc.
void processInt(T value) { }

template<std::floating_point T>  // float, double
void processFloat(T value) { }

// Comparable
template<std::totally_ordered T>
T maximum(T a, T b) {
    return (a > b) ? a : b;
}

// Sortable
template<std::sortable T>
void sortContainer(T& container) {
    sort(container.begin(), container.end());
}
\`\`\`

**Custom Concepts for CP:**

\`\`\`cpp
// Container with size
template<typename T>
concept HasSize = requires(T container) {
    { container.size() } -> std::convertible_to<size_t>;
};

// Iterable
template<typename T>
concept Iterable = requires(T container) {
    { container.begin() };
    { container.end() };
};

// Usage
template<HasSize T>
size_t getSize(const T& container) {
    return container.size();
}
\`\`\`

**C++20 Other Features:**

**Three-Way Comparison (Spaceship Operator):**
\`\`\`cpp
struct Point {
    int x, y;
    
    // Before: Need to write ==, !=, <, >, <=, >=
    // After: Just one operator
    auto operator<=>(const Point& other) const = default;
};

Point p1{1, 2}, p2{1, 3};
bool less = p1 < p2;  // Automatically available!
\`\`\`

**Designated Initializers:**
\`\`\`cpp
struct Config {
    int timeout;
    bool verbose;
    string mode;
};

Config cfg = {
    .timeout = 30,
    .verbose = true,
    .mode = "fast"
};
\`\`\`

**constexpr Improvements:**
\`\`\`cpp
// constexpr vectors
constexpr vector<int> makeVector() {
    vector<int> v = {1, 2, 3};
    v.push_back(4);
    return v;
}
\`\`\`

**How C++20 Will Change CP:**

**Current State:**
- Most judges: C++17
- Some judges: C++20 available but not default
- Problem: Code must be portable

**Future (2-3 years):**
- C++20 becomes standard
- Ranges widely used for clarity
- Concepts improve template code
- Faster development

**Adoption Timeline:**
- Codeforces: C++20 partially available
- AtCoder: C++20 available
- Google Code Jam: Usually latest
- ICPC: Slower adoption (varies by site)

**What to Learn Now:**

1. **Ranges:** Familiarize with syntax, but don't rely on it yet
2. **Concepts:** Understand the idea, useful for library code
3. **Other C++20:** Three-way comparison, designated initializers

**Practical Strategy for Contests:**

**Today (C++17):**
\`\`\`cpp
// Use traditional STL
vector<int> result;
copy_if(v.begin(), v.end(), back_inserter(result),
        [](int x) { return x % 2 == 0; });
\`\`\`

**When C++20 is universal:**
\`\`\`cpp
// Use ranges
auto result = v | views::filter([](int x) { return x % 2 == 0; });
\`\`\`

**Transition Period:**
\`\`\`cpp
// Have both ready
#if __cplusplus >= 202002L  // C++20
    #include <ranges>
    // Use ranges
#else
    // Use traditional
#endif
\`\`\`

**Learning Resources:**

1. **Practice locally** with C++20
2. **Read cppreference** for examples
3. **Watch conference talks** on ranges
4. **Experiment** with online compilers

**Bottom Line:**

**C++20 is the future:**
- **Ranges:** Will revolutionize how we write algorithms
- **Concepts:** Better template error messages
- **Other features:** Quality of life improvements

**But for now:**
- Master C++17 (current standard on most judges)
- Learn C++20 features for when they become available
- Write portable code that works on both

**The paradigm shift:** From imperative loops to declarative pipelines. From error-prone templates to constraint-checked concepts.

**When C++20 is universal, CP will be faster and more expressive!**`,
        },
        {
            question: 'Explain the practical differences between using initializer lists, uniform initialization, and traditional initialization in competitive programming. When does each method matter, and what are the potential pitfalls?',
            answer: `C++11 introduced multiple initialization syntaxes. Here's when each matters and their pitfalls:

**Three Initialization Styles:**

**1. Traditional Initialization:**
\`\`\`cpp
int x = 5;
string s = "hello";
vector<int> v = vector<int>(10);
pair<int, int> p = make_pair(1, 2);
\`\`\`

**2. Uniform Initialization (Brace Init):**
\`\`\`cpp
int x{5};
string s{"hello"};
vector<int> v{10};  // Creates vector with one element: 10
pair<int, int> p{1, 2};
\`\`\`

**3. Initializer Lists:**
\`\`\`cpp
vector<int> v = {1, 2, 3, 4, 5};
set<int> s = {3, 1, 4, 1, 5};
map<string, int> m = {{"Alice", 25}, {"Bob", 30}};
\`\`\`

**Key Differences:**

**Narrowing Conversions:**

\`\`\`cpp
// Traditional: Allows narrowing (data loss)
int x = 3.14;  // OK, x = 3

// Uniform: Prevents narrowing
int y{3.14};  // Compilation error!

// But in CP, we usually want traditional:
double d = 3.14;
int i = d;  // Intentional truncation
\`\`\`

**Vector Ambiguity (MAJOR PITFALL):**

\`\`\`cpp
// Traditional: Size constructor
vector<int> v1(10);     // 10 elements, all 0
// Result: {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

// Uniform: Initializer list
vector<int> v2{10};     // 1 element with value 10
// Result: {10}

// This is confusing!
vector<int> v3(10, 5);  // 10 elements, all 5
vector<int> v4{10, 5};  // 2 elements: 10 and 5

// In CP, prefer traditional for size:
vector<int> v(n);       // n elements
vector<int> v(n, init); // n elements, all init
\`\`\`

**Initializer Lists for Containers:**

\`\`\`cpp
// Very convenient in CP:
vector<int> v = {1, 2, 3, 4, 5};
set<int> s = {3, 1, 4};
map<int, string> m = {{1, "one"}, {2, "two"}};
pair<int, int> p = {1, 2};

// Direction vectors:
int dx[] = {0, 1, 0, -1};
int dy[] = {1, 0, -1, 0};

// Grid initialization:
vector<string> grid = {
    "#####",
    "#...#",
    "#.#.#",
    "#...#",
    "#####"
};
\`\`\`

**Most Vexing Parse (C++ Gotcha):**

\`\`\`cpp
// Traditional: Ambiguous!
Widget w();  // Function declaration, NOT object!

// Uniform: Clear
Widget w{};  // Object initialization

// In CP, rarely matters because:
Widget w;    // Default constructor (no parens needed)
\`\`\`

**auto with Initializers (IMPORTANT PITFALL):**

\`\`\`cpp
// Different types!
auto x = {1, 2, 3};  // x is initializer_list<int>
auto y{1, 2, 3};     // C++17: Error (too many initializers)
auto z{1};           // C++17: int (single element)

vector<int> v = {1, 2, 3};     // v is vector<int>
auto w = {1, 2, 3};            // w is initializer_list!

// In CP, be explicit:
vector<int> v = {1, 2, 3};     // Clear and correct
\`\`\`

**Practical Guidelines for CP:**

**Use Traditional () for:**
\`\`\`cpp
// Constructors with size
vector<int> v(n);
vector<int> v(n, value);
string s(n, 'a');

// Default construction
int x = 0;
double d = 0.0;

// Intentional conversions
int i = static_cast<int>(d);
\`\`\`

**Use Initializer Lists {} for:**
\`\`\`cpp
// Literal collections
vector<int> v = {1, 2, 3, 4, 5};
set<int> s = {1, 2, 3};
map<int, int> m = {{1,2}, {3,4}};

// Small constant arrays
int dx[] = {0, 1, 0, -1};

// Pairs
pair<int, int> p = {1, 2};

// Struct initialization
struct Point { int x, y; };
Point p = {10, 20};
\`\`\`

**Use Uniform {} for:**
\`\`\`cpp
// Preventing narrowing (when you want it)
int x{5};  // Will error if initialized from double

// Aggregate initialization
array<int, 5> arr{1, 2, 3, 4, 5};

// When style guide requires it (rare in CP)
\`\`\`

**Common Pitfalls in CP:**

**Pitfall 1: Vector Size vs Elements:**
\`\`\`cpp
// Want: 10 zeros
vector<int> wrong{10};  // 1 element: {10}
vector<int> right(10);  // 10 elements: {0,0,0,0,0,0,0,0,0,0}

// Want: 10 fives
vector<int> wrong{10, 5};  // 2 elements: {10, 5}
vector<int> right(10, 5);  // 10 elements: all 5
\`\`\`

**Pitfall 2: 2D Vector Initialization:**
\`\`\`cpp
int n = 3, m = 4;

// WRONG ways:
vector<vector<int>> grid1(n, m);  // Error! m is not a vector
vector<vector<int>> grid2{n, m};  // 2 rows, weird sizes

// RIGHT way:
vector<vector<int>> grid(n, vector<int>(m));
// Or with initial value:
vector<vector<int>> grid(n, vector<int>(m, 0));
\`\`\`

**Pitfall 3: auto with Braces:**
\`\`\`cpp
auto v = {1, 2, 3};  // v is initializer_list, NOT vector!

// Use explicit type:
vector<int> v = {1, 2, 3};  // Clear
\`\`\`

**Pitfall 4: Narrowing in Uniform Init:**
\`\`\`cpp
double pi = 3.14159;
int x{pi};  // Error: narrowing conversion

// In CP, often want this:
int x = pi;  // OK, x = 3
\`\`\`

**Performance Considerations:**

All three have **same performance** after optimization:
\`\`\`cpp
vector<int> v1(1000000);           // Fast
vector<int> v2{};                  // Same
v2.resize(1000000);
vector<int> v3 = vector<int>(1000000);  // Same
\`\`\`

**Consistency in Templates:**

\`\`\`cpp
// Template-safe initialization
template<typename T>
void initialize() {
    T value{};     // Always works (zero/default initialization)
    T value = T(); // Also works
    T value;       // May not initialize primitives!
}

int x{};     // x = 0 (zero-initialized)
int y;       // y = garbage (uninitialized)
\`\`\`

**Best Practices for CP:**

1. **Be consistent within a file**
2. **Use () for constructors with sizes**
3. **Use {} for literal lists**
4. **Avoid uniform {} for vectors**
5. **Be explicit with auto**

**My Recommended Style:**

\`\`\`cpp
// Primitives
int n = 0;
long long sum = 0;

// Vectors with size
vector<int> v(n);
vector<int> v(n, init);

// Vectors with elements
vector<int> v = {1, 2, 3, 4, 5};

// Sets/maps
set<int> s = {1, 2, 3};
map<int, int> m = {{1,2}, {3,4}};

// Pairs
pair<int, int> p = {1, 2};

// Arrays
int dx[] = {0, 1, 0, -1};
\`\`\`

**Quick Reference:**

| Syntax | When to Use | Example |
|--------|-------------|---------|
| \`= value\` | Primitives, simple init | \`int x = 5;\` |
| \`(args)\` | Constructor with size | \`vector < int > v(n); \` |
| \`{ list }\` | Literal elements | \`vector < int > v = { 1, 2, 3}; \` |
| \`{ value }\` | Uniform (avoid in CP) | \`int x{ 5}; \` |

**Bottom Line:**

For competitive programming:
- **Use ()** for sized containers: \`vector < int > v(n); \`
- **Use {}** for literal collections: \`vector < int > v = { 1, 2, 3}; \`
- **Be careful** with vector initialization ambiguity
- **Avoid auto** with brace initializers
- **Choose one style** and be consistent!

The main pitfall is the vector ambiguity—remember it and you'll be fine!`,
    },
  ],
} as const ;

