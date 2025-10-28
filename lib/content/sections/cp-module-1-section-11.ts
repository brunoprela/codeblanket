export const templateMetaprogrammingBasicsSection = {
  id: 'cp-m1-s11',
  title: 'Template Metaprogramming Basics',
  content: `

# Template Metaprogramming Basics

## Introduction

Templates are one of C++'s most powerful features, enabling you to write **generic code** that works with multiple types. While competitive programming doesn't require advanced template metaprogramming like production C++ does, understanding basic templates is essential for writing clean, reusable, and efficient code.

Many competitive programmers avoid templates thinking they're too complex or slow. This is a misconception! Basic templates are straightforward, compile to optimized code, and can save you significant time when used appropriately.

In this comprehensive section, we'll explore function templates, class templates, template specialization, when templates are useful in CP, and practical examples you can add to your toolkit.

**Goal**: Master basic C++ templates to write generic, reusable competitive programming code without over-engineering.

---

## Why Templates Matter in Competitive Programming

### The Problem: Code Duplication

**Without templates**, you might write:

\`\`\`cpp
int maxInt(int a, int b) {
    return (a > b) ? a : b;
}

long long maxLL(long long a, long long b) {
    return (a > b) ? a : b;
}

double maxDouble(double a, double b) {
    return (a > b) ? a : b;
}

// Need to write separate function for each type!
\`\`\`

**With templates**:

\`\`\`cpp
template<typename T>
T maximum(T a, T b) {
    return (a > b) ? a : b;
}

// Works with any type!
cout << maximum(5, 10) << endl;         // int
cout << maximum(3.14, 2.71) << endl;    // double
cout << maximum(5LL, 10LL) << endl;     // long long
\`\`\`

### Benefits in CP

✅ **Write Once, Use Everywhere**: One implementation for all types
✅ **Type Safety**: Compiler checks types automatically
✅ **Zero Overhead**: Templates compile to optimized code (no runtime cost)
✅ **Flexibility**: Easy to adapt to new types
✅ **Code Reusability**: Build a library of generic functions

---

## Function Templates: The Basics

### Basic Function Template

\`\`\`cpp
template<typename T>
T maximum(T a, T b) {
    return (a > b) ? a : b;
}

int main() {
    cout << maximum(5, 10) << endl;           // T = int
    cout << maximum(3.14, 2.71) << endl;      // T = double
    cout << maximum('a', 'z') << endl;        // T = char
    return 0;
}
\`\`\`

**How it works:**
1. Compiler sees \`maximum(5, 10)\`
2. Deduces \`T = int\`
3. Generates: \`int maximum(int a, int b) { return (a > b) ? a : b; }\`
4. This happens at **compile time** (zero runtime overhead!)

### Template with Explicit Type

\`\`\`cpp
template<typename T>
T square(T x) {
    return x * x;
}

int main() {
    // Implicit type deduction
    cout << square(5) << endl;        // T = int

    // Explicit type specification
    cout << square<long long>(5) << endl;  // T = long long
    cout << square<double>(5) << endl;     // T = double
}
\`\`\`

**When to use explicit types:**
- Type deduction is ambiguous
- You want a specific type regardless of input
- Converting between types

### Multiple Template Parameters

\`\`\`cpp
template<typename T, typename U>
auto add(T a, U b) -> decltype(a + b) {
    return a + b;
}

int main() {
    cout << add(5, 3.14) << endl;      // int + double = double
    cout << add(10LL, 5) << endl;      // long long + int = long long
    cout << add(1.5, 2.5) << endl;     // double + double = double
}
\`\`\`

**Modern C++14 version (simpler):**
\`\`\`cpp
template<typename T, typename U>
auto add(T a, U b) {
    return a + b;  // Return type deduced automatically!
}
\`\`\`

---

## Generic Functions for Competitive Programming

### Generic Print Function

\`\`\`cpp
template<typename T>
void print(const vector<T>& v) {
    for (const auto& x : v) {
        cout << x << " ";
    }
    cout << "\\n";
}

int main() {
    vector<int> v1 = {1, 2, 3, 4, 5};
    vector<string> v2 = {"hello", "world"};
    vector<double> v3 = {1.1, 2.2, 3.3};
    
    print(v1);  // 1 2 3 4 5
    print(v2);  // hello world
    print(v3);  // 1.1 2.2 3.3
}
\`\`\`

**Enhanced version with different containers:**
\`\`\`cpp
template<typename Container>
void print(const Container& c) {
    for (const auto& x : c) {
        cout << x << " ";
    }
    cout << "\\n";
}

vector<int> v = {1, 2, 3};
set<int> s = {3, 1, 4, 1, 5};
list<string> l = {"a", "b", "c"};

print(v);  // Works!
print(s);  // Works!
print(l);  // Works!
\`\`\`

### Generic Read Function

\`\`\`cpp
template<typename T>
vector<T> read(int n) {
    vector<T> v(n);
    for (auto& x : v) cin >> x;
    return v;
}

int main() {
    auto arr = read<int>(5);        // Read 5 integers
    auto names = read<string>(3);   // Read 3 strings
    auto prices = read<double>(10); // Read 10 doubles
    
    print(arr);
    print(names);
}
\`\`\`

### Generic Min/Max Functions

\`\`\`cpp
template<typename T>
T min3(T a, T b, T c) {
    return min(a, min(b, c));
}

template<typename T>
T max3(T a, T b, T c) {
    return max(a, max(b, c));
}

// Variadic template for any number of arguments (C++11)
template<typename T>
T minimum(T t) {
    return t;
}

template<typename T, typename... Args>
T minimum(T t, Args... args) {
    return min(t, minimum(args...));
}

int main() {
    cout << min3(5, 2, 8) << endl;           // 2
    cout << max3(5, 2, 8) << endl;           // 8
    cout << minimum(5, 2, 8, 1, 9, 3) << endl; // 1
}
\`\`\`

### Generic Swap

\`\`\`cpp
template<typename T>
void swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

int main() {
    int x = 5, y = 10;
    swap(x, y);  // x=10, y=5
    
    string s1 = "hello", s2 = "world";
    swap(s1, s2);  // s1="world", s2="hello"
    
    // Note: std::swap already exists and is better optimized!
    // This is just an example
}
\`\`\`

---

## Template Specialization

Sometimes you need different behavior for specific types.

### Full Specialization

\`\`\`cpp
// Generic template
template<typename T>
string typeString(T value) {
    return "unknown type";
}

// Specialization for int
template<>
string typeString<int>(int value) {
    return "int: " + to_string(value);
}

// Specialization for double
template<>
string typeString<double>(double value) {
    return "double: " + to_string(value);
}

int main() {
    cout << typeString(42) << endl;        // int: 42
    cout << typeString(3.14) << endl;      // double: 3.14
    cout << typeString("hello") << endl;   // unknown type
}
\`\`\`

### Practical CP Example: Generic Comparator

\`\`\`cpp
// Generic ascending comparator
template<typename T>
struct Comparator {
    bool operator()(const T& a, const T& b) const {
        return a < b;
    }
};

// Specialization for pairs: sort by first, then second
template<typename T, typename U>
struct Comparator<pair<T, U>> {
    bool operator()(const pair<T, U>& a, const pair<T, U>& b) const {
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    }
};

int main() {
    vector<int> v = {5, 2, 8, 1};
    sort(v.begin(), v.end(), Comparator<int>());
    
    vector<pair<int, int>> p = {{1, 5}, {2, 3}, {1, 2}};
    sort(p.begin(), p.end(), Comparator<pair<int, int>>());
}
\`\`\`

---

## Class Templates

While function templates are more common in CP, class templates can be useful too.

### Basic Class Template

\`\`\`cpp
template<typename T>
class Stack {
private:
    vector<T> data;
    
public:
    void push(T value) {
        data.push_back(value);
    }
    
    T pop() {
        T value = data.back();
        data.pop_back();
        return value;
    }
    
    bool empty() const {
        return data.empty();
    }
    
    size_t size() const {
        return data.size();
    }
};

int main() {
    Stack<int> intStack;
    intStack.push(1);
    intStack.push(2);
    cout << intStack.pop() << endl;  // 2
    
    Stack<string> strStack;
    strStack.push("hello");
    strStack.push("world");
    cout << strStack.pop() << endl;  // world
}
\`\`\`

### Template with Non-Type Parameters

\`\`\`cpp
template<typename T, int SIZE>
class Array {
private:
    T data[SIZE];
    
public:
    T& operator[](int index) {
        return data[index];
    }
    
    constexpr int size() const {
        return SIZE;
    }
};

int main() {
    Array<int, 10> arr;
    for (int i = 0; i < arr.size(); i++) {
        arr[i] = i * i;
    }
}
\`\`\`

**In CP:** Usually just use \`vector\` or \`array\`, but this shows the concept.

---

## Practical CP Template Examples

### Example 1: Generic Binary Search

\`\`\`cpp
template<typename T, typename Compare>
int binarySearch(const vector<T>& arr, T target, Compare comp) {
    int left = 0, right = arr.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) return mid;
        
        if (comp(arr[mid], target)) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1;
}

int main() {
    vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // Ascending search
    cout << binarySearch(v, 5, less<int>()) << endl;  // 4
    
    // Descending search
    vector<int> v2 = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    cout << binarySearch(v2, 5, greater<int>()) << endl;  // 5
}
\`\`\`

### Example 2: Generic Power Function

\`\`\`cpp
template<typename T>
T power(T base, int exp) {
    T result = 1;
    T current = base;
    
    while (exp > 0) {
        if (exp & 1) result *= current;
        current *= current;
        exp >>= 1;
    }
    
    return result;
}

int main() {
    cout << power(2, 10) << endl;        // 1024
    cout << power(2LL, 30) << endl;      // 1073741824
    cout << power(1.5, 3) << endl;       // 3.375
}
\`\`\`

### Example 3: Generic GCD

\`\`\`cpp
template<typename T>
T gcd(T a, T b) {
    while (b != 0) {
        T temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

int main() {
    cout << gcd(48, 18) << endl;         // 6
    cout << gcd(100LL, 75LL) << endl;    // 25
}
\`\`\`

### Example 4: Generic 2D Vector Creation

\`\`\`cpp
template<typename T>
vector<vector<T>> create2D(int rows, int cols, T init = T()) {
    return vector<vector<T>>(rows, vector<T>(cols, init));
}

int main() {
    auto grid = create2D<int>(10, 20, 0);      // 10x20 grid of 0s
    auto dp = create2D<long long>(100, 100, -1); // 100x100 grid of -1s
    auto visited = create2D<bool>(50, 50, false); // 50x50 grid of false
}
\`\`\`

### Example 5: Generic Debug Print

\`\`\`cpp
template<typename T>
void debug(const char* name, const T& value) {
    cerr << name << " = " << value << endl;
}

template<typename T>
void debug(const char* name, const vector<T>& v) {
    cerr << name << " = [";
    for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) cerr << ", ";
        cerr << v[i];
    }
    cerr << "]" << endl;
}

#define DEBUG(x) debug(#x, x)

int main() {
    int x = 42;
    vector<int> v = {1, 2, 3, 4, 5};
    
    DEBUG(x);  // x = 42
    DEBUG(v);  // v = [1, 2, 3, 4, 5]
}
\`\`\`

---

## When to Use Templates in CP

### ✅ DO Use Templates For:

**1. Utility Functions You Use Often**
\`\`\`cpp
template<typename T>
vector<T> read(int n) {
    vector<T> v(n);
    for (auto& x : v) cin >> x;
    return v;
}
\`\`\`

**2. Generic Algorithms**
\`\`\`cpp
template<typename T>
T binarySearch(const vector<T>& arr, T target) {
    // Generic binary search
}
\`\`\`

**3. Custom Comparators**
\`\`\`cpp
template<typename T>
struct Greater {
    bool operator()(const T& a, const T& b) const {
        return a > b;
    }
};

priority_queue<int, vector<int>, Greater<int>> pq;
\`\`\`

**4. Type-Agnostic Math Functions**
\`\`\`cpp
template<typename T>
T gcd(T a, T b) {
    return b ? gcd(b, a % b) : a;
}
\`\`\`

### ❌ DON'T Use Templates For:

**1. One-Time Use Functions**
\`\`\`cpp
// Don't template this if you only use it once:
template<typename T>
T processData(T x) {
    return x * 2 + 1;
}

// Just write:
int processData(int x) {
    return x * 2 + 1;
}
\`\`\`

**2. Complex Problem-Specific Logic**
\`\`\`cpp
// Don't over-engineer the solution:
template<typename T, typename U, typename V>
auto complexSolution(T a, U b, V c) -> decltype(/* complex stuff */) {
    // This is overkill for CP!
}
\`\`\`

**3. When Simple Code Is Clearer**
\`\`\`cpp
// Overkill:
template<typename Iterator>
void process(Iterator begin, Iterator end) { }

// Better:
void process(vector<int>& v) { }
\`\`\`

---

## Template Compilation

### How Templates Are Compiled

**Important**: Templates are compiled differently than regular functions!

\`\`\`cpp
// template.h
template<typename T>
T add(T a, T b) {
    return a + b;  // Definition must be in header!
}

// main.cpp
#include "template.h"

int main() {
    cout << add(5, 10) << endl;  // Compiler generates int version here
}
\`\`\`

**Key points:**
- Templates are **not compiled** until used
- Compiler generates code for each type used
- Template definitions must be in **header files** (or same file)
- Compilation can be slower (more code to generate)

**In CP**: Usually everything is in one file, so this doesn't matter!

---

## Common Template Errors

### Error 1: Type Deduction Failure

\`\`\`cpp
template<typename T>
T maximum(T a, T b) {
    return (a > b) ? a : b;
}

int main() {
    cout << maximum(5, 3.14) << endl;  // ERROR!
    // Can't deduce T: int or double?
}
\`\`\`

**Fix:**
\`\`\`cpp
cout << maximum<double>(5, 3.14) << endl;  // Explicit type
// Or:
cout << maximum(5.0, 3.14) << endl;  // Both double
\`\`\`

### Error 2: Missing Operator

\`\`\`cpp
template<typename T>
bool isGreater(T a, T b) {
    return a > b;
}

struct Point {
    int x, y;
};

Point p1{1, 2}, p2{3, 4};
isGreater(p1, p2);  // ERROR! Point doesn't have operator>
\`\`\`

**Fix**: Define operator>
\`\`\`cpp
bool operator>(const Point& a, const Point& b) {
    return a.x > b.x;
}
\`\`\`

### Error 3: Complex Error Messages

Template error messages can be LONG and confusing:

\`\`\`
error: no matching function for call to 'maximum(int, const char [6])'
note: candidate: template<class T> T maximum(T, T)
note:   template argument deduction/substitution failed:
note:   deduced conflicting types for parameter 'T' ('int' and 'const char [6]')
\`\`\`

**Tip**: Read the FIRST line of error, ignore the rest initially.

---

## Advanced Template Techniques (Brief Overview)

These are advanced and **rarely needed** in CP, but good to know they exist:

### Variadic Templates

\`\`\`cpp
template<typename... Args>
void print(Args... args) {
    (cout << ... << args) << endl;  // C++17 fold expression
}

print(1, 2, 3, "hello", 4.5);  // Prints: 123hello4.5
\`\`\`

### SFINAE (Substitution Failure Is Not An Error)

\`\`\`cpp
template<typename T>
typename enable_if<is_integral<T>::value, T>::type
add(T a, T b) {
    return a + b;  // Only works for integer types
}
\`\`\`

### Concepts (C++20)

\`\`\`cpp
template<typename T>
concept Numeric = is_arithmetic_v<T>;

template<Numeric T>
T add(T a, T b) {
    return a + b;  // Only works for numeric types
}
\`\`\`

**For CP**: These are overkill. Stick to basic templates!

---

## Practical Template Library for CP

Here's a collection of useful templates for your CP toolkit:

\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

// ==================== TEMPLATE LIBRARY ====================

// Read n elements of type T
template<typename T>
vector<T> read(int n) {
    vector<T> v(n);
    for (auto& x : v) cin >> x;
    return v;
}

// Print container
template<typename Container>
void print(const Container& c) {
    for (const auto& x : c) cout << x << " ";
    cout << "\\n";
}

// Create 2D vector
template<typename T>
vector<vector<T>> create2D(int rows, int cols, T init = T()) {
    return vector<vector<T>>(rows, vector<T>(cols, init));
}

// Generic GCD
template<typename T>
T gcd(T a, T b) {
    return b ? gcd(b, a % b) : a;
}

// Generic LCM
template<typename T>
T lcm(T a, T b) {
    return a / gcd(a, b) * b;
}

// Generic power
template<typename T>
T power(T base, long long exp) {
    T result = 1;
    while (exp > 0) {
        if (exp & 1) result *= base;
        base *= base;
        exp >>= 1;
    }
    return result;
}

// Generic modular power
template<typename T>
T modPower(T base, long long exp, T mod) {
    T result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) result = (result * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
    }
    return result;
}

// Min of multiple values
template<typename T>
T minimum(T t) {
    return t;
}

template<typename T, typename... Args>
T minimum(T t, Args... args) {
    return min(t, minimum(args...));
}

// Max of multiple values
template<typename T>
T maximum(T t) {
    return t;
}

template<typename T, typename... Args>
T maximum(T t, Args... args) {
    return max(t, maximum(args...));
}

// Debug print (only in LOCAL mode)
#ifdef LOCAL
template<typename T>
void debug(const char* name, const T& value) {
    cerr << name << " = " << value << endl;
}

template<typename T>
void debug(const char* name, const vector<T>& v) {
    cerr << name << " = [";
    for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) cerr << ", ";
        cerr << v[i];
    }
    cerr << "]" << endl;
}

#define DEBUG(x) debug(#x, x)
#else
#define DEBUG(x)
#endif

// ==================== SOLUTION ====================

void solve() {
    // Use templates here
    auto arr = read<int>(5);
    print(arr);
    
    auto grid = create2D<int>(10, 10, 0);
    
    DEBUG(arr);
    DEBUG(grid[0]);
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    solve();
    
    return 0;
}
\`\`\`

---

## Templates vs Macros

### When to Use Each

**Templates:**
✅ Type safety
✅ Better error messages
✅ Namespace support
✅ Can be specialized

**Macros:**
✅ Simpler syntax sometimes
✅ Can use for non-type things
✅ Conditional compilation

**Example comparison:**

\`\`\`cpp
// Template version
template<typename T>
T square(T x) {
    return x * x;
}

// Macro version
#define SQUARE(x) ((x) * (x))

int a = 5;
square(a);    // Safe
SQUARE(a++);  // BUG! a incremented twice

// Winner: Template (type-safe, no side effects)
\`\`\`

---

## Performance Considerations

### Do Templates Make Code Slower?

**NO!** Templates have **zero runtime overhead**.

**Why:**
- Templates are resolved at **compile time**
- Generated code is identical to hand-written version
- Compiler can optimize template code fully

**Example:**
\`\`\`cpp
// Template version
template<typename T>
T add(T a, T b) {
    return a + b;
}

// Non-template version
int add(int a, int b) {
    return a + b;
}

// Both compile to identical assembly code!
\`\`\`

**Compilation time:**
- Templates can increase compilation time
- Not a concern in CP (compile once)

---

## Summary

**Key Concepts:**

✅ **Function Templates**: Write once, use for any type
✅ **Generic Code**: Reusable, type-safe utilities
✅ **Zero Overhead**: No runtime performance cost
✅ **Compile-Time**: All resolved during compilation

**When to Use Templates:**

✅ Utility functions (read, print, etc.)
✅ Generic algorithms (gcd, power, etc.)
✅ Custom comparators
✅ Math functions

**When NOT to Use:**

❌ Over-engineering
❌ One-time use code
❌ When simple code is clearer
❌ Complex problem-specific logic

**Practical Tips:**

1. Start with basic templates
2. Add to your template library gradually
3. Don't over-complicate
4. Test your templates thoroughly
5. Keep it simple in contests

---

## Next Steps

Now that you understand basic templates, let's learn about **Common Compilation Errors** and how to fix them quickly in contests!

**Key Takeaway**: Templates are powerful but should be used judiciously in competitive programming. Master the basics (function templates for utilities), build a small library of generic functions, and don't over-engineer. The goal is to save time, not waste it on complex template metaprogramming!
`,
  quizId: 'cp-m1-s11-quiz',
  discussionId: 'cp-m1-s11-discussion',
} as const;
