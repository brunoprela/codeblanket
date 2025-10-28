export const cppBasicsReviewSection = {
  id: 'cp-m1-s6',
  title: 'C++ Basics Review',
  content: `

# C++ Basics Review

## Introduction

Before diving into advanced competitive programming techniques, let's ensure you have a solid grasp of C++ fundamentals. This section reviews essential C++ concepts with a CP-specific lens—what matters most for contests, common pitfalls, and best practices.

**Goal**: Build a strong foundation in C++ basics optimized for competitive programming.

---

## Data Types and Their Ranges

Understanding data types and their ranges is CRITICAL to avoid **Wrong Answer** due to overflow.

### Integer Types

\`\`\`cpp
// signed integers
char c;          // 8 bits:  -128 to 127
short s;         // 16 bits: -32,768 to 32,767
int i;           // 32 bits: -2×10^9 to 2×10^9
long long ll;    // 64 bits: -9×10^18 to 9×10^18

// unsigned integers
unsigned char uc;       // 0 to 255
unsigned short us;      // 0 to 65,535
unsigned int ui;        // 0 to 4×10^9
unsigned long long ull; // 0 to 18×10^18
\`\`\`

### Exact Ranges

\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    cout << "int min: " << INT_MIN << "\\n";       // -2,147,483,648
    cout << "int max: " << INT_MAX << "\\n";       //  2,147,483,647
    
    cout << "long long min: " << LLONG_MIN << "\\n";  // -9,223,372,036,854,775,808
    cout << "long long max: " << LLONG_MAX << "\\n";  //  9,223,372,036,854,775,807
    
    return 0;
}
\`\`\`

### When to Use Each Type

| Type | Range | Use When |
|------|-------|----------|
| int | ~2×10^9 | N ≤ 10^9, sums < 2×10^9 |
| long long | ~9×10^18 | N ≤ 10^18, or int × int |
| unsigned long long | ~18×10^18 | Need all positive, huge values |

**Rule of thumb in CP:**
- **Use int** for most things
- **Use long long** for:
  - Products of two ints
  - Sums of many ints
  - Problem constraints > 10^9

---

## Integer Overflow: The Silent Killer

**Integer overflow** is one of the most common bugs in CP. It happens when a calculation exceeds the data type's range.

### Example of Overflow

\`\`\`cpp
int a = 1000000;
int b = 1000000;
int c = a * b;  // What is c?

cout << c << endl;
// Output: -727379968 (WRONG! Should be 10^12)
\`\`\`

**Why?**
- \`a * b\` is calculated as **int × int = int**
- Result (10^12) > INT_MAX (2×10^9)
- **Overflow!** Wraps around to negative

### Solution: Use long long

\`\`\`cpp
long long a = 1000000;
long long b = 1000000;
long long c = a * b;  // Correct!

cout << c << endl;
// Output: 1000000000000 ✓
\`\`\`

### Subtle Overflow Bug

\`\`\`cpp
int n = 100000;
int sum = 0;

for (int i = 1; i <= n; i++) {
    sum += i;  // Overflow when sum > INT_MAX!
}

cout << sum << endl;
// Output: incorrect (overflow occurred mid-loop)
\`\`\`

**Fix:**
\`\`\`cpp
int n = 100000;
long long sum = 0;  // Use long long for sum!

for (int i = 1; i <= n; i++) {
    sum += i;
}

cout << sum << endl;  // Correct!
\`\`\`

### Detection Trick

**Can the answer overflow int?**
- If answer ≤ 2×10^9: **int is safe**
- If answer > 2×10^9: **use long long**
- If unsure: **use long long** (safer)

**Intermediate calculations:**
\`\`\`cpp
// If a and b are int, but a*b might overflow:
long long result = (long long)a * b;  // Cast before multiply!
\`\`\`

---

## Long Long (ll) Usage

In competitive programming, \`long long\` is used so frequently that we create shortcuts.

### typedef

\`\`\`cpp
typedef long long ll;

ll a = 1000000000000LL;
ll b = a * 2;
\`\`\`

### using (Modern C++)

\`\`\`cpp
using ll = long long;

ll a = 1000000000000LL;
ll b = a * 2;
\`\`\`

### LL Suffix

\`\`\`cpp
long long a = 1000000000;      // int literal, then converted
long long b = 1000000000LL;    // long long literal directly

long long c = 1e18;   // double literal (might lose precision!)
long long d = 1000000000000LL; // Correct long long literal
\`\`\`

**Best practice:**
\`\`\`cpp
using ll = long long;

int main() {
    ll n = 1000000000LL;
    ll result = n * n;  // Safe!
    return 0;
}
\`\`\`

---

## Floating Point Precision Issues

Floating point numbers (\`float\`, \`double\`) have precision limitations.

### The Problem

\`\`\`cpp
double a = 0.1;
double b = 0.2;
double c = a + b;

cout << (c == 0.3) << endl;  // Output: 0 (false!)
cout << setprecision(20) << c << endl;
// Output: 0.30000000000000004441...
\`\`\`

**Why?** Binary floating point can't represent 0.1 exactly!

### Solution: Epsilon Comparison

\`\`\`cpp
const double EPS = 1e-9;

bool areEqual(double a, double b) {
    return abs(a - b) < EPS;
}

double a = 0.1;
double b = 0.2;
double c = a + b;

cout << areEqual(c, 0.3) << endl;  // Output: 1 (true!)
\`\`\`

### When to Use double vs int

**Use int/long long when:**
- Working with whole numbers
- Exact precision needed
- **Most CP problems!**

**Use double when:**
- Problem explicitly involves decimals
- Geometry problems
- Need sqrt, sin, cos, etc.
- Answer asks for decimal output

### double Precision

\`\`\`cpp
#include <iomanip>

double pi = 3.14159265358979;

cout << pi << endl;                    // Default precision
cout << fixed << setprecision(2) << pi << endl;  // 3.14
cout << fixed << setprecision(6) << pi << endl;  // 3.141593
\`\`\`

---

## Arrays vs Vectors

Both arrays and vectors store sequences, but vectors are more flexible.

### C-Style Arrays

\`\`\`cpp
int arr[100];  // Fixed size 100

arr[0] = 1;
arr[1] = 2;

for (int i = 0; i < 100; i++) {
    cout << arr[i] << " ";
}
\`\`\`

**Advantages:**
- Fast (stack allocation)
- Simple syntax

**Disadvantages:**
- Fixed size (must know at compile time)
- No bounds checking
- No size() method

### Vectors

\`\`\`cpp
#include <vector>
using namespace std;

vector<int> v;  // Empty vector

v.push_back(1);
v.push_back(2);
v.push_back(3);

cout << v.size() << endl;  // 3

for (int x : v) {
    cout << x << " ";
}
\`\`\`

**Advantages:**
- Dynamic size
- size() method
- Many useful methods (push_back, pop_back, etc.)
- Can pass to functions easily

**Disadvantages:**
- Slightly slower (heap allocation)
- More memory overhead

### Which to Use in CP?

**Use vectors most of the time!**

**Use arrays when:**
- Size is known and small (< 10^6)
- Need maximum speed
- Global array (easier to initialize to 0)

\`\`\`cpp
// Global array - automatically initialized to 0
int dp[1000][1000];

// Vectors
vector<int> v(n);  // Size n
vector<vector<int>> grid(n, vector<int>(m));  // n×m grid
\`\`\`

---

## String Handling

Strings in C++ can be handled in two ways: C-style char arrays or C++ string class.

### C++ string (Recommended)

\`\`\`cpp
#include <string>
using namespace std;

string s = "hello";

// Length
cout << s.size() << endl;     // 5
cout << s.length() << endl;   // 5

// Concatenation
string t = s + " world";  // "hello world"

// Character access
cout << s[0] << endl;  // 'h'

// Substring
string sub = s.substr(1, 3);  // "ell" (start at 1, length 3)

// Find
int pos = s.find("ll");  // 2 (position of "ll")

// Compare
if (s == "hello") { }
if (s < "world") { }  // Lexicographic comparison
\`\`\`

### Useful String Operations

\`\`\`cpp
string s = "hello";

// Append
s += " world";  // "hello world"
s.append("!");  // "hello world!"

// Insert
s.insert(5, " there");  // "hello there world!"

// Erase
s.erase(5, 6);  // "hello world!" (erase 6 chars from position 5)

// Replace
s.replace(0, 5, "hi");  // "hi world!"

// Convert to uppercase
for (char& c : s) c = toupper(c);

// Convert to lowercase
for (char& c : s) c = tolower(c);

// Reverse
reverse(s.begin(), s.end());
\`\`\`

### String Input

\`\`\`cpp
string s;

// Read single word (stops at whitespace)
cin >> s;

// Read entire line (including spaces)
getline(cin, s);
\`\`\`

**Common bug:**
\`\`\`cpp
int n;
cin >> n;
string s;
getline(cin, s);  // Gets empty line! (newline after n)

// Fix:
cin >> n;
cin.ignore();  // Ignore the newline
getline(cin, s);
\`\`\`

---

## Basic Control Structures

### Loops

\`\`\`cpp
// For loop
for (int i = 0; i < n; i++) {
    // i goes from 0 to n-1
}

// Range-based for loop (C++11)
vector<int> v = {1, 2, 3, 4, 5};
for (int x : v) {
    cout << x << " ";
}

// Range-based for loop with reference (to modify)
for (int& x : v) {
    x *= 2;  // Double each element
}

// While loop
while (condition) {
    // ...
}

// Do-while loop
do {
    // ...
} while (condition);
\`\`\`

### If Statements

\`\`\`cpp
if (condition) {
    // ...
} else if (other_condition) {
    // ...
} else {
    // ...
}

// Ternary operator
int result = (a > b) ? a : b;  // max(a, b)
\`\`\`

### Switch Statement

\`\`\`cpp
switch (value) {
    case 1:
        // ...
        break;
    case 2:
    case 3:  // Fall-through
        // ...
        break;
    default:
        // ...
}
\`\`\`

---

## Functions

### Basic Function

\`\`\`cpp
int add(int a, int b) {
    return a + b;
}

int main() {
    cout << add(3, 5) << endl;  // 8
    return 0;
}
\`\`\`

### Pass by Value vs Pass by Reference

\`\`\`cpp
// Pass by value (copy)
void modifyValue(int x) {
    x = 10;  // Original unchanged
}

// Pass by reference (no copy)
void modifyReference(int& x) {
    x = 10;  // Original changed!
}

int main() {
    int a = 5;
    modifyValue(a);
    cout << a << endl;  // 5 (unchanged)
    
    modifyReference(a);
    cout << a << endl;  // 10 (changed!)
    
    return 0;
}
\`\`\`

**For vectors (large objects), use reference to avoid copying:**
\`\`\`cpp
// Slow: copies entire vector
void process(vector<int> v) {
    // ...
}

// Fast: uses reference
void process(const vector<int>& v) {
    // ...
}
\`\`\`

### Function Overloading

\`\`\`cpp
int max(int a, int b) {
    return (a > b) ? a : b;
}

long long max(long long a, long long b) {
    return (a > b) ? a : b;
}

double max(double a, double b) {
    return (a > b) ? a : b;
}
\`\`\`

---

## Input/Output Basics

### Basic I/O

\`\`\`cpp
int n;
cin >> n;

cout << n << endl;
\`\`\`

### Multiple Input

\`\`\`cpp
int a, b, c;
cin >> a >> b >> c;

vector<int> v(n);
for (int& x : v) {
    cin >> x;
}
\`\`\`

### Multiple Output

\`\`\`cpp
cout << a << " " << b << " " << c << "\\n";

for (int x : v) {
    cout << x << " ";
}
cout << "\\n";
\`\`\`

---

## Common C++ Idioms for CP

### Read n integers into vector

\`\`\`cpp
int n;
cin >> n;
vector<int> a(n);
for (int& x : a) cin >> x;
\`\`\`

### Min and Max

\`\`\`cpp
int a = 5, b = 10;
int minimum = min(a, b);  // 5
int maximum = max(a, b);  // 10

// Min of three
int min3 = min({a, b, c});
int max3 = max({a, b, c});
\`\`\`

### Swap

\`\`\`cpp
int a = 5, b = 10;
swap(a, b);
// Now a = 10, b = 5
\`\`\`

### Fill array with value

\`\`\`cpp
int arr[100];
fill(arr, arr + 100, -1);

vector<int> v(100);
fill(v.begin(), v.end(), -1);
\`\`\`

### Sort

\`\`\`cpp
vector<int> v = {5, 2, 8, 1, 9};
sort(v.begin(), v.end());
// Now v = {1, 2, 5, 8, 9}

// Descending
sort(v.begin(), v.end(), greater<int>());
// Now v = {9, 8, 5, 2, 1}
\`\`\`

### Reverse

\`\`\`cpp
vector<int> v = {1, 2, 3, 4, 5};
reverse(v.begin(), v.end());
// Now v = {5, 4, 3, 2, 1}
\`\`\`

---

## Common Beginner Mistakes

### 1. Array Index Out of Bounds

❌ **WRONG:**
\`\`\`cpp
int arr[5];
arr[5] = 10;  // Index 5 is out of bounds! (0-4 only)
\`\`\`

### 2. Uninitialized Variables

❌ **WRONG:**
\`\`\`cpp
int x;
cout << x;  // Undefined behavior! x has garbage value
\`\`\`

✅ **CORRECT:**
\`\`\`cpp
int x = 0;
cout << x;
\`\`\`

### 3. Integer Division

❌ **WRONG:**
\`\`\`cpp
int a = 5, b = 2;
double result = a / b;
cout << result;  // Output: 2 (not 2.5!)
// a / b is int / int = int, then converted to double
\`\`\`

✅ **CORRECT:**
\`\`\`cpp
int a = 5, b = 2;
double result = (double)a / b;
cout << result;  // Output: 2.5
\`\`\`

### 4. Using = Instead of ==

❌ **WRONG:**
\`\`\`cpp
if (x = 5) {  // Assignment, not comparison!
    // Always executes (x is assigned 5, which is truthy)
}
\`\`\`

✅ **CORRECT:**
\`\`\`cpp
if (x == 5) {  // Comparison
    // ...
}
\`\`\`

### 5. Off-by-One Errors

❌ **WRONG:**
\`\`\`cpp
for (int i = 1; i <= n; i++) {
    arr[i] = i;  // If arr has size n, arr[n] is out of bounds!
}
\`\`\`

✅ **CORRECT:**
\`\`\`cpp
for (int i = 0; i < n; i++) {
    arr[i] = i;
}
// Or
for (int i = 1; i <= n; i++) {
    arr[i - 1] = i;
}
\`\`\`

---

## Essential Code Snippets

### Template for Most CP Problems

\`\`\`cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int n;
    cin >> n;
    
    vector<int> a(n);
    for (int& x : a) cin >> x;
    
    // Solve problem
    
    cout << answer << "\\n";
    
    return 0;
}
\`\`\`

### Reading Multiple Test Cases

\`\`\`cpp
int t;
cin >> t;
while (t--) {
    // Solve one test case
    int n;
    cin >> n;
    
    // ...
    
    cout << answer << "\\n";
}
\`\`\`

### 2D Grid Input

\`\`\`cpp
int n, m;
cin >> n >> m;

vector<vector<int>> grid(n, vector<int>(m));

for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
        cin >> grid[i][j];
    }
}
\`\`\`

---

## Summary

**Essential C++ Knowledge for CP:**

✅ **Data Types**: Know int, long long, double ranges
✅ **Overflow**: Watch for integer overflow
✅ **Vectors**: Use vectors for dynamic arrays
✅ **Strings**: Use C++ string class
✅ **Fast I/O**: Always include fast I/O setup
✅ **Common Idioms**: sort, min, max, swap, fill

**Common Bugs to Avoid:**

❌ Integer overflow
❌ Array out of bounds
❌ Uninitialized variables
❌ Using = instead of ==
❌ Off-by-one errors

**Next Steps:**

Now that you know C++ basics, let's explore **modern C++ features** (C++11/14/17/20) that make competitive programming easier and faster!

**Key Takeaway**: Solid C++ fundamentals prevent bugs and save time. Master these basics before moving to advanced techniques.
`,
  quizId: 'cp-m1-s6-quiz',
  discussionId: 'cp-m1-s6-discussion',
} as const;
