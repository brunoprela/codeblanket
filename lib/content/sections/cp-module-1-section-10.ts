export const memoryManagementCpSection = {
  id: 'cp-m1-s10',
  title: 'Memory Management for CP',
  content: `

# Memory Management for CP

## Introduction

**Memory Limit Exceeded (MLE)** can be just as frustrating as Time Limit Exceeded in competitive programming. While many beginners focus solely on time complexity, understanding memory management is equally crucial. A brilliant O(N log N) algorithm is useless if it requires 10 GB of memory when the limit is 256 MB!

In this comprehensive section, we'll explore memory allocation strategies, calculate memory usage precisely, understand stack vs heap, optimize memory consumption, and learn to avoid MLE while maintaining efficiency.

**Goal**: Master memory management to avoid MLE and optimize space complexity for competitive programming.

---

## Memory Layout in C++ Programs

### The Five Memory Segments

When your program runs, memory is divided into five segments:

\`\`\`
High Memory Addresses
┌─────────────────┐
│      Stack      │  ← Local variables, function calls
├─────────────────┤
│       ↓         │  ← Stack grows downward
│   (free space)  │
│       ↑         │  ← Heap grows upward
├─────────────────┤
│      Heap       │  ← Dynamic memory (new/malloc)
├─────────────────┤
│  BSS Segment    │  ← Uninitialized global variables
├─────────────────┤
│  Data Segment   │  ← Initialized global variables
├─────────────────┤
│  Code Segment   │  ← Program instructions
└─────────────────┘
Low Memory Addresses
\`\`\`

### Understanding Each Segment

**1. Stack:**
- Size: ~1-8 MB (OS dependent)
- Fast allocation/deallocation
- Automatic memory management
- LIFO structure
- Limited size!

**2. Heap:**
- Size: Up to memory limit (256MB-1GB in CP)
- Slower allocation/deallocation
- Manual management (\`new\`/\`delete\`)
- Can be fragmented

**3. Data Segment:**
- Initialized global/static variables
- Size determined at compile time
- Not counted toward stack limit!

**4. BSS (Block Started by Symbol):**
- Uninitialized global/static variables
- Automatically initialized to 0
- Efficient (doesn't store zeros in executable)

**5. Code:**
- Your program's machine code
- Read-only
- Shared among multiple program instances

---

## Stack vs Heap Allocation: Deep Dive

### Stack Allocation

**What Goes on the Stack:**

\`\`\`cpp
void function() {
    int x = 5;                    // Stack
    int arr[100];                 // Stack
    char str[50];                 // Stack
    double d;                     // Stack
    // All freed automatically when function returns
}
\`\`\`

**Stack Characteristics:**

**Advantages:**
- ⚡ **Extremely fast**: Just moving stack pointer
- 🤖 **Automatic management**: No manual free()
- 🔒 **Cache-friendly**: Better CPU cache utilization
- 🐛 **Less prone to memory leaks**: Automatic cleanup

**Disadvantages:**
- ⚠️ **Limited size**: ~1-8 MB typically
- 📏 **Fixed size**: Must know size at compile time
- 💥 **Stack overflow**: Easy to exceed limit
- 🚫 **Can't return**: Local arrays can't be returned

**Stack Size by Platform:**
\`\`\`
Linux:         8 MB default (can change with ulimit)
Windows:       1 MB default
macOS:         8 MB default
Online judges: Usually 8 MB
\`\`\`

### Heap Allocation

**What Goes on the Heap:**

\`\`\`cpp
int* ptr = new int;                    // Single int on heap
int* arr = new int[1000000];           // Array on heap
vector<int> v(1000000);                // Vector uses heap internally

delete ptr;                             // Must free!
delete[] arr;                           // Must free!
// Vector automatically frees in destructor
\`\`\`

**Heap Characteristics:**

**Advantages:**
- 💾 **Large size**: Up to memory limit (256MB-1GB)
- 📐 **Dynamic size**: Size determined at runtime
- ♻️ **Flexible lifetime**: Survives function return
- 🔄 **Resizable**: Can grow/shrink during execution

**Disadvantages:**
- 🐌 **Slower**: Complex allocation algorithm
- 🧠 **Manual management**: Must delete what you new
- 🕳️ **Fragmentation**: Memory can become fragmented
- 💣 **Memory leaks**: Easy to forget to free

---

## Memory in Competitive Programming

### The Three Safe Strategies

#### Strategy 1: Global Arrays (Data Segment)

\`\`\`cpp
// Global array - goes in data segment, NOT stack!
int arr[1000000];              // 4 MB - perfectly fine
long long big[500000];         // 4 MB - perfectly fine
int dp[3000][3000];            // 36 MB - perfectly fine

int main() {
    // Use arrays here
    for (int i = 0; i < 1000000; i++) {
        arr[i] = i;
    }
    return 0;
}
\`\`\`

**Why Global Arrays Work:**
- Stored in data/BSS segment
- NOT on stack
- Can be HUGE (up to memory limit)
- Automatically initialized to 0

**Example:**
\`\`\`cpp
int visited[1000000];  // All zeros by default

int main() {
    // visited[i] is already 0 for all i
}
\`\`\`

#### Strategy 2: Vectors (Heap)

\`\`\`cpp
int main() {
    // Vector uses heap internally
    vector<int> v(1000000);                           // 4 MB - OK
    vector<vector<int>> grid(1000, vector<int>(1000)); // 4 MB - OK
    
    // Automatic memory management!
    return 0;  // Vector destructor frees memory
}
\`\`\`

**Why Vectors Work:**
- Allocated on heap
- Automatic memory management
- Can resize dynamically
- Exception-safe

**Vector Memory Management:**
\`\`\`cpp
{
    vector<int> v(1000000);  // Heap allocation
    // Use v...
}  // v goes out of scope, memory automatically freed!
\`\`\`

#### Strategy 3: Dynamic Allocation (Heap)

\`\`\`cpp
int main() {
    // Manual heap allocation
    int* arr = new int[1000000];     // 4 MB on heap
    
    // Use arr...
    for (int i = 0; i < 1000000; i++) {
        arr[i] = i;
    }
    
    delete[] arr;  // Must free!
    return 0;
}
\`\`\`

**When to Use:**
- Need very precise control
- Interfacing with C libraries
- Usually **vectors are better!**

### What NOT to Do

#### ❌ Large Local Arrays

\`\`\`cpp
int main() {
    int arr[1000000];  // 4 MB on stack - STACK OVERFLOW!
    return 0;
}
// Program crashes before even starting!
\`\`\`

**Why it fails:**
- Stack limit is ~8 MB
- 4 MB is half the stack!
- Other things on stack too
- **Instant stack overflow**

#### ❌ Nested Function Large Arrays

\`\`\`cpp
void dfs(int node) {
    int temp[100000];  // 400 KB on stack
    // ... some processing
    dfs(child);  // Recursive call adds another 400 KB!
}

int main() {
    dfs(0);  // Stack overflow after ~20 recursive calls!
}
\`\`\`

**Why it fails:**
- Each recursive call adds to stack
- 400 KB × 20 calls = 8 MB
- Stack overflow!

**Fix:**
\`\`\`cpp
int temp[100000];  // Make it global

void dfs(int node) {
    // Use global temp array
    dfs(child);
}
\`\`\`

---

## Memory Limits in Competitive Programming

### Typical Memory Limits by Platform

| Platform | Typical Limit | Notes |
|----------|---------------|-------|
| **Codeforces** | 256 MB | Some problems: 512 MB |
| **AtCoder** | 1024 MB | More generous |
| **CodeChef** | 256 MB | Standard |
| **SPOJ** | 256 MB | Varies by problem |
| **HackerRank** | 256 MB | Standard |
| **USACO** | 256 MB | Some 512 MB |
| **Google Code Jam** | 1 GB | Generous |
| **ICPC** | 256 MB | Standard |

### What Memory Limit Means

**256 MB limit means:**
- Total memory usage ≤ 256 MB
- Includes:
  - Your arrays/vectors
  - Stack space
  - Heap allocations
  - OS overhead (~20 MB)
- **Effective limit: ~230 MB usable**

---

## Calculating Memory Usage

### Data Type Sizes

\`\`\`cpp
#include <iostream>
using namespace std;

int main() {
    cout << "bool:          " << sizeof(bool) << " bytes\\n";
    cout << "char:          " << sizeof(char) << " bytes\\n";
    cout << "short:         " << sizeof(short) << " bytes\\n";
    cout << "int:           " << sizeof(int) << " bytes\\n";
    cout << "long long:     " << sizeof(long long) << " bytes\\n";
    cout << "float:         " << sizeof(float) << " bytes\\n";
    cout << "double:        " << sizeof(double) << " bytes\\n";
    cout << "long double:   " << sizeof(long double) << " bytes\\n";
    cout << "pointer:       " << sizeof(void*) << " bytes\\n";
}
\`\`\`

**Output (typical 64-bit system):**
\`\`\`
bool:          1 bytes
char:          1 bytes
short:         2 bytes
int:           4 bytes
long long:     8 bytes
float:         4 bytes
double:        8 bytes
long double:   16 bytes (sometimes 8 or 12)
pointer:       8 bytes (64-bit system)
\`\`\`

### Memory Calculation Examples

#### Example 1: Simple Array

\`\`\`cpp
int arr[100000];
\`\`\`

**Calculation:**
\`\`\`
Memory = 100,000 elements × 4 bytes/int
       = 400,000 bytes
       = 400 KB
       = 0.4 MB ✅
\`\`\`

#### Example 2: Large Array

\`\`\`cpp
int arr[10000000];
\`\`\`

**Calculation:**
\`\`\`
Memory = 10,000,000 × 4
       = 40,000,000 bytes
       = 40 MB ✅
\`\`\`

#### Example 3: 2D Array

\`\`\`cpp
int grid[1000][1000];
\`\`\`

**Calculation:**
\`\`\`
Memory = 1000 × 1000 × 4
       = 1,000,000 × 4
       = 4,000,000 bytes
       = 4 MB ✅
\`\`\`

#### Example 4: Large 2D Array

\`\`\`cpp
int dp[5000][5000];
\`\`\`

**Calculation:**
\`\`\`
Memory = 5000 × 5000 × 4
       = 25,000,000 × 4
       = 100,000,000 bytes
       = 100 MB ✅ (fits in 256 MB)
\`\`\`

#### Example 5: Too Large!

\`\`\`cpp
long long dp[10000][10000];
\`\`\`

**Calculation:**
\`\`\`
Memory = 10,000 × 10,000 × 8
       = 100,000,000 × 8
       = 800,000,000 bytes
       = 800 MB ❌ (doesn't fit in 256 MB!)
\`\`\`

### Quick Memory Reference Table

| Size | int (4B) | long long (8B) | double (8B) |
|------|----------|----------------|-------------|
| 10³ | 4 KB | 8 KB | 8 KB |
| 10⁴ | 40 KB | 80 KB | 80 KB |
| 10⁵ | 400 KB | 800 KB | 800 KB |
| 10⁶ | 4 MB | 8 MB | 8 MB |
| 10⁷ | 40 MB | 80 MB | 80 MB |
| 10⁸ | 400 MB ❌ | 800 MB ❌ | 800 MB ❌ |

**2D Arrays:**

| Size | int | long long |
|------|-----|-----------|
| 1000² | 4 MB | 8 MB |
| 2000² | 16 MB | 32 MB |
| 3000² | 36 MB | 72 MB |
| 4000² | 64 MB | 128 MB |
| 5000² | 100 MB | 200 MB |
| 6000² | 144 MB | 288 MB ❌ |
| 10000² | 400 MB ❌ | 800 MB ❌ |

---

## Memory Optimization Techniques

### Technique 1: Use Smaller Data Types

**Problem:** Array values are small, but using long long

\`\`\`cpp
// Wasteful:
long long arr[1000000];  // 8 MB

// If values fit in int (≤ 2×10⁹):
int arr[1000000];        // 4 MB (50% savings!)

// If values fit in short (≤ 32767):
short arr[1000000];      // 2 MB (75% savings!)

// If boolean:
bool arr[1000000];       // 1 MB (87.5% savings!)
// Or even better:
bitset<1000000> arr;     // 125 KB (98.4% savings!)
\`\`\`

**When to use:**
- Know maximum values
- Values fit in smaller type
- Memory is tight

**Example:**
\`\`\`cpp
// Problem: Count frequency of ages (0-120)
// Wasteful:
long long age_count[1000000];  // 8 MB

// Optimized:
short age_count[121];  // 242 bytes!
\`\`\`

### Technique 2: Rolling Array / Space Optimization

**Problem:** DP uses 2D array, but only need previous row

#### Before Optimization

\`\`\`cpp
// Knapsack DP: dp[i][w] = max value using first i items, weight ≤ w
int n = 1000, W = 1000000;
int dp[1000][1000000];  // 4 GB! MLE!

for (int i = 0; i < n; i++) {
    for (int w = 0; w <= W; w++) {
        dp[i][w] = // ... depends on dp[i-1][...]
    }
}
\`\`\`

#### After Optimization (Rolling Array)

\`\`\`cpp
int n = 1000, W = 1000000;
int dp[2][1000000];  // Only 8 MB! 500x reduction!

for (int i = 0; i < n; i++) {
    int curr = i & 1;      // Current row (0 or 1)
    int prev = 1 - curr;   // Previous row
    
    for (int w = 0; w <= W; w++) {
        dp[curr][w] = // ... depends on dp[prev][...]
    }
}

// Answer is in dp[(n-1) & 1][W]
\`\`\`

#### Even Better: Single Array

\`\`\`cpp
int dp[1000000];  // Only 4 MB!

for (int i = 0; i < n; i++) {
    for (int w = W; w >= weight[i]; w--) {  // Iterate backwards!
        dp[w] = max(dp[w], dp[w - weight[i]] + value[i]);
    }
}
\`\`\`

**Key insight:** Iterating backwards allows in-place updates!

### Technique 3: Coordinate Compression

**Problem:** Values are sparse, but range is huge

\`\`\`cpp
// Input: coordinates in range [0, 10⁹]
// Only N ≤ 100,000 coordinates used

// Wasteful:
int arr[1000000001];  // 4 GB! Impossible!

// Optimized: Compress coordinates to [0, N-1]
vector<int> coords = {5, 1000000, 42, 999999999};
sort(coords.begin(), coords.end());
coords.erase(unique(coords.begin(), coords.end()), coords.end());

// Now map original → compressed:
// 5 → 0
// 42 → 1
// 1000000 → 2
// 999999999 → 3

int arr[4];  // Only 16 bytes!
\`\`\`

**Full example:**
\`\`\`cpp
int n;
cin >> n;
vector<int> a(n);
for (int& x : a) cin >> x;

// Coordinate compression
vector<int> sorted = a;
sort(sorted.begin(), sorted.end());
sorted.erase(unique(sorted.begin(), sorted.end()), sorted.end());

// Compress
for (int& x : a) {
    x = lower_bound(sorted.begin(), sorted.end(), x) - sorted.begin();
}

// Now a[i] is in range [0, unique_count)
int freq[n];  // Much smaller!
\`\`\`

### Technique 4: Boolean Packing (bitset)

**Problem:** Large boolean array

\`\`\`cpp
// Wasteful:
bool visited[10000000];  // 10 MB

// Optimized:
bitset<10000000> visited;  // 1.25 MB (8x reduction!)
\`\`\`

**Why bitset is better:**
- 1 bit per boolean vs 1 byte
- Fast bitwise operations
- Many optimizations

**Example:**
\`\`\`cpp
bitset<100000> is_prime;
is_prime.set();  // Set all to 1

// Sieve of Eratosthenes
is_prime[0] = is_prime[1] = 0;
for (int i = 2; i * i < 100000; i++) {
    if (is_prime[i]) {
        for (int j = i * i; j < 100000; j += i) {
            is_prime[j] = 0;
        }
    }
}

cout << is_prime.count() << endl;  // Count primes efficiently
\`\`\`

### Technique 5: Reuse Memory

**Problem:** Multiple arrays needed, but not simultaneously

\`\`\`cpp
// Wasteful:
int arr1[1000000];  // 4 MB
int arr2[1000000];  // 4 MB
int arr3[1000000];  // 4 MB
// Total: 12 MB

// Optimized (if used at different times):
int arr[1000000];  // 4 MB total!

// Use as arr1
for (int i = 0; i < n; i++) {
    arr[i] = ...;
}
// Process...

// Reuse as arr2
for (int i = 0; i < n; i++) {
    arr[i] = ...;
}
// Process...
\`\`\`

### Technique 6: In-Place Algorithms

**Problem:** Creating copy of array

\`\`\`cpp
// Wasteful:
vector<int> reverse(const vector<int>& arr) {
    vector<int> result = arr;  // Copy! 2x memory
    reverse(result.begin(), result.end());
    return result;
}

// Optimized:
void reverseInPlace(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n / 2; i++) {
        swap(arr[i], arr[n - 1 - i]);
    }
}
\`\`\`

---

## Common Memory Issues and Solutions

### Issue 1: Stack Overflow

#### Symptom
\`\`\`
Segmentation fault (core dumped)
Runtime Error
\`\`\`

#### Cause
\`\`\`cpp
int main() {
    int arr[10000000];  // 40 MB on stack - BOOM!
    return 0;
}
\`\`\`

#### Solutions

**Solution 1: Make it global**
\`\`\`cpp
int arr[10000000];  // Global - in data segment

int main() {
    // Use arr
    return 0;
}
\`\`\`

**Solution 2: Use vector**
\`\`\`cpp
int main() {
    vector<int> arr(10000000);  // Heap allocation
    return 0;
}
\`\`\`

**Solution 3: Dynamic allocation**
\`\`\`cpp
int main() {
    int* arr = new int[10000000];
    // Use arr...
    delete[] arr;
    return 0;
}
\`\`\`

**Solution 4: Increase stack size (Linux)**
\`\`\`bash
ulimit -s unlimited  # Unlimited stack
ulimit -s 65536      # 64 MB stack
\`\`\`

### Issue 2: Memory Limit Exceeded

#### Symptom
\`\`\`
Memory Limit Exceeded (MLE)
Runtime Error: Memory limit exceeded
\`\`\`

#### Cause
\`\`\`cpp
// Total memory > 256 MB
int dp[10000][10000];      // 400 MB - exceeds limit!
vector<int> extra(1000000); // + more memory
\`\`\`

#### Solutions

**Solution 1: Optimize space complexity**
\`\`\`cpp
// Instead of O(N²) space:
int dp[10000][10000];

// Use O(N) with rolling array:
int dp[2][10000];
\`\`\`

**Solution 2: Use smaller types**
\`\`\`cpp
// Instead of:
long long dp[10000][10000];  // 800 MB

// Use if values fit:
int dp[10000][10000];  // 400 MB
\`\`\`

**Solution 3: Bitset for booleans**
\`\`\`cpp
// Instead of:
bool visited[10000000];  // 10 MB

// Use:
bitset<10000000> visited;  // 1.25 MB
\`\`\`

**Solution 4: Reuse memory**
\`\`\`cpp
// Instead of multiple arrays:
int temp1[1000000];
int temp2[1000000];

// Use one:
int temp[1000000];
// Use for different purposes at different times
\`\`\`

### Issue 3: Recursive Stack Overflow

#### Symptom
\`\`\`
Segmentation fault during recursion
Stack overflow in DFS/BFS
\`\`\`

#### Cause
\`\`\`cpp
void dfs(int node) {
    int temp[100000];  // 400 KB per call!
    // ...
    for (int child : adj[node]) {
        dfs(child);  // Each call adds 400 KB
    }
}
// After 20 recursive calls: 8 MB stack exhausted!
\`\`\`

#### Solutions

**Solution 1: Make local arrays global**
\`\`\`cpp
int temp[100000];  // Global

void dfs(int node) {
    // Use global temp
    for (int child : adj[node]) {
        dfs(child);
    }
}
\`\`\`

**Solution 2: Use iterative approach**
\`\`\`cpp
void dfs(int start) {
    stack<int> st;
    st.push(start);
    
    while (!st.empty()) {
        int node = st.top();
        st.pop();
        
        // Process node
        for (int child : adj[node]) {
            st.push(child);
        }
    }
}
\`\`\`

**Solution 3: Tail recursion optimization**
\`\`\`cpp
// Compiler can optimize tail recursion
void dfs(int node) {
    // Process node...
    
    if (base_case) return;
    
    dfs(next_node);  // Tail recursion - no stack growth
}
\`\`\`

---

## Memory Management Best Practices

### ✅ DO:

1. **Calculate memory before coding**
   \`\`\`
   N × M × sizeof(type) ≤ 256 MB?
   \`\`\`

2. **Use global arrays for large data**
   \`\`\`cpp
   int arr[1000000];  // Global
   \`\`\`

3. **Prefer vectors for dynamic size**
   \`\`\`cpp
   vector<int> v(n);
   \`\`\`

4. **Use smallest type that fits**
   \`\`\`cpp
   short vs int vs long long
   \`\`\`

5. **Optimize space when memory is tight**
   \`\`\`cpp
   Rolling array, bitset, etc.
   \`\`\`

6. **Free memory when done**
   \`\`\`cpp
   delete[] arr;
   v.clear();
   v.shrink_to_fit();
   \`\`\`

### ❌ DON'T:

1. **Large local arrays**
   \`\`\`cpp
   int main() {
       int arr[1000000];  // Stack overflow!
   }
   \`\`\`

2. **Allocate more than needed**
   \`\`\`cpp
   int arr[1000000];  // Only use 100 elements - waste!
   \`\`\`

3. **Memory leaks**
   \`\`\`cpp
   int* arr = new int[n];
   // Forgot delete[]!
   \`\`\`

4. **Unnecessary copies**
   \`\`\`cpp
   void process(vector<int> v) {  // Copy!
       // Use const vector<int>& v
   }
   \`\`\`

5. **Wrong size calculations**
   \`\`\`cpp
   // Forgot it's 2D:
   int arr[5000][5000];  // 100 MB, not 20 KB!
   \`\`\`

---

## Advanced Memory Techniques

### Custom Allocators

For advanced users, custom allocators can improve performance:

\`\`\`cpp
// Pool allocator for frequent small allocations
char pool[10000000];
int pool_ptr = 0;

void* allocate(size_t size) {
    void* ptr = pool + pool_ptr;
    pool_ptr += size;
    return ptr;
}

// Use for tree nodes, etc.
struct Node {
    int value;
    Node *left, *right;
    
    void* operator new(size_t size) {
        return allocate(size);
    }
};
\`\`\`

### Memory Pools

\`\`\`cpp
template<typename T, int SIZE>
class MemoryPool {
    T pool[SIZE];
    int used = 0;
    
public:
    T* allocate() {
        return &pool[used++];
    }
    
    void reset() {
        used = 0;
    }
};

MemoryPool<int, 1000000> int_pool;
int* p = int_pool.allocate();
\`\`\`

---

## Debugging Memory Issues

### Tool 1: Print Memory Usage

\`\`\`cpp
#include <sys/resource.h>

void printMemoryUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    cout << "Memory used: " << usage.ru_maxrss << " KB\\n";
}
\`\`\`

### Tool 2: Valgrind (Linux)

\`\`\`bash
valgrind --leak-check=full ./solution
\`\`\`

### Tool 3: AddressSanitizer

\`\`\`bash
g++ -fsanitize=address -g solution.cpp -o solution
./solution
\`\`\`

---

## Memory Optimization Checklist

Before submitting, check:

✅ **Calculated total memory?**
✅ **Used smallest type possible?**
✅ **No large local arrays?**
✅ **Optimized space complexity if needed?**
✅ **Used bitset for booleans if applicable?**
✅ **Reused memory where possible?**
✅ **No memory leaks?**

---

## Summary

**Key Concepts:**

✅ **Stack**: Fast, limited (~8 MB), automatic
✅ **Heap**: Large, slower, manual/vector
✅ **Global**: Data segment, not stack, can be huge
✅ **Memory Limits**: Usually 256 MB effective
✅ **Calculate**: N × M × sizeof(type) before coding

**Optimization Techniques:**

✅ **Smaller types**: short vs int vs long long
✅ **Rolling array**: 2D DP → 1D
✅ **Bitset**: Boolean compression
✅ **Coordinate compression**: Sparse data
✅ **In-place**: Avoid copies

**Common Issues:**

✅ **Stack overflow**: Use global/vector
✅ **MLE**: Optimize space complexity
✅ **Recursive overflow**: Make locals global

**Next Steps:**

In the next section, we'll explore **Template Metaprogramming Basics** for writing generic, reusable competitive programming code!

**Key Takeaway**: Memory management is as important as time complexity. Always calculate memory usage before implementing. A simple calculation can save you from MLE frustration and help you choose the right optimization strategy!
`,
  quizId: 'cp-m1-s10-quiz',
  discussionId: 'cp-m1-s10-discussion',
} as const;
