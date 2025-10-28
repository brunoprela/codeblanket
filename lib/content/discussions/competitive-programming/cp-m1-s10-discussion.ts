export default {
    id: 'cp-m1-s10-discussion',
    title: 'Memory Management for CP - Discussion Questions',
    questions: [
        {
            question: 'Explain the difference between stack, heap, and global/static memory allocation in C++. When should you use each in competitive programming, and what are the common pitfalls that lead to stack overflow or memory limit exceeded?',
            answer: `Understanding memory segments is crucial for avoiding runtime errors. Here's the complete picture:

**The Three Memory Regions:**

**1. Stack (Local Variables):**

Characteristics:
- Fast allocation/deallocation (just move pointer)
- Limited size (~1-8 MB typically)
- Automatic cleanup (LIFO)
- Function call overhead

Example:
\`\`\`cpp
void function() {
    int x = 5;              // Stack
    int arr[100];           // Stack
    double d;               // Stack
    // All freed when function returns
}
\`\`\`

**When to use:**
✅ Small local variables
✅ Function parameters
✅ Small arrays (<10KB)
✅ Temporary data

**Pitfalls:**
❌ Large local arrays
\`\`\`cpp
int main() {
    int arr[1000000];  // 4MB - Stack overflow!
    return 0;
}
\`\`\`

❌ Deep recursion with local data
\`\`\`cpp
void dfs(int node) {
    int temp[100000];  // 400KB per call
    dfs(child);  // Stack overflow after ~20 calls
}
\`\`\`

**2. Heap (Dynamic Memory):**

Characteristics:
- Slower allocation (complex algorithm)
- Large size (up to memory limit)
- Manual or automatic management
- Can fragment

Example:
\`\`\`cpp
int* ptr = new int[1000000];  // Heap
vector<int> v(1000000);       // Heap (internally)

delete[] ptr;  // Manual cleanup
// Vector auto-cleans in destructor
\`\`\`

**When to use:**
✅ Dynamic size (unknown at compile time)
✅ Large data structures
✅ Need to return from function
✅ Variable lifetime

**Best practice in CP:** Use vectors!
\`\`\`cpp
int n;
cin >> n;
vector<int> arr(n);  // Heap, automatic cleanup
\`\`\`

**Pitfalls:**
❌ Memory leaks
\`\`\`cpp
void leak() {
    int* arr = new int[1000000];
    // Forgot delete[]!
}  // Memory leaked
\`\`\`

❌ Using after delete
\`\`\`cpp
int* ptr = new int[10];
delete[] ptr;
ptr[0] = 5;  // Undefined behavior!
\`\`\`

**3. Global/Static (Data Segment):**

Characteristics:
- Allocated at program start
- Lives entire program
- NOT on stack!
- Zero-initialized by default
- Can be very large

Example:
\`\`\`cpp
int globalArr[1000000];  // Data segment, NOT stack!
static int staticArr[1000000];  // Also data segment

int main() {
    // Both arrays are 0-initialized
    // No stack space used
}
\`\`\`

**When to use:**
✅ Large fixed-size arrays
✅ DP tables
✅ Precalculated data
✅ When stack would overflow

**Why it works:**
\`\`\`cpp
// WRONG: Stack overflow
int main() {
    int dp[3000][3000];  // 36MB on stack - BOOM!
}

// RIGHT: Data segment
int dp[3000][3000];  // Global - 36MB OK!

int main() {
    // Use dp here
}
\`\`\`

**Pitfalls:**
❌ Not reinitializing between test cases
\`\`\`cpp
int visited[100000];  // Global

void solve() {
    // Forgot to reset visited!
    // Previous test case data still there
}
\`\`\`

Fix:
\`\`\`cpp
void solve() {
    memset(visited, 0, sizeof(visited));
    // Or: fill(visited, visited + n, 0);
}
\`\`\`

**Complete Comparison:**

| Feature | Stack | Heap | Global/Static |
|---------|-------|------|---------------|
| Size | ~8MB | Up to limit | Up to limit |
| Speed | Fastest | Slower | Fast |
| Lifetime | Function scope | Manual/RAII | Program life |
| Initialization | Undefined | Undefined | Zero |
| Good for | Small, temp | Dynamic size | Large, fixed |

**Memory Limits in CP:**

Typical: 256MB

What fits:
\`\`\`cpp
// 256 MB = 256 * 1024 * 1024 bytes = 268,435,456 bytes

// Int array (4 bytes each):
int arr[67108864];  // 256MB - maximum

// 2D array:
int grid[5000][5000];  // 100MB - OK
int grid[10000][10000];  // 400MB - Too much!

// Long long (8 bytes):
long long arr[33554432];  // 256MB - maximum
\`\`\`

**Calculating Memory:**

Formula: \`elements × bytes_per_element\`

Examples:
\`\`\`cpp
// 1D array
int arr[1000000];  // 1M × 4 = 4MB ✅

// 2D array
int dp[1000][1000];  // 1M × 4 = 4MB ✅
int dp[5000][5000];  // 25M × 4 = 100MB ✅
int dp[10000][10000];  // 100M × 4 = 400MB ❌

// Vector of vectors
vector<vector<int>> grid(1000, vector<int>(1000));  // 4MB ✅
\`\`\`

**Common Scenarios:**

**Scenario 1: Large DP Table**

Problem: Need dp[5000][5000]

❌ **Wrong:**
\`\`\`cpp
int main() {
    int dp[5000][5000];  // 100MB on stack - overflow!
}
\`\`\`

✅ **Right:**
\`\`\`cpp
int dp[5000][5000];  // Global - OK!

int main() {
    // Use dp
}
\`\`\`

Or use vector:
\`\`\`cpp
int main() {
    vector<vector<int>> dp(5000, vector<int>(5000));  // Heap - OK!
}
\`\`\`

**Scenario 2: Dynamic Size**

Problem: Size n from input

❌ **Wrong:**
\`\`\`cpp
int n;
cin >> n;
int arr[n];  // VLA - not standard C++!
\`\`\`

✅ **Right:**
\`\`\`cpp
int n;
cin >> n;
vector<int> arr(n);  // Vector - standard and safe
\`\`\`

Or use global with max size:
\`\`\`cpp
const int MAXN = 100000;
int arr[MAXN];  // Global

int main() {
    int n;
    cin >> n;
    // Use arr[0] to arr[n-1]
}
\`\`\`

**Scenario 3: Recursion with Large Data**

❌ **Wrong:**
\`\`\`cpp
void dfs(int node) {
    int temp[100000];  // Stack overflow with deep recursion
    // ...
    dfs(child);
}
\`\`\`

✅ **Right:**
\`\`\`cpp
int temp[100000];  // Global

void dfs(int node) {
    // Use global temp
    dfs(child);
}
\`\`\`

**Memory Optimization Techniques:**

**1. Rolling Array:**
\`\`\`cpp
// Instead of dp[1000][1000000] = 4GB
int dp[2][1000000];  // Only 8MB!

for(int i = 0; i < n; i++) {
    int curr = i & 1;
    int prev = 1 - curr;
    for(int j = 0; j < m; j++) {
        dp[curr][j] = dp[prev][j] + ...;
    }
}
\`\`\`

**2. Smaller Data Types:**
\`\`\`cpp
// Instead of:
long long arr[10000000];  // 80MB

// Use if values fit:
int arr[10000000];  // 40MB
short arr[10000000];  // 20MB
\`\`\`

**3. Bitset:**
\`\`\`cpp
// Instead of:
bool visited[10000000];  // 10MB

// Use:
bitset<10000000> visited;  // 1.25MB (8x smaller!)
\`\`\`

**Detecting Memory Issues:**

**Stack Overflow:**
Symptoms:
- Segmentation fault
- Runtime error
- Works with small input, fails with large

Detection:
\`\`\`bash
# Linux: Check stack size
ulimit -s  # In KB

# Increase stack size (for testing)
ulimit -s unlimited
\`\`\`

**Memory Limit Exceeded:**
Symptoms:
- MLE verdict
- Works locally, fails on judge

Detection:
\`\`\`cpp
// Calculate before coding:
// n × m × sizeof(type) ≤ 256MB?

// Example:
// dp[10000][10000] with int
// = 10000 × 10000 × 4
// = 400MB > 256MB  // Won't work!
\`\`\`

**Best Practices:**

1. **Calculate memory first**
   \`\`\`
   Total = elements × bytes_per_element
   Must be ≤ memory limit
            \`\`\`

2. **Use appropriate storage**
   - Small local: Stack
   - Dynamic: Vector (heap)
   - Large fixed: Global

3. **Initialize global arrays**
   \`\`\`cpp
   int arr[MAXN];  // Zeros by default

        void solve() {
            // But reset between tests!
            memset(arr, 0, sizeof(arr));
        }
            \`\`\`

4. **Avoid memory leaks**
   \`\`\`cpp
   // Prefer:
   vector < int > v(n);  // Auto-cleanup

        // Over:
        int * arr = new int[n];  // Must delete[]
        \`\`\`

5. **Check constraints**
   \`\`\`
   If n ≤ 100: Can use O(n³) and any structure
   If n ≤ 10⁵: Need O(n log n), careful with 2D arrays
   If n ≤ 10⁶: Need O(n), 1D arrays only
   \`\`\`

**Quick Reference:**

**Stack overflow? Make it global!**
\`\`\`cpp
// From:
int main() { int arr[1000000]; }

// To:
int arr[1000000];
int main() { /* use arr */ }
\`\`\`

**Need dynamic size? Use vector!**
\`\`\`cpp
vector<int> v(n);
\`\`\`

**MLE? Optimize space!**
- Rolling array
- Smaller types
- Bitset

**Bottom Line:**

Memory management in CP:
- **Stack:** Small, fast, automatic (but limited)
- **Heap:** Large, flexible, use vectors
- **Global:** Large, fast, perfect for DP tables

Always calculate memory before coding!`,
    },
{
    question: 'Memory optimization is often necessary in competitive programming. Describe specific techniques like rolling arrays, space-efficient data structures, and coordinate compression that can reduce memory usage while maintaining correctness.',
        answer: `Memory optimization can turn MLE into AC. Here are the essential techniques:

**Technique 1: Rolling Array (Space Reduction)**

**Problem:** 2D DP where each row only depends on previous row

**Before (O(n×m) space):**
\`\`\`cpp
int n = 1000, m = 1000000;
int dp[1000][1000000];  // 4GB - MLE!

for(int i = 0; i < n; i++) {
    for(int j = 0; j < m; j++) {
        dp[i][j] = dp[i-1][j] + ...;  // Depends on previous row
    }
}
\`\`\`

**After (O(2×m) space):**
\`\`\`cpp
int dp[2][1000000];  // Only 8MB!

for(int i = 0; i < n; i++) {
    int curr = i & 1;      // Current row (0 or 1)
    int prev = 1 - curr;   // Previous row
    
    for(int j = 0; j < m; j++) {
        dp[curr][j] = dp[prev][j] + ...;
    }
}
// Answer in dp[(n-1) & 1][...]
\`\`\`

**Savings:** 1000×1000000 → 2×1000000 (500x reduction!)

**Even Better (O(m) space):**
\`\`\`cpp
int dp[1000000];  // Just 4MB!

for(int i = 0; i < n; i++) {
    for(int j = m-1; j >= 0; j--) {  // Backwards!
        dp[j] = dp[j] + ...;  // Update in-place
    }
}
\`\`\`

**When applicable:**
- DP where dp[i] only depends on dp[i-1]
- Knapsack, longest common subsequence, etc.

**Technique 2: Coordinate Compression**

**Problem:** Large coordinate range, but few actual values

**Before (O(max_value) space):**
\`\`\`cpp
// Values up to 10⁹, but only 10⁵ values
int freq[1000000001];  // 4GB - impossible!
\`\`\`

**After (O(n) space):**
\`\`\`cpp
vector<int> coords = {5, 1000000, 42, 999999999};

// Step 1: Sort and remove duplicates
sort(coords.begin(), coords.end());
coords.erase(unique(coords.begin(), coords.end()), coords.end());

// Step 2: Map original → compressed
map<int, int> compressed;
for(int i = 0; i < coords.size(); i++) {
    compressed[coords[i]] = i;
}

// Now use compressed coordinates
int freq[coords.size()];  // Only as big as needed!

// Usage:
for(int x : original_values) {
    freq[compressed[x]]++;
}
\`\`\`

**Complete example:**
\`\`\`cpp
vector<int> compress(vector<int>& arr) {
    vector<int> sorted = arr;
    sort(sorted.begin(), sorted.end());
    sorted.erase(unique(sorted.begin(), sorted.end()), sorted.end());
    
    for(int& x : arr) {
        x = lower_bound(sorted.begin(), sorted.end(), x) - sorted.begin();
    }
    
    return arr;  // Now values are 0, 1, 2, ..., unique_count-1
}
\`\`\`

**When applicable:**
- Sparse data (few values in large range)
- Coordinate-based problems
- Range queries with few distinct values

**Technique 3: Bitset (Boolean Compression)**

**Before (1 byte per boolean):**
\`\`\`cpp
bool visited[10000000];  // 10MB
\`\`\`

**After (1 bit per boolean):**
\`\`\`cpp
bitset<10000000> visited;  // 1.25MB (8x smaller!)

// Usage same as array:
visited[i] = 1;
if(visited[i]) { ... }
visited.reset();  // All zeros
visited.set();    // All ones
\`\`\`

**Bonus features:**
\`\`\`cpp
bitset<100> b1, b2;
b1 &= b2;  // Bitwise AND
b1 |= b2;  // Bitwise OR
b1 ^= b2;  // Bitwise XOR
cout << b1.count();  // Count set bits
\`\`\`

**When applicable:**
- Boolean arrays
- Sieve of Eratosthenes
- Bit manipulation problems
- Graph visited arrays

**Technique 4: Smaller Data Types**

**Before:**
\`\`\`cpp
long long dp[10000000];  // 80MB
\`\`\`

**After (if values fit):**
\`\`\`cpp
int dp[10000000];    // 40MB (if ≤ 2×10⁹)
short dp[10000000];  // 20MB (if ≤ 32767)
char dp[10000000];   // 10MB (if ≤ 127)
\`\`\`

**Check what fits:**
\`\`\`
char:      -128 to 127
short:     -32,768 to 32,767
int:       -2×10⁹ to 2×10⁹
long long: -9×10¹⁸ to 9×10¹⁸
\`\`\`

**Technique 5: Reuse Memory**

**Before:**
\`\`\`cpp
int temp1[1000000];  // 4MB
int temp2[1000000];  // 4MB
int temp3[1000000];  // 4MB
// Total: 12MB
\`\`\`

**After (if not used simultaneously):**
\`\`\`cpp
int temp[1000000];  // 4MB total

// Use as temp1
for(int i = 0; i < n; i++) {
    temp[i] = ...;
}
process(temp);

// Reuse as temp2
for(int i = 0; i < n; i++) {
    temp[i] = ...;
}
process(temp);
\`\`\`

**Technique 6: Implicit State Storage**

**Before:**
\`\`\`cpp
// Store parent explicitly
int parent[100000];
\`\`\`

**After (if order matters):**
\`\`\`cpp
// Parent = i-1 implicitly
// No storage needed!
\`\`\`

**Example: BFS levels**
\`\`\`cpp
// Instead of storing level for each node:
int level[100000];

// Process by layer:
queue<int> q;
q.push(start);
int currentLevel = 0;

while(!q.empty()) {
    int size = q.size();
    for(int i = 0; i < size; i++) {
        int node = q.front();
        q.pop();
        // Process node at currentLevel
        // No need to store level[node]!
    }
    currentLevel++;
}
\`\`\`

**Technique 7: On-the-Fly Computation**

**Before (precompute and store):**
\`\`\`cpp
int factorial[1000000];
for(int i = 0; i < 1000000; i++) {
    factorial[i] = ...;  // 4MB
}
\`\`\`

**After (compute when needed):**
\`\`\`cpp
long long factorial(int n) {
    long long result = 1;
    for(int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}
// No storage needed!
\`\`\`

**Trade-off:** Time vs Space
- Use when computation is cheap
- Avoid if computed many times

**Technique 8: Sparse Matrix Optimization**

**Before (dense matrix):**
\`\`\`cpp
int adj[10000][10000];  // 400MB - most are 0!
\`\`\`

**After (adjacency list):**
\`\`\`cpp
vector<int> adj[10000];  // Only stores edges

// Memory: O(V + E) instead of O(V²)
// For sparse graph with E = 50000:
// Old: 400MB
// New: ~1MB
\`\`\`

**Technique 9: Delta Encoding**

**Before (store absolute values):**
\`\`\`cpp
int values[1000000];  // Each int is 4 bytes
// If values are 1000000, 1000001, 1000002, ...
\`\`\`

**After (store differences):**
\`\`\`cpp
int base = 1000000;
char delta[1000000];  // Each char is 1 byte
// Store: 0, +1, +1, +1, ...
// Reconstruct: base + delta[i]
\`\`\`

**Savings:** 4x if deltas fit in char

**Technique 10: Lazy Allocation**

**Before (allocate everything):**
\`\`\`cpp
vector<vector<int>> dp(1000000, vector<int>(100));
// Allocates 400MB even if only use 10%
\`\`\`

**After (allocate on demand):**
\`\`\`cpp
map<int, vector<int>> dp;  // Only allocate when needed

void use(int i) {
    if(!dp.count(i)) {
        dp[i] = vector<int>(100);
    }
    // Use dp[i]
}
\`\`\`

**Real-World Examples:**

**Example 1: Knapsack DP**

O(n×W) → O(W):
\`\`\`cpp
// Before: dp[1000][1000000] = 4GB
// After:  dp[1000000] = 4MB

vector<int> dp(W + 1, 0);
for(int i = 0; i < n; i++) {
    for(int w = W; w >= weight[i]; w--) {  // Backwards!
        dp[w] = max(dp[w], dp[w - weight[i]] + value[i]);
    }
}
\`\`\`

**Example 2: Longest Common Subsequence**

O(n×m) → O(min(n,m)):
\`\`\`cpp
int lcs(string& s, string& t) {
    int n = s.size(), m = t.size();
    vector<int> dp(m + 1, 0);
    
    for(int i = 1; i <= n; i++) {
        int prev = 0;
        for(int j = 1; j <= m; j++) {
            int temp = dp[j];
            if(s[i-1] == t[j-1]) {
                dp[j] = prev + 1;
            } else {
                dp[j] = max(dp[j], dp[j-1]);
            }
            prev = temp;
        }
    }
    
    return dp[m];
}
\`\`\`

**Memory Optimization Checklist:**

Before coding:
✅ Calculate memory: n × m × sizeof(type)
✅ Will it fit in memory limit?
✅ Can use rolling array?
✅ Can compress coordinates?
✅ Can use smaller types?
✅ Can compute on-the-fly?

**Bottom Line:**

Memory optimization techniques:
1. **Rolling Array** - 2D → 1D DP
2. **Coordinate Compression** - Sparse data
3. **Bitset** - Boolean arrays
4. **Smaller Types** - When values fit
5. **Reuse Memory** - Non-overlapping use
6. **Implicit Storage** - Don't store what can be inferred
7. **On-the-Fly** - Compute vs store
8. **Sparse Structures** - Adjacency list vs matrix
9. **Delta Encoding** - Store differences
10. **Lazy Allocation** - Allocate on demand

Often one technique is enough to pass! Calculate first, optimize when needed.`,
    },
{
    question: 'Discuss the practical implications of using global vs local arrays in competitive programming, including initialization behavior, memory limits, and how this affects code structure and debugging.',
        answer: `Global vs local arrays is a critical decision in CP. Here's the complete analysis:

**Global Arrays:**

Declaration:
\`\`\`cpp
int arr[1000000];  // Global/static storage

int main() {
    // Use arr here
}
\`\`\`

**Key Properties:**

**1. Memory Location:**
- Stored in data/BSS segment (NOT stack)
- Can be very large (up to memory limit)
- Won't cause stack overflow

**2. Zero Initialization:**
\`\`\`cpp
int arr[1000000];  // All zeros by default
bool visited[100000];  // All false

int main() {
    // arr[i] is already 0 for all i
    // No manual initialization needed
}
\`\`\`

**3. Lifetime:**
- Lives entire program
- Allocated at program start
- Freed at program end

**4. Scope:**
- Accessible from anywhere (if not in namespace/class)
- Can be used by multiple functions

**Advantages:**

✅ **No Stack Overflow**
\`\`\`cpp
int dp[5000][5000];  // 100MB - OK globally, crash if local
\`\`\`

✅ **Zero Initialized**
\`\`\`cpp
int cnt[100000];  // All zeros automatically
\`\`\`

✅ **Easy Sharing**
\`\`\`cpp
int arr[MAXN];
int n;

void process() {
    // Can access arr directly
}

int main() {
    cin >> n;
    process();
}
\`\`\`

✅ **Fast Access**
- No function parameter passing
- No return value copying

**Disadvantages:**

❌ **Must Reset Between Tests**
\`\`\`cpp
int visited[100000];

void solve() {
    // visited still has data from previous test!
    // Must reset:
    memset(visited, 0, sizeof(visited));
}

int main() {
    int t;
    cin >> t;
    while(t--) solve();
}
\`\`\`

❌ **Less Modular**
\`\`\`cpp
// Hard to reuse functions with different arrays
int arr1[MAXN];
int arr2[MAXN];

void process() {
    // Can only work with arr1, not arr2
}
\`\`\`

❌ **Namespace Pollution**
\`\`\`cpp
int n, m, k;  // Which variable is which?
// Easy to mix up or overwrite
\`\`\`

❌ **Debugging Harder**
\`\`\`cpp
// Can't inspect easily in debugger
// Values persist between runs (if not reset)
\`\`\`

**Local Arrays:**

Declaration:
\`\`\`cpp
int main() {
    int arr[100];  // Local (stack)
    // ...
}
\`\`\`

**Key Properties:**

**1. Memory Location:**
- Stored on stack
- Limited size (~1-8 MB)
- WILL cause stack overflow if too large

**2. Uninitialized:**
\`\`\`cpp
int main() {
    int arr[10];
    cout << arr[0];  // Garbage value!
}
\`\`\`

**3. Lifetime:**
- Created when entering scope
- Destroyed when leaving scope

**4. Scope:**
- Only accessible in that function
- Can't be shared easily

**Advantages:**

✅ **Automatic Cleanup**
\`\`\`cpp
void function() {
    int temp[100];
    // ...
}  // temp automatically freed
\`\`\`

✅ **Fresh Every Call**
\`\`\`cpp
void solve() {
    int visited[100];
    // Always fresh, no reset needed
}
\`\`\`

✅ **Better Encapsulation**
\`\`\`cpp
void process(int arr[], int n) {
    // Works with any array passed
}
\`\`\`

✅ **Thread-Safe**
- Each thread has own stack
- No race conditions

**Disadvantages:**

❌ **Stack Overflow Risk**
\`\`\`cpp
int main() {
    int arr[1000000];  // 4MB - might overflow!
}
\`\`\`

❌ **Not Zero-Initialized**
\`\`\`cpp
int main() {
    int arr[10];
    // arr[i] = garbage!
    
    // Must initialize:
    memset(arr, 0, sizeof(arr));
    // Or:
    int arr[10] = {};  // C++11
}
\`\`\`

❌ **Can't Return**
\`\`\`cpp
int* getArray() {
    int arr[100];
    return arr;  // Undefined behavior!
}
\`\`\`

**Practical Guidelines:**

**Use Global When:**

1. **Large Arrays:**
\`\`\`cpp
int dp[5000][5000];  // Global
int visited[1000000];  // Global
\`\`\`

2. **Multiple Functions Need Access:**
\`\`\`cpp
int adj[MAXN][MAXN];

void dfs(int u) { /* uses adj */ }
void bfs(int u) { /* uses adj */ }
\`\`\`

3. **CP Contests (Speed Matters):**
\`\`\`cpp
// Fast to write, no passing parameters
int arr[MAXN];
int n;

void solve() {
    cin >> n;
    // Use arr
}
\`\`\`

**Use Local When:**

1. **Small Arrays:**
\`\`\`cpp
int main() {
    int dx[] = {0, 1, 0, -1};  // Small, local OK
}
\`\`\`

2. **Temporary Data:**
\`\`\`cpp
void process() {
    int temp[100];  // Only needed here
    // ...
}
\`\`\`

3. **Multiple Test Cases (No Reset):**
\`\`\`cpp
void solve() {
    int arr[100];  // Fresh every test
}
\`\`\`

**Hybrid Approach:**

\`\`\`cpp
const int MAXN = 100005;
int arr[MAXN];  // Global for size

int main() {
    int n;
    cin >> n;
    
    // Use arr[0] to arr[n-1]
    // But pretend it's size n
}
\`\`\`

**Common Patterns:**

**Pattern 1: Graph Adjacency**
\`\`\`cpp
vector<int> adj[MAXN];  // Global
bool visited[MAXN];     // Global

void dfs(int u) {
    visited[u] = true;
    for(int v : adj[u]) {
        if(!visited[v]) dfs(v);
    }
}
\`\`\`

**Pattern 2: DP Table**
\`\`\`cpp
int dp[1005][1005];  // Global

int main() {
    int n, m;
    cin >> n >> m;
    
    // Use dp[0..n][0..m]
}
\`\`\`

**Pattern 3: Multiple Tests with Reset**
\`\`\`cpp
int arr[MAXN];

void solve() {
    // Reset at start
    memset(arr, 0, sizeof(arr));
    
    // Or only reset what's needed
    for(int i = 0; i < n; i++) {
        arr[i] = 0;
    }
}
\`\`\`

**Debugging Considerations:**

**Global Arrays:**
\`\`\`cpp
// Print helper
void debugArray(int arr[], int n) {
    for(int i = 0; i < n; i++) {
        cerr << arr[i] << " ";
    }
    cerr << endl;
}

int arr[MAXN];  // Global

int main() {
    debugArray(arr, n);  // Can access globally
}
\`\`\`

**Local Arrays:**
\`\`\`cpp
// Must pass to debug
void debugArray(int arr[], int n) {
    for(int i = 0; i < n; i++) {
        cerr << arr[i] << " ";
    }
    cerr << endl;
}

int main() {
    int arr[100];
    debugArray(arr, 100);  // Must pass explicitly
}
\`\`\`

**Initialization Comparison:**

\`\`\`cpp
// Global: Zeros
int global[10];  // {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

int main() {
    // Local: Garbage
    int local1[10];  // {?, ?, ?, ?, ?, ?, ?, ?, ?, ?}
    
    // Local: Zeros (explicit)
    int local2[10] = {};  // {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    
    // Local: Partial init
    int local3[10] = {1, 2};  // {1, 2, 0, 0, 0, 0, 0, 0, 0, 0}
}
\`\`\`

**Memory Limit Impact:**

Both count toward memory limit:
\`\`\`cpp
// Global:
int global[50000][5000];  // 1GB - counts toward limit

// Local won't even compile (stack overflow)
// So global is necessary for large arrays
\`\`\`

**My Recommendation:**

For CP specifically:

✅ **Use Global For:**
- Large arrays (>10KB)
- DP tables
- Graph structures
- Shared across functions

✅ **Use Local For:**
- Small temp arrays
- Direction vectors
- When fresh-per-call matters

✅ **Use Vector When:**
- Dynamic size
- Need to return from function
- Want automatic memory management

**Bottom Line:**

In competitive programming:
- **Global = Default for large arrays** (no stack overflow, zero-init)
- **Local = For small temporary data** (auto cleanup, fresh)
- **Remember to reset globals between test cases!**

The convenience and safety of global arrays outweighs the minor disadvantages in CP context.`,
    },
  ],
} as const ;

