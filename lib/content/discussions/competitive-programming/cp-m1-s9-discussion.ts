export default {
    id: 'cp-m1-s9-discussion',
    title: 'Bits, Bytes & Bitwise Operations - Discussion Questions',
    questions: [
        {
            question: 'Bitwise operations are fundamental in competitive programming for efficiency and elegance. Explain the key bitwise operations with practical CP examples and discuss when bitwise tricks can replace traditional approaches.',
            answer: `Bitwise operations work directly on binary representations of numbers, offering speed and elegance. Here's everything you need:

**The Six Core Bitwise Operators:**

**1. AND (&) - Both bits must be 1:**
\`\`\`cpp
5 & 3  // 101 & 011 = 001 = 1

// Common uses:
// Check if number is even
if(n & 1) { /* odd */ } 
else { /* even */ }

// Extract specific bits
int lastKBits = n & ((1 << k) - 1);

// Clear specific bits
n = n & ~(1 << i);  // Clear i-th bit
\`\`\`

**2. OR (|) - At least one bit is 1:**
\`\`\`cpp
5 | 3  // 101 | 011 = 111 = 7

// Common uses:
// Set specific bit
n = n | (1 << i);  // Set i-th bit to 1

// Combine flags
int flags = FLAG_A | FLAG_B | FLAG_C;
\`\`\`

**3. XOR (^) - Bits are different:**
\`\`\`cpp
5 ^ 3  // 101 ^ 011 = 110 = 6

// Properties:
// a ^ a = 0
// a ^ 0 = a
// a ^ b ^ b = a

// Common uses:
// Swap without temp variable
a = a ^ b;
b = a ^ b;  // b = (a^b)^b = a
a = a ^ b;  // a = (a^b)^a = b

// Find unique element (all others appear twice)
int unique = 0;
for(int x : arr) unique ^= x;
\`\`\`

**4. NOT (~) - Flip all bits:**
\`\`\`cpp
~5  // ~00000101 = 11111010 = -6 (two's complement)

// Common uses:
// Create mask
int mask = ~0;  // All 1s
int mask = ~(1 << i);  // All 1s except i-th bit
\`\`\`

**5. Left Shift (<<) - Multiply by 2^n:**
\`\`\`cpp
5 << 2  // 101 << 2 = 10100 = 20 (5 * 4)

// Common uses:
// Powers of 2
int powerOf2 = 1 << n;  // 2^n

// Set i-th bit
n |= (1 << i);

// Create masks
int mask = (1 << k) - 1;  // k bits all 1
\`\`\`

**6. Right Shift (>>) - Divide by 2^n:**
\`\`\`cpp
20 >> 2  // 10100 >> 2 = 101 = 5 (20 / 4)

// Common uses:
// Fast division by 2
n = n >> 1;  // Same as n / 2

// Check i-th bit
if((n >> i) & 1) { /* bit is set */ }
\`\`\`

**Practical CP Examples:**

**Example 1: Check if Power of 2**
\`\`\`cpp
// Traditional: O(log n)
bool isPowerOf2_slow(int n) {
    if(n <= 0) return false;
    while(n > 1) {
        if(n % 2 != 0) return false;
        n /= 2;
    }
    return true;
}

// Bitwise: O(1)
bool isPowerOf2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// Why? Power of 2: 1000...0
// n-1:         0111...1
// n & (n-1):   0000...0
\`\`\`

**Example 2: Count Set Bits (Popcount)**
\`\`\`cpp
// Method 1: Loop through bits
int countSetBits1(int n) {
    int count = 0;
    while(n) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}

// Method 2: Brian Kernighan's Algorithm
int countSetBits2(int n) {
    int count = 0;
    while(n) {
        n &= (n - 1);  // Removes rightmost set bit
        count++;
    }
    return count;
}

// Method 3: Built-in (fastest)
int countSetBits3(int n) {
    return __builtin_popcount(n);  // GCC built-in
}
\`\`\`

**Example 3: Generate All Subsets**
\`\`\`cpp
// Generate all 2^n subsets using bitmask
vector<int> arr = {1, 2, 3};
int n = arr.size();

for(int mask = 0; mask < (1 << n); mask++) {
    vector<int> subset;
    for(int i = 0; i < n; i++) {
        if(mask & (1 << i)) {  // Check if i-th bit is set
            subset.push_back(arr[i]);
        }
    }
    // Process subset
    for(int x : subset) cout << x << " ";
    cout << endl;
}

// Output:
// (empty)
// 1
// 2
// 1 2
// 3
// 1 3
// 2 3
// 1 2 3
\`\`\`

**Example 4: Find Missing Number**
\`\`\`cpp
// Array of 1 to n, one number missing
// XOR approach: O(n) time, O(1) space
int findMissing(vector<int>& arr, int n) {
    int xor_all = 0;
    
    // XOR all numbers 1 to n
    for(int i = 1; i <= n; i++) {
        xor_all ^= i;
    }
    
    // XOR with array elements
    for(int x : arr) {
        xor_all ^= x;
    }
    
    // Result is missing number
    // Because x ^ x = 0, only missing remains
    return xor_all;
}
\`\`\`

**Example 5: Swap Two Numbers**
\`\`\`cpp
// Without temp variable
void swap(int& a, int& b) {
    a = a ^ b;
    b = a ^ b;  // b = (a^b)^b = a
    a = a ^ b;  // a = (a^b)^a = b
}

// Or in one line:
a ^= b ^= a ^= b;
\`\`\`

**Example 6: Reverse Bits**
\`\`\`cpp
unsigned int reverseBits(unsigned int n) {
    unsigned int result = 0;
    for(int i = 0; i < 32; i++) {
        result <<= 1;  // Shift left
        result |= (n & 1);  // Add last bit of n
        n >>= 1;  // Shift n right
    }
    return result;
}
\`\`\`

**Example 7: Single Number Problems**

Problem: All numbers appear twice except one
\`\`\`cpp
int singleNumber(vector<int>& nums) {
    int result = 0;
    for(int x : nums) {
        result ^= x;
    }
    return result;  // a^a^b^b^c = c
}
\`\`\`

Problem: All numbers appear three times except one
\`\`\`cpp
int singleNumber(vector<int>& nums) {
    int ones = 0, twos = 0;
    for(int x : nums) {
        twos |= ones & x;
        ones ^= x;
        int threes = ones & twos;
        ones &= ~threes;
        twos &= ~threes;
    }
    return ones;
}
\`\`\`

**Bit Manipulation Tricks:**

**1. Get/Set/Clear/Toggle Bit:**
\`\`\`cpp
// Get i-th bit
bool getBit(int n, int i) {
    return (n >> i) & 1;
}

// Set i-th bit to 1
int setBit(int n, int i) {
    return n | (1 << i);
}

// Clear i-th bit (set to 0)
int clearBit(int n, int i) {
    return n & ~(1 << i);
}

// Toggle i-th bit
int toggleBit(int n, int i) {
    return n ^ (1 << i);
}
\`\`\`

**2. Get Rightmost Set Bit:**
\`\`\`cpp
int rightmostSetBit(int n) {
    return n & (-n);
}
// Example: 12 = 1100
// -12 in two's complement = 0100
// 12 & (-12) = 0100 = 4
\`\`\`

**3. Check if i-th bit is power of 2:**
\`\`\`cpp
bool isBitPowerOf2(int n, int i) {
    return (n & (1 << i)) != 0;
}
\`\`\`

**4. Fast Modulo Power of 2:**
\`\`\`cpp
// n % 8 when 8 is power of 2
int mod = n & 7;  // Same as n % 8
// Works because 7 = 0111 (all bits < 8)
\`\`\`

**When Bitwise Tricks Replace Traditional:**

**✅ Use Bitwise When:**
1. Working with powers of 2
2. Subset generation
3. Checking even/odd
4. Finding missing/duplicate numbers
5. Fast multiplication/division by powers of 2
6. Space optimization (bitmask DP)
7. Counting set bits

**❌ Avoid Bitwise When:**
1. Code becomes unreadable
2. Simple arithmetic is clearer
3. Working with non-powers of 2
4. Team projects (others may not understand)

**Performance Benefits:**

\`\`\`cpp
// Traditional: Division
int half = n / 2;  // Slower

// Bitwise: Right shift
int half = n >> 1;  // Faster

// Traditional: Modulo
int remainder = n % 4;  // Slower

// Bitwise: AND with mask
int remainder = n & 3;  // Faster (when divisor is power of 2)
\`\`\`

**Common Pitfalls:**

**1. Signed vs Unsigned:**
\`\`\`cpp
int n = -1;
cout << (n >> 1);  // -1 (arithmetic shift, preserves sign)

unsigned int u = -1;
cout << (u >> 1);  // Large positive (logical shift)
\`\`\`

**2. Operator Precedence:**
\`\`\`cpp
// WRONG:
if(n & 1 == 0)  // Parsed as: n & (1 == 0)

// RIGHT:
if((n & 1) == 0)  // Check if even
\`\`\`

**3. Overflow with Shifts:**
\`\`\`cpp
int x = 1 << 31;  // Undefined behavior (overflow)
long long x = 1LL << 31;  // OK
\`\`\`

**Built-in Functions (GCC):**

\`\`\`cpp
// Count set bits
__builtin_popcount(n);  // For int
__builtin_popcountll(n);  // For long long

// Count leading zeros
__builtin_clz(n);

// Count trailing zeros
__builtin_ctz(n);

// Find first set bit
__builtin_ffs(n);
\`\`\`

**Bottom Line:**

Bitwise operations are:
- **Fast:** Direct hardware operations
- **Elegant:** Solve problems concisely
- **Powerful:** Enable unique solutions

Master the basics, practice on problems, and you'll recognize when bitwise tricks apply naturally!`,
        },
        {
            question: 'Bitmask Dynamic Programming is a powerful technique for problems with small constraints. Explain what bitmask DP is, provide a detailed example, and discuss when this technique is applicable.',
            answer: `Bitmask DP uses binary representations to encode states, enabling DP on problems with exponential state spaces. Here's the complete guide:

**What is Bitmask DP?**

Instead of using arrays or maps for DP states, use integers where each bit represents a boolean state.

**Example:** Visited status of 4 cities
\`\`\`cpp
// Traditional: array or set
bool visited[4];

// Bitmask: single integer
int mask = 0;  // 0000 = no cities visited
mask = 5;      // 0101 = cities 0 and 2 visited
mask = 15;     // 1111 = all cities visited
\`\`\`

**Why Use Bitmasks?**

✅ **Space Efficient:** 1 int stores 32 booleans
✅ **Fast Operations:** Check/set/clear in O(1)
✅ **Natural DP States:** Integer key for memoization
✅ **Elegant Transitions:** Bitwise operations

**When to Use Bitmask DP:**

**Applicable when:**
- Small constraint (n ≤ 20, typically n ≤ 15)
- Need to track subset/combination
- States involve "visited", "selected", "available"
- Exponential states (2^n)

**Not applicable when:**
- Large n (n > 20): 2^20 = 1M OK, 2^30 = 1B too much
- States don't involve binary choices
- Simple greedy/divide-conquer works

**Classic Problem: Traveling Salesman (TSP)**

**Problem:** Visit all n cities exactly once, return to start. Minimize distance.

**Constraint:** n ≤ 15

**Approach:**

State: \`dp[mask][i]\` = minimum cost to visit cities in mask, ending at city i

\`\`\`cpp
const int INF = 1e9;
int n;  // Number of cities
int dist[16][16];  // Distance between cities
int dp[1 << 16][16];  // dp[mask][last_city]

int tsp() {
    // Initialize
    for(int mask = 0; mask < (1 << n); mask++) {
        for(int i = 0; i < n; i++) {
            dp[mask][i] = INF;
        }
    }
    
    // Base case: Start at city 0
    dp[1][0] = 0;  // mask = 0001, at city 0
    
    // Iterate all masks
    for(int mask = 0; mask < (1 << n); mask++) {
        for(int i = 0; i < n; i++) {
            // If city i not in mask, skip
            if(!(mask & (1 << i))) continue;
            
            // If this state is unreachable, skip
            if(dp[mask][i] == INF) continue;
            
            // Try visiting next city j
            for(int j = 0; j < n; j++) {
                // If city j already visited, skip
                if(mask & (1 << j)) continue;
                
                // New mask with city j added
                int newMask = mask | (1 << j);
                
                // Update DP
                dp[newMask][j] = min(dp[newMask][j], 
                                     dp[mask][i] + dist[i][j]);
            }
        }
    }
    
    // Answer: All cities visited, return to city 0
    int allVisited = (1 << n) - 1;
    int answer = INF;
    for(int i = 0; i < n; i++) {
        answer = min(answer, dp[allVisited][i] + dist[i][0]);
    }
    
    return answer;
}
\`\`\`

**Complexity:** O(2^n * n^2)
- States: 2^n * n
- Transitions: n per state

**Example 2: Assignment Problem**

**Problem:** n tasks, n people. Each person can do each task with different cost. Assign each task to exactly one person, minimize total cost.

\`\`\`cpp
int n;
int cost[20][20];  // cost[person][task]
int dp[1 << 20];   // dp[mask] = min cost for tasks in mask

int assignment() {
    fill(dp, dp + (1 << n), INF);
    dp[0] = 0;  // No tasks assigned
    
    for(int mask = 0; mask < (1 << n); mask++) {
        // Count how many tasks assigned (= which person is next)
        int person = __builtin_popcount(mask);
        
        if(person >= n) continue;
        
        // Try assigning each remaining task to this person
        for(int task = 0; task < n; task++) {
            // If task already assigned, skip
            if(mask & (1 << task)) continue;
            
            // Assign task to person
            int newMask = mask | (1 << task);
            dp[newMask] = min(dp[newMask], 
                            dp[mask] + cost[person][task]);
        }
    }
    
    return dp[(1 << n) - 1];  // All tasks assigned
}
\`\`\`

**Example 3: Subset Sum**

**Problem:** Given array and target sum, how many subsets sum to target?

\`\`\`cpp
int n, target;
int arr[20];
int dp[1 << 20];  // dp[mask] = sum of elements in mask
int count = 0;

void subsetSum() {
    for(int mask = 0; mask < (1 << n); mask++) {
        int sum = 0;
        for(int i = 0; i < n; i++) {
            if(mask & (1 << i)) {
                sum += arr[i];
            }
        }
        if(sum == target) count++;
    }
}
\`\`\`

**Example 4: Hamiltonian Path**

**Problem:** Does graph have path visiting each node exactly once?

\`\`\`cpp
int n;
vector<int> adj[20];
bool dp[1 << 20][20];  // dp[mask][node] = can reach this state

bool hamiltonianPath() {
    // Try starting from each node
    for(int start = 0; start < n; start++) {
        memset(dp, false, sizeof(dp));
        dp[1 << start][start] = true;
        
        for(int mask = 0; mask < (1 << n); mask++) {
            for(int u = 0; u < n; u++) {
                if(!dp[mask][u]) continue;
                
                // Try going to neighbors
                for(int v : adj[u]) {
                    if(mask & (1 << v)) continue;  // Already visited
                    
                    int newMask = mask | (1 << v);
                    dp[newMask][v] = true;
                }
            }
        }
        
        // Check if we visited all nodes
        int allVisited = (1 << n) - 1;
        for(int i = 0; i < n; i++) {
            if(dp[allVisited][i]) return true;
        }
    }
    return false;
}
\`\`\`

**Bitmask Operations in DP:**

**1. Check if i-th element in subset:**
\`\`\`cpp
if(mask & (1 << i)) {
    // Element i is in subset
}
\`\`\`

**2. Add i-th element:**
\`\`\`cpp
int newMask = mask | (1 << i);
\`\`\`

**3. Remove i-th element:**
\`\`\`cpp
int newMask = mask & ~(1 << i);
\`\`\`

**4. Iterate subsets:**
\`\`\`cpp
for(int mask = 0; mask < (1 << n); mask++) {
    // Process each subset
}
\`\`\`

**5. Iterate elements in mask:**
\`\`\`cpp
for(int i = 0; i < n; i++) {
    if(mask & (1 << i)) {
        // Element i is in mask
    }
}
\`\`\`

**6. Iterate subsets of mask:**
\`\`\`cpp
for(int submask = mask; submask > 0; submask = (submask - 1) & mask) {
    // submask is a subset of mask
}
\`\`\`

**Space Optimization:**

If DP only depends on previous mask:
\`\`\`cpp
// Instead of dp[mask][other_state]
// Use two arrays and alternate
int dp[2][1 << 15];
int curr = 0, prev = 1;

for(...) {
    swap(curr, prev);
    // dp[curr] updated based on dp[prev]
}
\`\`\`

**Common Patterns:**

**Pattern 1: Iterate masks by population:**
\`\`\`cpp
for(int cnt = 0; cnt <= n; cnt++) {  // Number of bits set
    for(int mask = 0; mask < (1 << n); mask++) {
        if(__builtin_popcount(mask) != cnt) continue;
        // Process masks with exactly cnt bits set
    }
}
\`\`\`

**Pattern 2: Include/Exclude DP:**
\`\`\`cpp
for(int i = 0; i < n; i++) {
    for(int mask = (1 << n) - 1; mask >= 0; mask--) {
        // Option 1: Don't include i
        dp[mask] = dp[mask];
        
        // Option 2: Include i (if not already)
        if(!(mask & (1 << i))) {
            dp[mask | (1 << i)] = max(dp[mask | (1 << i)], 
                                     dp[mask] + value[i]);
        }
    }
}
\`\`\`

**Debugging Bitmask DP:**

\`\`\`cpp
// Print mask in binary
void printMask(int mask, int n) {
    for(int i = n-1; i >= 0; i--) {
        cout << ((mask >> i) & 1);
    }
    cout << endl;
}

// Print which elements are in mask
void printElements(int mask, int n) {
    cout << "{";
    bool first = true;
    for(int i = 0; i < n; i++) {
        if(mask & (1 << i)) {
            if(!first) cout << ", ";
            cout << i;
            first = false;
        }
    }
    cout << "}" << endl;
}
\`\`\`

**Performance Tips:**

1. **Use int for masks up to 30:** \`int mask\` handles up to 32 bits
2. **Use long long for larger:** \`long long mask\` for 64 bits
3. **Precompute when possible:** Calculate subset sums once
4. **Optimize inner loops:** Minimize work inside mask iterations

**Memory Constraints:**

\`\`\`
n = 15: 2^15 * 15 = 491K states (OK)
n = 20: 2^20 * 20 = 21M states (OK with int)
n = 25: 2^25 * 25 = 838M states (might MLE)
n = 30: 2^30 * 30 = 32B states (definitely MLE)
\`\`\`

**Bottom Line:**

Bitmask DP is powerful for:
- Small n (typically n ≤ 20)
- Subset/combination problems
- NP-hard problems with small input

Key skills:
- Recognize when applicable (exponential states, small n)
- Design DP state with bitmask
- Use bitwise operations efficiently
- Debug by printing masks

Practice on TSP, assignment, and subset problems to build intuition!`,
        },
        {
            question: 'Explain the relationship between binary representation and various number theory concepts used in competitive programming, such as checking if a number is a power of 2, finding the next power of 2, and computing XOR properties.',
            answer: `Binary representation reveals elegant patterns in number theory. Here's the deep dive:

**Binary Fundamentals:**

Every integer has a binary representation:
\`\`\`
Decimal  Binary    Pattern
0        0000      All zeros
1        0001      Single 1
2        0010      Single 1, shifted
3        0011      Two 1s
4        0100      Single 1, shifted more
5        0101      Two 1s
...
15       1111      All ones (2^4 - 1)
16       10000     Single 1, new position
\`\`\`

**Powers of 2:**

Powers of 2 have exactly ONE bit set:
\`\`\`
2^0  = 1  = 0001
2^1  = 2  = 0010
2^2  = 4  = 0100
2^3  = 8  = 1000
2^4  = 16 = 10000
\`\`\`

**Check if Power of 2:**

Method 1: Brian Kernighan's trick
\`\`\`cpp
bool isPowerOf2(int n) {
    return n > 0 && (n & (n-1)) == 0;
}

// Why it works:
// If n is power of 2: n   = 1000...0
//                    n-1 = 0111...1
//                    n & (n-1) = 0
//
// If n is NOT power of 2: n = 1xx1...x  (multiple 1s)
//                        n-1 = 1xx0...x
//                   n & (n-1) != 0
\`\`\`

**Find Next Power of 2:**

\`\`\`cpp
// Method 1: Bit manipulation
int nextPowerOf2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// How it works:
// Example: n = 12 = 0000 1100
// n--:          0000 1011
// n |= n >> 1:  0000 1111 (propagate highest bit)
// n |= n >> 2:  0000 1111 (already all 1s below highest)
// ... (continue propagating)
// n++:          0001 0000 = 16

// Method 2: Using built-in (simpler)
int nextPowerOf2_v2(int n) {
    return 1 << (32 - __builtin_clz(n-1));
}

// Method 3: Loop (slower but clearer)
int nextPowerOf2_v3(int n) {
    int power = 1;
    while(power < n) power <<= 1;
    return power;
}
\`\`\`

**Find Previous Power of 2:**

\`\`\`cpp
int prevPowerOf2(int n) {
    return 1 << (31 - __builtin_clz(n));
}

// Or using bit manipulation:
int prevPowerOf2_v2(int n) {
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n - (n >> 1);
}
\`\`\`

**XOR Properties and Number Theory:**

**Property 1: Self-Cancellation**
\`\`\`cpp
a ^ a = 0
a ^ 0 = a

// Application: Find unique element
int findUnique(vector<int>& arr) {
    int result = 0;
    for(int x : arr) result ^= x;
    return result;  // All duplicates cancel out
}
\`\`\`

**Property 2: Commutative and Associative**
\`\`\`cpp
a ^ b = b ^ a
(a ^ b) ^ c = a ^ (b ^ c)

// Application: XOR of range
int xorRange(int l, int r) {
    // XOR of 1^2^3^...^n
    auto xor1ToN = [](int n) {
        int mod = n % 4;
        if(mod == 0) return n;
        if(mod == 1) return 1;
        if(mod == 2) return n + 1;
        return 0;
    };
    
    return xor1ToN(r) ^ xor1ToN(l-1);
}
\`\`\`

**Property 3: XOR is its own inverse**
\`\`\`cpp
(a ^ b) ^ b = a

// Application: Simple encryption
string encrypt(string s, int key) {
    for(char& c : s) c ^= key;
    return s;
}

string decrypt(string s, int key) {
    return encrypt(s, key);  // Same operation!
}
\`\`\`

**Property 4: XOR and addition modulo 2**
\`\`\`cpp
// XOR is addition without carry
5 + 3 = 8    // 0101 + 0011 = 1000
5 ^ 3 = 6    // 0101 ^ 0011 = 0110 (addition mod 2)

// Carries: (a & b) << 1
5 & 3 = 1    // 0101 & 0011 = 0001
1 << 1 = 2   // Carry

// So: a + b = (a ^ b) + ((a & b) << 1)
// Can add using only bitwise operations
\`\`\`

**Find Two Non-Repeating Numbers:**

Problem: All numbers appear twice except two. Find them.

\`\`\`cpp
pair<int, int> findTwoUnique(vector<int>& arr) {
    // XOR all numbers
    int xor_all = 0;
    for(int x : arr) xor_all ^= x;
    
    // xor_all = a ^ b (where a, b are unique numbers)
    
    // Find any set bit in xor_all
    int rightmost_bit = xor_all & (-xor_all);
    
    // Divide numbers into two groups based on this bit
    int a = 0, b = 0;
    for(int x : arr) {
        if(x & rightmost_bit) {
            a ^= x;  // Group 1
        } else {
            b ^= x;  // Group 2
        }
    }
    
    return {a, b};
}
\`\`\`

**Binary Exponentiation:**

Fast computation of a^n using binary representation of n:

\`\`\`cpp
long long power(long long a, long long n) {
    long long result = 1;
    while(n > 0) {
        if(n & 1) result *= a;  // If bit is set
        a *= a;  // Square base
        n >>= 1;  // Next bit
    }
    return result;
}

// Why it works:
// n = 13 = 1101 (binary)
// a^13 = a^8 * a^4 * a^1
//      = a^(2^3) * a^(2^2) * a^(2^0)
\`\`\`

**Modular Exponentiation:**

\`\`\`cpp
long long modPower(long long a, long long n, long long mod) {
    long long result = 1;
    a %= mod;
    while(n > 0) {
        if(n & 1) result = (result * a) % mod;
        a = (a * a) % mod;
        n >>= 1;
    }
    return result;
}
\`\`\`

**Gray Code:**

Binary sequences where consecutive numbers differ by 1 bit:

\`\`\`cpp
int binaryToGray(int n) {
    return n ^ (n >> 1);
}

int grayToBinary(int gray) {
    int binary = gray;
    while(gray >>= 1) {
        binary ^= gray;
    }
    return binary;
}

// Generate Gray code sequence
vector<int> grayCode(int n) {
    vector<int> result;
    for(int i = 0; i < (1 << n); i++) {
        result.push_back(i ^ (i >> 1));
    }
    return result;
}
\`\`\`

**Hamming Distance:**

Number of positions where bits differ:

\`\`\`cpp
int hammingDistance(int a, int b) {
    return __builtin_popcount(a ^ b);
}

// Why: XOR gives 1 where bits differ
// Count set bits in XOR
\`\`\`

**Binary GCD (Stein's Algorithm):**

Faster GCD using binary properties:

\`\`\`cpp
int gcd_binary(int a, int b) {
    if(a == 0) return b;
    if(b == 0) return a;
    
    // Find common factor of 2
    int shift = 0;
    while(((a | b) & 1) == 0) {
        a >>= 1;
        b >>= 1;
        shift++;
    }
    
    // Divide out factors of 2 from a
    while((a & 1) == 0) a >>= 1;
    
    while(b != 0) {
        // Divide out factors of 2 from b
        while((b & 1) == 0) b >>= 1;
        
        // Ensure a <= b
        if(a > b) swap(a, b);
        
        b -= a;
    }
    
    return a << shift;
}
\`\`\`

**Bit Tricks in Number Theory:**

**1. Check if n divides 2^k - 1:**
\`\`\`cpp
bool dividesMersenne(int n, int k) {
    return ((1LL << k) - 1) % n == 0;
}
\`\`\`

**2. Multiply by 3:**
\`\`\`cpp
int multiplyBy3(int n) {
    return (n << 1) + n;  // n * 2 + n = n * 3
}
\`\`\`

**3. Divide by 3 (approximate):**
\`\`\`cpp
int divideBy3(int n) {
    return (n * 0x55555556) >> 32;  // Magic constant
}
\`\`\`

**4. Check if two numbers have opposite signs:**
\`\`\`cpp
bool oppositeSign(int a, int b) {
    return (a ^ b) < 0;
}
\`\`\`

**5. Absolute value:**
\`\`\`cpp
int absolute(int n) {
    int mask = n >> 31;  // All 1s if negative, all 0s if positive
    return (n + mask) ^ mask;
}
\`\`\`

**Bottom Line:**

Binary representation reveals:
- **Powers of 2:** Single bit set, n & (n-1) = 0
- **XOR properties:** Self-cancellation, find unique elements
- **Fast operations:** Binary exponentiation, GCD
- **Bit tricks:** Clever number theory shortcuts

Understanding binary opens doors to elegant, efficient solutions!`,
        },
    ],
} as const;

