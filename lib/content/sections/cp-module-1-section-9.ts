export const bitsBytesOperationsSection = {
  id: 'cp-m1-s9',
  title: 'Bits, Bytes & Bitwise Operations',
  content: `

# Bits, Bytes & Bitwise Operations

## Introduction

Bit manipulation is a fundamental skill in competitive programming. Many problems can be solved elegantly and efficiently using bitwise operations, and understanding bits is crucial for bitmask dynamic programming, subset enumeration, and optimization tricks.

In this section, we'll master binary representation, bitwise operators, common bit tricks, and practical applications in CP.

**Goal**: Become proficient with bit manipulation for competitive programming.

---

## Binary Representation

### Understanding Bits

A **bit** is the smallest unit of data—either 0 or 1.

**1 byte = 8 bits**

\`\`\`
Binary:    10110101
Positions: 76543210  (bit positions, right to left)
\`\`\`

### Integer Representation

\`\`\`cpp
int x = 13;

// Binary representation: 00000000 00000000 00000000 00001101
// In bits:               0        1        1        0        1
//                        ↑        ↑        ↑        ↑        ↑
//                       bit 3    bit 2    bit 1    bit 0
\`\`\`

**Place values** (powers of 2):
\`\`\`
Bit 0:  2^0 = 1
Bit 1:  2^1 = 2
Bit 2:  2^2 = 4
Bit 3:  2^3 = 8
Bit 4:  2^4 = 16
...
\`\`\`

**13 in binary:**
\`\`\`
  1101
= 1×8 + 1×4 + 0×2 + 1×1
= 8 + 4 + 0 + 1
= 13
\`\`\`

### Signed vs Unsigned

**Unsigned:** All bits represent magnitude
\`\`\`cpp
unsigned int x = 255;  // 11111111 in binary
\`\`\`

**Signed:** Uses two's complement for negative numbers
\`\`\`cpp
int x = -1;  // 11111111 11111111 11111111 11111111 (in 32-bit)
\`\`\`

---

## Bitwise Operators

C++ provides six bitwise operators:

| Operator | Name | Example |
|----------|------|---------|
| \`&\` | AND | \`a & b\` |
| \`|\` | OR | \`a | b\` |
| \`^\` | XOR | \`a ^ b\` |
| \`~\` | NOT | \`~a\` |
| \`<<\` | Left Shift | \`a << n\` |
| \`>>\` | Right Shift | \`a >> n\` |

### AND (&)

**Result is 1 only if BOTH bits are 1**

\`\`\`
  1010  (10)
& 1100  (12)
------
  1000  (8)
\`\`\`

\`\`\`cpp
int a = 10;  // 1010
int b = 12;  // 1100
int c = a & b;  // 1000 = 8
\`\`\`

**Use case:** Check if a bit is set
\`\`\`cpp
if (x & (1 << k)) {
    // Bit k is set in x
}
\`\`\`

### OR (|)

**Result is 1 if AT LEAST ONE bit is 1**

\`\`\`
  1010  (10)
| 1100  (12)
------
  1110  (14)
\`\`\`

\`\`\`cpp
int a = 10;  // 1010
int b = 12;  // 1100
int c = a | b;  // 1110 = 14
\`\`\`

**Use case:** Set a bit
\`\`\`cpp
x = x | (1 << k);  // Set bit k
// Or shorthand:
x |= (1 << k);
\`\`\`

### XOR (^)

**Result is 1 if bits are DIFFERENT**

\`\`\`
  1010  (10)
^ 1100  (12)
------
  0110  (6)
\`\`\`

\`\`\`cpp
int a = 10;  // 1010
int b = 12;  // 1100
int c = a ^ b;  // 0110 = 6
\`\`\`

**Properties of XOR:**
- \`a ^ a = 0\` (anything XOR itself is 0)
- \`a ^ 0 = a\` (anything XOR 0 is itself)
- \`a ^ b ^ b = a\` (XOR is associative and self-inverse)

**Use case:** Toggle a bit, find unique element

### NOT (~)

**Flips all bits (1→0, 0→1)**

\`\`\`
  1010  (10)
~
------
  0101  (-11 in two's complement)
\`\`\`

\`\`\`cpp
int a = 10;     // 00001010
int b = ~a;     // 11110101
\`\`\`

**Note:** \`~a\` for signed integers gives \`-(a+1)\` due to two's complement.

### Left Shift (<<)

**Shifts bits to the left, fills with 0s on the right**

\`\`\`
  1010  (10)
<< 2
------
 101000  (40)
\`\`\`

\`\`\`cpp
int a = 10;       // 1010
int b = a << 2;   // 101000 = 40
\`\`\`

**Effect:** Multiplies by 2^n
\`\`\`cpp
x << 1  // x * 2
x << 2  // x * 4
x << 3  // x * 8
x << k  // x * 2^k
\`\`\`

**Use case:** Fast multiplication by powers of 2

### Right Shift (>>)

**Shifts bits to the right**

\`\`\`
  1010  (10)
>> 2
------
  0010  (2)
\`\`\`

\`\`\`cpp
int a = 10;       // 1010
int b = a >> 2;   // 0010 = 2
\`\`\`

**Effect:** Divides by 2^n (integer division)
\`\`\`cpp
x >> 1  // x / 2
x >> 2  // x / 4
x >> k  // x / 2^k
\`\`\`

---

## Common Bit Tricks

### 1. Check if Bit k is Set

\`\`\`cpp
bool isSet(int x, int k) {
    return (x & (1 << k)) != 0;
}

// Example:
int x = 13;  // 1101
cout << isSet(x, 0) << endl;  // 1 (bit 0 is set)
cout << isSet(x, 1) << endl;  // 0 (bit 1 is not set)
cout << isSet(x, 2) << endl;  // 1 (bit 2 is set)
\`\`\`

### 2. Set Bit k

\`\`\`cpp
int setBit(int x, int k) {
    return x | (1 << k);
}

// Example:
int x = 12;  // 1100
x = setBit(x, 0);  // 1101 = 13
\`\`\`

### 3. Clear Bit k

\`\`\`cpp
int clearBit(int x, int k) {
    return x & ~(1 << k);
}

// Example:
int x = 13;  // 1101
x = clearBit(x, 0);  // 1100 = 12
\`\`\`

### 4. Toggle Bit k

\`\`\`cpp
int toggleBit(int x, int k) {
    return x ^ (1 << k);
}

// Example:
int x = 12;  // 1100
x = toggleBit(x, 0);  // 1101 = 13
x = toggleBit(x, 0);  // 1100 = 12 (toggled back)
\`\`\`

### 5. Check if Power of 2

\`\`\`cpp
bool isPowerOf2(int x) {
    return x > 0 && (x & (x - 1)) == 0;
}

// Why it works:
// Power of 2: 1000  (8)
//      - 1:  0111  (7)
//        &:  0000  (0)

// Not power of 2: 1010  (10)
//            - 1: 1001  (9)
//              &: 1000  (8) ≠ 0
\`\`\`

### 6. Count Set Bits (Popcount)

\`\`\`cpp
int countBits(int x) {
    int count = 0;
    while (x) {
        count++;
        x &= (x - 1);  // Clear rightmost set bit
    }
    return count;
}

// Or use built-in:
int count = __builtin_popcount(x);
\`\`\`

### 7. Get Rightmost Set Bit

\`\`\`cpp
int rightmostSetBit(int x) {
    return x & (-x);
}

// Example:
// x = 12:     1100
// -x:         0100  (two's complement)
// x & (-x):   0100  (4)
\`\`\`

### 8. Clear Rightmost Set Bit

\`\`\`cpp
int clearRightmost(int x) {
    return x & (x - 1);
}

// Example:
// x = 12:     1100
// x - 1:      1011
// x & (x-1):  1000  (8)
\`\`\`

### 9. Get All Subsets of a Set

\`\`\`cpp
int n = 5;  // Set {0, 1, 2, 3, 4}

for (int mask = 0; mask < (1 << n); mask++) {
    cout << "Subset: ";
    for (int i = 0; i < n; i++) {
        if (mask & (1 << i)) {
            cout << i << " ";
        }
    }
    cout << endl;
}
\`\`\`

### 10. Iterate Through All Submasks

\`\`\`cpp
int mask = 23;  // 10111

for (int submask = mask; submask > 0; submask = (submask - 1) & mask) {
    // Process submask
}
\`\`\`

---

## Built-in Functions for Bit Manipulation

GCC provides highly optimized built-in functions:

### __builtin_popcount(x)

**Count number of 1-bits**

\`\`\`cpp
int x = 13;  // 1101
cout << __builtin_popcount(x) << endl;  // 3
\`\`\`

For \`long long\`:
\`\`\`cpp
long long x = 1000000000000LL;
cout << __builtin_popcountll(x) << endl;
\`\`\`

### __builtin_clz(x)

**Count Leading Zeros** (from left)

\`\`\`cpp
int x = 8;  // 00000000 00000000 00000000 00001000
cout << __builtin_clz(x) << endl;  // 28 leading zeros
\`\`\`

**Use case:** Find highest set bit
\`\`\`cpp
int highestBit(int x) {
    return 31 - __builtin_clz(x);
}
\`\`\`

### __builtin_ctz(x)

**Count Trailing Zeros** (from right)

\`\`\`cpp
int x = 8;  // 1000
cout << __builtin_ctz(x) << endl;  // 3 trailing zeros
\`\`\`

**Use case:** Find position of rightmost set bit

### __builtin_ffs(x)

**Find First Set** (position of rightmost 1-bit + 1)

\`\`\`cpp
int x = 12;  // 1100
cout << __builtin_ffs(x) << endl;  // 3 (bit 2 is rightmost set bit, return 2+1)
\`\`\`

---

## Bit Manipulation Problem Patterns

### Pattern 1: XOR Properties

**Problem:** Find unique element in array where every other element appears twice

\`\`\`cpp
int findUnique(vector<int>& arr) {
    int result = 0;
    for (int x : arr) {
        result ^= x;
    }
    return result;  // All duplicates cancel out, leaving unique
}

// Example: [2, 3, 2, 4, 3] → 4
\`\`\`

**Why it works:** \`a ^ a = 0\`, \`a ^ 0 = a\`

### Pattern 2: Subset Enumeration

**Problem:** Try all possible subsets of size N

\`\`\`cpp
int n = 5;
for (int mask = 0; mask < (1 << n); mask++) {
    // mask represents a subset
    // bit i is set → element i is in subset
}
// Tries all 2^n subsets
\`\`\`

### Pattern 3: Bitmask DP

**Problem:** Traveling Salesman (visit all cities, N ≤ 20)

\`\`\`cpp
int n = 10;
int dp[1 << n][n];  // dp[mask][last] = min cost

// mask: which cities visited (bit i = 1 if city i visited)
// last: last city visited
\`\`\`

### Pattern 4: Count Set Bits in Range

**Problem:** Count total 1-bits in numbers from 1 to N

\`\`\`cpp
long long countBits(long long n) {
    long long count = 0;
    for (long long i = 1; i <= n; i++) {
        count += __builtin_popcountll(i);
    }
    return count;
}
\`\`\`

---

## Bitset Introduction

\`std::bitset\` is a space-efficient container for bits.

### Basic Usage

\`\`\`cpp
#include <bitset>
using namespace std;

bitset<8> b1;           // 00000000
bitset<8> b2(42);       // 00101010
bitset<8> b3("10101010");  // 10101010

cout << b2 << endl;     // 00101010
cout << b2[1] << endl;  // 0 (bit at position 1)
\`\`\`

### Bitset Operations

\`\`\`cpp
bitset<8> b1("10101010");
bitset<8> b2("11001100");

bitset<8> b3 = b1 & b2;  // AND
bitset<8> b4 = b1 | b2;  // OR
bitset<8> b5 = b1 ^ b2;  // XOR
bitset<8> b6 = ~b1;      // NOT

cout << b3 << endl;  // 10001000
\`\`\`

### Useful Methods

\`\`\`cpp
bitset<8> b("10101010");

b.set();        // Set all bits to 1
b.reset();      // Set all bits to 0
b.flip();       // Flip all bits

b.set(3);       // Set bit 3 to 1
b.reset(3);     // Set bit 3 to 0
b.flip(3);      // Flip bit 3

cout << b.count() << endl;  // Count set bits
cout << b.any() << endl;    // True if any bit is set
cout << b.none() << endl;   // True if no bit is set
cout << b.all() << endl;    // True if all bits are set

cout << b.to_ulong() << endl;  // Convert to unsigned long
cout << b.to_string() << endl; // Convert to string
\`\`\`

### Why Use Bitset?

**Space efficiency:**
\`\`\`cpp
bool arr[1000000];         // 1,000,000 bytes
bitset<1000000> b;         // 125,000 bytes (8x smaller!)
\`\`\`

**Speed:**
- Operations on 64 bits at once
- Much faster than boolean arrays for large N

**Use case in CP:** Dynamic programming with large state space

---

## Practical CP Examples

### Example 1: Power of Two Check

\`\`\`cpp
bool isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}
\`\`\`

### Example 2: Swap Without Temporary

\`\`\`cpp
void swap(int& a, int& b) {
    a ^= b;
    b ^= a;
    a ^= b;
}
// Clever but usually just use std::swap!
\`\`\`

### Example 3: Gray Code

\`\`\`cpp
int binaryToGray(int n) {
    return n ^ (n >> 1);
}

int grayToBinary(int n) {
    int result = n;
    while (n >>= 1) {
        result ^= n;
    }
    return result;
}
\`\`\`

### Example 4: Count Different Bits

\`\`\`cpp
int countDifferentBits(int a, int b) {
    return __builtin_popcount(a ^ b);
}
\`\`\`

### Example 5: Next Power of 2

\`\`\`cpp
int nextPowerOf2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}
\`\`\`

---

## Common Mistakes

### 1. Operator Precedence

❌ **WRONG:**
\`\`\`cpp
if (x & 1 == 0) {  // Parsed as: x & (1 == 0)
    // Bug!
}
\`\`\`

✅ **CORRECT:**
\`\`\`cpp
if ((x & 1) == 0) {
    // Check if x is even
}
\`\`\`

### 2. Signed vs Unsigned Shifts

\`\`\`cpp
int x = -1;
cout << (x >> 1) << endl;  // -1 (sign bit preserved)

unsigned int y = -1;
cout << (y >> 1) << endl;  // Large number (logical shift)
\`\`\`

### 3. Overflow in Bit Shifts

\`\`\`cpp
int x = 1 << 31;  // Undefined behavior for signed int!

// Use:
unsigned int x = 1U << 31;  // OK
long long x = 1LL << 40;    // OK for long long
\`\`\`

---

## Summary

**Essential Bit Operations:**

✅ AND (\`&\`): Check if bit is set
✅ OR (\`|\`): Set a bit
✅ XOR (\`^\`): Toggle bit, find unique
✅ Left shift (\`<<\`): Multiply by 2^k
✅ Right shift (\`>>\`): Divide by 2^k

**Essential Built-ins:**

✅ \`__builtin_popcount(x)\`: Count set bits
✅ \`__builtin_clz(x)\`: Count leading zeros
✅ \`__builtin_ctz(x)\`: Count trailing zeros

**Common Patterns:**

✅ XOR for finding unique elements
✅ Bitmask for subset enumeration
✅ Power of 2 checks: \`x & (x-1) == 0\`
✅ Bitset for space-efficient DP

**Next Steps:**

In the next section, we'll explore **Memory Management for CP**—understanding stack vs heap, memory limits, and how to avoid Memory Limit Exceeded!

**Key Takeaway**: Bit manipulation is powerful for optimization and enables elegant solutions to many problems. Master these basics and you'll unlock a whole category of CP problems!
`,
  quizId: 'cp-m1-s9-quiz',
  discussionId: 'cp-m1-s9-discussion',
} as const;
