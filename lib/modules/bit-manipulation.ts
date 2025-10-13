import { Module } from '@/lib/types';

export const bitManipulationModule: Module = {
  id: 'bit-manipulation',
  title: 'Bit Manipulation',
  description:
    'Master bitwise operations and clever bit tricks for ultra-efficient problem solving.',
  icon: '⚡',
  timeComplexity: 'Often O(1) or O(log n)',
  spaceComplexity: 'Usually O(1)',
  sections: [
    {
      id: 'introduction',
      title: 'Introduction to Bit Manipulation',
      content: `**Bit manipulation** involves using bitwise operators to perform operations directly on the binary representations of numbers. This is one of the most efficient techniques in programming, often providing O(1) solutions to problems that would otherwise require more time and space.

**Why Learn Bit Manipulation?**
- **Ultra-efficient**: Operations are O(1) or O(log n)
- **Space-saving**: Usually requires O(1) extra space
- **Interview favorite**: Common in FAANG interviews
- **Practical applications**: Used in systems programming, cryptography, graphics, and optimization

**When to Use Bit Manipulation:**
- Finding single/missing elements in arrays
- Checking or manipulating specific bit positions
- Optimizing space usage (bit flags, bit sets)
- Mathematical operations (power of 2, counting set bits)
- Subset generation and manipulation`,
      quiz: [
        {
          id: 'q1',
          question:
            'Explain what bit manipulation is and why it is useful in programming. Give real-world examples.',
          sampleAnswer:
            'Bit manipulation is working directly with individual bits in binary representation of numbers using bitwise operators. Useful because: 1) Fast - bitwise operations are single CPU instructions, 2) Space-efficient - store multiple flags in single integer, 3) Mathematical tricks - check power of 2, swap variables without temp. Real-world examples: permissions (Unix: read=4, write=2, execute=1, combined with OR), IP addresses and subnet masks, image compression, cryptography, database indexes, embedded systems (limited memory). For example, file permissions 755 = rwxr-xr-x stored as bits. Graphics: RGB color as 24 bits (8 per channel). Networking: check if IP in subnet by masking. Games: entity flags (isAlive, canFly, isInvincible) in single int. Hardware control: set specific pin high/low.',
          keyPoints: [
            'Direct manipulation of binary representation',
            'Fast (single CPU ops), space-efficient',
            'Uses: permissions, flags, compression, crypto',
            'Examples: Unix permissions, RGB colors, network masks',
            'Store multiple booleans in single integer',
          ],
        },
        {
          id: 'q2',
          question:
            'Compare bit manipulation to regular arithmetic operations. When is bit manipulation preferred?',
          sampleAnswer:
            'Bit manipulation uses &, |, ^, ~, <<, >> operating on binary level. Regular arithmetic uses +, -, *, /, % operating on decimal level. Prefer bit manipulation when: checking even/odd (n & 1 vs n % 2), multiplying/dividing by powers of 2 (n << 1 vs n * 2), swapping (XOR swap vs temp variable), checking power of 2 (n & (n-1) vs division loop), setting/clearing flags (OR/AND vs multiple booleans). Speed: bit ops are O(1) single instruction, while division/modulo can be slower. Space: single int holds 32 flags vs 32 boolean variables. However, prefer readable code unless performance-critical. For example, in tight loop processing millions of pixels, n << 1 faster than n * 2. But in business logic, n * 2 is clearer.',
          keyPoints: [
            'Bit: &, |, ^, ~, <<, >>. Arithmetic: +, -, *, /, %',
            'Bit faster: single instruction vs complex ops',
            'Prefer for: powers of 2, flags, even/odd',
            'Space efficient: 32 flags in one int',
            'Tradeoff: performance vs readability',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk me through the binary representation of positive and negative numbers. How does twos complement work?',
          sampleAnswer:
            'Positive numbers: standard binary (5 = 101). Negative numbers use twos complement: flip all bits (ones complement) then add 1. For example, -5 in 8-bit: 5 = 00000101, flip = 11111010, add 1 = 11111011. Why twos complement? Makes subtraction same as adding negative: 5 + (-5) = 5 + 11111011 = 00000000 (carry ignored). No separate subtraction circuit needed. The most significant bit indicates sign (0=positive, 1=negative). Range for n bits: -2^(n-1) to 2^(n-1)-1. For example, 8 bits: -128 to 127. One asymmetry: -128 exists but 128 does not. Benefit: single representation for zero (no +0 and -0). This is why ~5 gives -6 (flip 101 gives 11111010 which is -6).',
          keyPoints: [
            'Positive: standard binary',
            'Negative: flip bits, add 1 (twos complement)',
            'MSB indicates sign (1=negative)',
            'Makes addition/subtraction use same circuit',
            'Range: -2^(n-1) to 2^(n-1)-1',
          ],
        },
      ],
    },
    {
      id: 'operators',
      title: 'Bitwise Operators',
      content: `**Core Bitwise Operators:**

**1. AND (&)**
- Result is 1 only if both bits are 1
- Use: Testing if bits are set, masking
\`\`\`
  1010  (10)
& 1100  (12)
------
  1000  (8)
\`\`\`

**2. OR (|)**
- Result is 1 if at least one bit is 1
- Use: Setting bits, combining flags
\`\`\`
  1010  (10)
| 1100  (12)
------
  1110  (14)
\`\`\`

**3. XOR (^)**
- Result is 1 if bits are different
- Use: Toggling bits, finding unique elements
\`\`\`
  1010  (10)
^ 1100  (12)
------
  0110  (6)
\`\`\`

**4. NOT (~)**
- Flips all bits (0→1, 1→0)
- Use: Inverting bits
\`\`\`
~1010 = 0101 (in 4-bit system)
\`\`\`

**5. Left Shift (<<)**
- Shifts bits left, fills with 0
- Effect: Multiplies by 2^n
\`\`\`
5 << 2 = 20
0101 << 2 = 10100
\`\`\`

**6. Right Shift (>>)**
- Shifts bits right, fills with sign bit
- Effect: Divides by 2^n (integer division)
\`\`\`
20 >> 2 = 5
10100 >> 2 = 00101
\`\`\``,
      codeExample: `# Bitwise operator examples
a = 10  # 1010 in binary
b = 12  # 1100 in binary

print(f"AND: {a & b}")   # 8  (1000)
print(f"OR:  {a | b}")   # 14 (1110)
print(f"XOR: {a ^ b}")   # 6  (0110)
print(f"NOT: {~a}")      # -11 (two's complement)
print(f"Left shift: {a << 2}")   # 40 (multiply by 4)
print(f"Right shift: {a >> 1}")  # 5  (divide by 2)`,
      quiz: [
        {
          id: 'q1',
          question:
            'Explain each bitwise operator (AND, OR, XOR, NOT, shift). What are their primary use cases?',
          sampleAnswer:
            'AND (&): both bits 1 → result 1, else 0. Use: check if bit set (n & (1 << i)), clear bit (n & ~(1 << i)), mask to keep certain bits. OR (|): any bit 1 → result 1. Use: set bit (n | (1 << i)), combine flags. XOR (^): bits different → result 1. Use: toggle bit (n ^ (1 << i)), swap variables, find unique element. NOT (~): flip all bits. Use: create mask, ones complement. Left shift (<<): multiply by 2^k. Use: fast multiplication, create bit mask (1 << i). Right shift (>>): divide by 2^k. Use: fast division, extract high bits. For example: permissions check (hasPermission = userPerms & WRITE_PERM), toggle LED (ledState ^= 1), fast multiply by 8 (n << 3).',
          keyPoints: [
            'AND: check/clear bits, masking',
            'OR: set bits, combine flags',
            'XOR: toggle, swap, find unique',
            'NOT: flip all bits, create mask',
            'Shifts: fast multiply/divide by powers of 2',
          ],
        },
        {
          id: 'q2',
          question:
            'Describe left vs right shift. What is the difference between arithmetic and logical right shift?',
          sampleAnswer:
            'Left shift (<<): move bits left, fill right with 0s. Each left shift multiplies by 2. Example: 5 << 2 = 20 (101 → 10100). Right shift (>>): move bits right. Two types: logical (fill left with 0s) and arithmetic (fill left with sign bit, preserving sign). Logical: treats number as unsigned (5 >> 1 = 2, 101 → 010). Arithmetic: preserves sign for negative numbers (-8 >> 1 = -4, maintains negative). Language behavior: Python >> is arithmetic, Java >> is arithmetic and >>> is logical. Use cases: left shift for fast multiply by powers of 2, right shift for fast divide. Caution: shifting negative numbers can be tricky (sign extension). Shifting by 32+ bits is undefined in 32-bit integers.',
          keyPoints: [
            'Left shift: move left, fill 0s, multiply by 2^k',
            'Right shift: move right, divide by 2^k',
            'Logical: fill with 0s (unsigned)',
            'Arithmetic: fill with sign bit (signed)',
            'Language differences: Python/Java behavior',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk me through operator precedence and common pitfalls with bitwise operators.',
          sampleAnswer:
            'Precedence (high to low): NOT (~), then shifts (<<, >>), then AND (&), then XOR (^), then OR (|). All bitwise ops have LOWER precedence than comparison operators (==, <, >). Common pitfall: x & 1 == 0 is parsed as x & (1 == 0) = x & 0 = 0, always false! Should be (x & 1) == 0. Another: if (flags & MASK) could be 0 (false) or non-zero (true) - works in boolean context. Mixing signed and unsigned: ~0 is -1 (signed) but 0xFFFFFFFF (unsigned). Overflow: left shift can overflow (5 << 30). Right shift on negative: sign extension vs zero fill. Best practice: use parentheses liberally, understand precedence, test with edge cases (0, -1, max values).',
          keyPoints: [
            'Precedence: ~, shifts, &, ^, |',
            'Lower precedence than comparisons',
            'Pitfall: x & 1 == 0 parsed wrong',
            'Use parentheses: (x & 1) == 0',
            'Careful: signed/unsigned, overflow, edge cases',
          ],
        },
      ],
    },
    {
      id: 'common-patterns',
      title: 'Common Bit Manipulation Patterns',
      content: `**Essential Bit Manipulation Techniques:**

**1. Check if bit i is set:**
\`\`\`python
(n & (1 << i)) != 0
\`\`\`

**2. Set bit i:**
\`\`\`python
n | (1 << i)
\`\`\`

**3. Clear bit i:**
\`\`\`python
n & ~(1 << i)
\`\`\`

**4. Toggle bit i:**
\`\`\`python
n ^ (1 << i)
\`\`\`

**5. Check if power of 2:**
\`\`\`python
n > 0 and (n & (n - 1)) == 0
\`\`\`
Why? Powers of 2 have exactly one bit set. n-1 flips all bits after that bit, so AND gives 0.

**6. Get rightmost set bit:**
\`\`\`python
n & -n
\`\`\`
Why? -n is the two's complement (flip bits and add 1), isolating the rightmost 1.

**7. Remove rightmost set bit:**
\`\`\`python
n & (n - 1)
\`\`\`
Why? n-1 flips all bits from rightmost 1, AND removes it.

**8. Count set bits (Brian Kernighan's algorithm):**
\`\`\`python
count = 0
while n:
    n &= (n - 1)  # Remove rightmost set bit
    count += 1
\`\`\``,
      codeExample: `def bit_manipulation_patterns(n: int):
    """Demonstrate common bit patterns."""
    
    # 1. Check if bit i is set
    i = 2
    is_set = (n & (1 << i)) != 0
    print(f"Bit {i} is set: {is_set}")
    
    # 2. Set bit i
    n_with_bit_set = n | (1 << i)
    print(f"After setting bit {i}: {bin(n_with_bit_set)}")
    
    # 3. Clear bit i
    n_with_bit_cleared = n & ~(1 << i)
    print(f"After clearing bit {i}: {bin(n_with_bit_cleared)}")
    
    # 4. Toggle bit i
    n_toggled = n ^ (1 << i)
    print(f"After toggling bit {i}: {bin(n_toggled)}")
    
    # 5. Check if power of 2
    is_power_of_2 = n > 0 and (n & (n - 1)) == 0
    print(f"Is power of 2: {is_power_of_2}")
    
    # 6. Get rightmost set bit
    rightmost = n & -n
    print(f"Rightmost set bit: {bin(rightmost)}")
    
    # 7. Remove rightmost set bit
    without_rightmost = n & (n - 1)
    print(f"Without rightmost: {bin(without_rightmost)}")
    
    # 8. Count set bits
    count = 0
    temp = n
    while temp:
        temp &= (temp - 1)
        count += 1
    print(f"Number of set bits: {count}")


# Example usage
bit_manipulation_patterns(10)  # Binary: 1010`,
      quiz: [
        {
          id: 'q1',
          question:
            'Explain common bit manipulation patterns: check/set/clear/toggle bit. How do they work?',
          sampleAnswer:
            'Check bit i: (n & (1 << i)) != 0. Create mask with only bit i set (1 << i), AND with n. If result non-zero, bit is set. Set bit i: n | (1 << i). OR with mask sets bit to 1 without affecting others. Clear bit i: n & ~(1 << i). Create mask with all bits 1 except i, AND clears only bit i. Toggle bit i: n ^ (1 << i). XOR flips bit i. For example, n=10 (1010), check bit 1: 10 & (1 << 1) = 10 & 2 = 2 (true). Set bit 0: 10 | 1 = 11 (1011). Clear bit 3: 10 & ~8 = 10 & 7 = 2 (0010). Toggle bit 2: 10 ^ 4 = 14 (1110). These are building blocks for all bit manipulation algorithms.',
          keyPoints: [
            'Check: n & (1 << i) != 0',
            'Set: n | (1 << i)',
            'Clear: n & ~(1 << i)',
            'Toggle: n ^ (1 << i)',
            'Foundation for all bit manipulation',
          ],
        },
        {
          id: 'q2',
          question:
            'Describe the n & (n-1) trick. Why is it useful and what problems does it solve?',
          sampleAnswer:
            'n & (n-1) removes the rightmost set bit. How it works: n-1 flips all bits after rightmost 1 (including that 1). ANDing clears that bit. Example: n=12 (1100), n-1=11 (1011), 1100 & 1011 = 1000 (8). Applications: 1) Check power of 2: (n & (n-1)) == 0 (power of 2 has only one bit). 2) Count set bits: loop n = n & (n-1) until n=0. 3) Find rightmost set bit: n & ~(n-1) or n & -n. For example, check 8 is power of 2: 8 & 7 = 1000 & 0111 = 0 (yes). Count bits in 13 (1101): 13 & 12 = 12, 12 & 11 = 8, 8 & 7 = 0, so 3 bits. This trick is O(k) where k is number of set bits, more efficient than checking all 32 bits.',
          keyPoints: [
            'n & (n-1) removes rightmost set bit',
            'n-1 flips bits after rightmost 1',
            'Check power of 2: (n & (n-1)) == 0',
            'Count bits: loop until n=0',
            'O(k) where k = number of set bits',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk me through counting set bits (Hamming weight). What are different approaches and their complexities?',
          sampleAnswer:
            'Counting set bits (1s in binary). Approach 1: check each bit with (n & (1 << i)), O(log n) or O(32) for 32-bit. Approach 2: n & (n-1) removes rightmost bit, count iterations until n=0, O(k) where k is set bits (better for sparse). Approach 3: lookup table for 8-bit chunks, O(1) with O(256) space. Approach 4: divide-and-conquer (bit tricks), O(1). Example for 13 (1101) using approach 2: 13 & 12 = 12 (1100), count=1; 12 & 11 = 8 (1000), count=2; 8 & 7 = 0, count=3. For 0b10101010 (sparse), approach 2 does 4 iterations vs 8 for approach 1. Best choice depends on: sparse vs dense bits, need constant time, space constraints.',
          keyPoints: [
            'Approach 1: check each bit O(log n)',
            'Approach 2: n & (n-1) loop O(k)',
            'Approach 3: lookup table O(1) with space',
            'Choose based on: sparse/dense, time/space',
            'Brian Kernighan algorithm: n & (n-1) most common',
          ],
        },
      ],
    },
    {
      id: 'xor-properties',
      title: 'XOR Properties and Applications',
      content: `**XOR (Exclusive OR) is a superstar operator** with unique properties that make it invaluable for many problems.

**Key XOR Properties:**

**1. Self-cancellation:**
\`\`\`python
a ^ a = 0  # Any number XOR itself equals 0
\`\`\`

**2. Identity:**
\`\`\`python
a ^ 0 = a  # Any number XOR 0 equals itself
\`\`\`

**3. Commutative:**
\`\`\`python
a ^ b = b ^ a
\`\`\`

**4. Associative:**
\`\`\`python
(a ^ b) ^ c = a ^ (b ^ c)
\`\`\`

**5. Self-inverse:**
\`\`\`python
a ^ b ^ b = a  # XOR twice with same value cancels out
\`\`\`

**Classic XOR Applications:**

**Finding Single Element:**
When every element appears twice except one, XOR all elements. Duplicates cancel to 0, leaving the unique element.

**Swapping Without Temp Variable:**
\`\`\`python
a ^= b  # a = a ^ b
b ^= a  # b = b ^ (a ^ b) = a
a ^= b  # a = (a ^ b) ^ a = b
\`\`\`

**Missing Number:**
XOR all indices and all array values. The missing index won't have a pair to cancel with.

**Parity Check:**
XOR all bits to check if count of 1s is odd or even.`,
      codeExample: `def xor_applications():
    """Demonstrate powerful XOR applications."""
    
    # 1. Find single number (others appear twice)
    def find_single(nums):
        result = 0
        for num in nums:
            result ^= num
        return result
    
    print(find_single([4, 1, 2, 1, 2]))  # 4
    
    # 2. Swap two numbers
    def swap_xor(a, b):
        print(f"Before: a={a}, b={b}")
a ^= b
b ^= a
a ^= b
        print(f"After: a={a}, b={b}")
        return a, b
    
    swap_xor(5, 10)
    
    # 3. Find missing number in range [0, n]
    def find_missing(nums):
        result = len(nums)
        for i, num in enumerate(nums):
            result ^= i ^ num
        return result
    
    print(find_missing([3, 0, 1]))  # 2
    
    # 4. Check if two numbers have opposite signs
    def opposite_signs(a, b):
        return (a ^ b) < 0
    
    print(opposite_signs(-5, 10))  # True
    print(opposite_signs(5, 10))   # False
    
    # 5. Find two non-repeating elements
    def find_two_unique(nums):
        # XOR all to get xor of the two unique numbers
        xor = 0
        for num in nums:
            xor ^= num
        
        # Find rightmost set bit (where they differ)
        rightmost = xor & -xor
        
        # Partition numbers based on this bit
        num1 = num2 = 0
        for num in nums:
            if num & rightmost:
                num1 ^= num
            else:
                num2 ^= num
        
        return num1, num2
    
    print(find_two_unique([1, 2, 1, 3, 2, 5]))  # (3, 5)`,
      quiz: [
        {
          id: 'q1',
          question:
            'Explain XOR properties and why XOR is special. What makes it useful for problem-solving?',
          sampleAnswer:
            'XOR properties: 1) Commutative: a ^ b = b ^ a. 2) Associative: (a ^ b) ^ c = a ^ (b ^ c). 3) Identity: a ^ 0 = a. 4) Self-inverse: a ^ a = 0. 5) Reversible: if a ^ b = c, then c ^ b = a. These make XOR unique - it is its own inverse. Applications: 1) Find unique element (all others appear twice): XOR all elements, duplicates cancel out. 2) Swap variables: a ^= b, b ^= a, a ^= b (no temp). 3) Detect different bits: a ^ b gives 1 where bits differ. 4) Simple encryption: msg ^ key encrypts, cipher ^ key decrypts. For example, [4,1,2,1,2] → 4^1^2^1^2 = 4 (1s and 2s cancel). The self-inverse property (a ^ a = 0) is key to many algorithms.',
          keyPoints: [
            'Properties: commutative, associative, self-inverse',
            'a ^ a = 0, a ^ 0 = a (key properties)',
            'Find unique: XOR all, duplicates cancel',
            'Swap without temp: a ^= b, b ^= a, a ^= b',
            'Encryption: XOR is reversible cipher',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through the "find single number" problem where all others appear twice. Why does XOR solve it elegantly?',
          sampleAnswer:
            'Problem: array where every element appears twice except one. Find the single one. XOR solution: XOR all elements, result is the single number. Why it works: XOR is commutative and associative, so order does not matter. Pair elements: a^b^a^c^b = (a^a)^(b^b)^c = 0^0^c = c. Each duplicate pair XORs to 0, only single element remains. Example: [4,1,2,1,2] → 4^1^2^1^2 = 4^(1^1)^(2^2) = 4^0^0 = 4. Time O(n), space O(1) - optimal. Alternative approaches: hash set O(n) space, sort O(n log n) time. XOR is elegant because: single pass, constant space, leverages mathematical property. Extension: if all appear three times except one, need different technique (bit counting modulo 3).',
          keyPoints: [
            'XOR all elements, duplicates cancel (a^a=0)',
            'Order does not matter (commutative, associative)',
            'O(n) time, O(1) space - optimal',
            'vs Hash: O(n) space, vs Sort: O(n log n) time',
            'Extension: three times needs different approach',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe how to find two unique numbers when all others appear twice using XOR. What is the key insight?',
          sampleAnswer:
            'Problem: array where every element appears twice except two unique numbers a and b. XOR all elements gives a^b (duplicates cancel). But how to separate a and b? Key insight: find any bit where a and b differ (rightmost set bit in a^b). Use this bit to partition array into two groups: bit set and bit clear. Each group has one unique number and matching pairs. XOR each group separately to find a and b. Example: [1,2,1,3,2,5], xor_all=3^5=6 (110). Rightmost set bit: 6 & -6 = 2 (010). Group 1 (bit 1 set): 2,3,2 → XOR=3. Group 2 (bit 1 clear): 1,1,5 → XOR=5. Result: 3 and 5. Time O(n), space O(1). Brilliant use of XOR partitioning.',
          keyPoints: [
            'XOR all → a^b (duplicates cancel)',
            'Find bit where a and b differ',
            'Partition by that bit: two groups',
            'XOR each group to find unique in each',
            'O(n) time, O(1) space, elegant solution',
          ],
        },
      ],
    },
    {
      id: 'advanced-techniques',
      title: 'Advanced Techniques',
      content: `**Brian Kernighan's Algorithm**
Efficiently count set bits by repeatedly removing the rightmost set bit:
\`\`\`python
def count_set_bits(n):
    count = 0
    while n:
        n &= (n - 1)  # Remove rightmost set bit
        count += 1
    return count
\`\`\`
**Complexity:** O(k) where k = number of set bits

**Bit Masking**
Use bits to represent sets of options:
\`\`\`python
# Permissions example
READ = 1 << 0    # 001
WRITE = 1 << 1   # 010
EXECUTE = 1 << 2 # 100

permissions = READ | WRITE  # 011 (has read and write)
has_read = (permissions & READ) != 0  # Check permission
permissions |= EXECUTE  # Add execute permission
permissions &= ~WRITE   # Remove write permission
\`\`\`

**Generating Subsets**
Use bits to represent which elements are included:
\`\`\`python
def generate_subsets(nums):
    n = len(nums)
    subsets = []
    
    # Iterate through all 2^n possibilities
    for mask in range(1 << n):
        subset = []
        for i in range(n):
            # Check if bit i is set in mask
            if mask & (1 << i):
                subset.append(nums[i])
        subsets.append(subset)
    
    return subsets
\`\`\`

**Fast Exponentiation**
Use bit representation of exponent:
\`\`\`python
def fast_power(base, exp):
    result = 1
    while exp > 0:
        if exp & 1:  # If bit is set
            result *= base
        base *= base
        exp >>= 1  # Divide exponent by 2
    return result
\`\`\``,
      codeExample: `# Advanced bit manipulation techniques

def brian_kernighan(n: int) -> int:
    """Count set bits efficiently."""
    count = 0
    while n:
        n &= (n - 1)
        count += 1
    return count

print(f"Set bits in 13: {brian_kernighan(13)}")  # 3 (binary: 1101)


def is_power_of_two(n: int) -> bool:
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0

print(f"16 is power of 2: {is_power_of_two(16)}")   # True
print(f"18 is power of 2: {is_power_of_two(18)}")   # False


def generate_all_subsets(nums: list) -> list:
    """Generate all 2^n subsets using bit manipulation."""
    n = len(nums)
    all_subsets = []
    
    for mask in range(1 << n):  # 2^n iterations
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(nums[i])
        all_subsets.append(subset)
    
    return all_subsets

print(generate_all_subsets([1, 2, 3]))


def reverse_bits(n: int) -> int:
    """Reverse bits of a 32-bit integer."""
    result = 0
    for i in range(32):
        # Get bit at position i from right
        bit = (n >> i) & 1
        # Set bit at position (31-i) from right
        result |= (bit << (31 - i))
    return result

print(f"Reverse of 43261596: {reverse_bits(43261596)}")`,
      quiz: [
        {
          id: 'q1',
          question:
            'Explain bit masking and how it is used for subset generation. Why is it efficient?',
          sampleAnswer:
            'Bit masking uses integers as bit arrays to represent sets. Each bit position represents an element (bit i → element i). For n elements, there are 2^n subsets, represented by integers 0 to 2^n-1. For example, set {a,b,c}, binary 101 (5) represents subset {a,c}. Generate all subsets: iterate 0 to 2^n-1, check each bit to see which elements included. For {a,b,c}: 000→{}, 001→{c}, 010→{b}, 011→{b,c}, 100→{a}, 101→{a,c}, 110→{a,b}, 111→{a,b,c}. Efficient because: compact (32 bits in one int), fast operations (check/set/clear bits), natural enumeration. Applications: DP with bitmask (TSP, job assignment), subset sum, combination generation. This is why many DP problems use "mask" to represent state.',
          keyPoints: [
            'Integer as bit array, each bit = element',
            '2^n integers represent 2^n subsets',
            'Iterate 0 to 2^n-1, check bits',
            'Compact (32 in int), fast operations',
            'Uses: DP bitmask, subset problems',
          ],
        },
        {
          id: 'q2',
          question:
            'Describe bit manipulation tricks for arithmetic: multiply/divide by powers of 2, check even/odd, power of 2.',
          sampleAnswer:
            'Multiply by 2^k: n << k. Each left shift doubles (n << 3 = n * 8). Divide by 2^k: n >> k. Each right shift halves (n >> 2 = n / 4). Check even/odd: n & 1 (0=even, 1=odd). Check power of 2: (n & (n-1)) == 0 and n != 0. Power of 2 has only one bit set. Get rightmost set bit: n & -n. Round up to next power of 2: set all bits to right of highest bit, then add 1. For example, check 16 is power of 2: 16 & 15 = 10000 & 01111 = 0 (yes). Multiply 7 by 8: 7 << 3 = 56. Check 13 even/odd: 13 & 1 = 1 (odd). These tricks are single CPU instructions vs expensive arithmetic operations.',
          keyPoints: [
            'Multiply: n << k (n * 2^k)',
            'Divide: n >> k (n / 2^k)',
            'Even/odd: n & 1',
            'Power of 2: (n & (n-1)) == 0',
            'Single instruction vs arithmetic',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk me through reversing bits in an integer. What are different approaches?',
          sampleAnswer:
            'Reverse 32-bit integer: 10110... → ...01101. Approach 1: iterate bits, extract from right, build from left. For each i from 0 to 31, check bit i in input, set bit (31-i) in output. O(32). Approach 2: swap pairs, then nibbles, then bytes (divide-and-conquer). Swap bits 0↔1, 2↔3, ..., then pairs 0-1↔2-3, etc. O(log 32) operations. Approach 3: lookup table for 8-bit chunks, reverse and swap positions of 4 bytes. O(1) with O(256) space. Example approach 1 for 43261596 (binary ...11010): bit 0 is 0, set bit 31=0; bit 1 is 0, set bit 30=0; ...; bit 31 is 0, set bit 0=0. Result: reversed bits. Choice depends on: one-time (approach 1) vs frequent (lookup table).',
          keyPoints: [
            'Approach 1: iterate, extract right, build left O(32)',
            'Approach 2: divide-and-conquer swaps O(log n)',
            'Approach 3: lookup table per byte O(1)',
            'Trade: simplicity vs speed vs space',
            'Frequent calls → lookup table',
          ],
        },
      ],
    },
    {
      id: 'complexity',
      title: 'Time and Space Complexity',
      content: `**Time Complexity:**

**Bit Operations:** O(1)
- AND, OR, XOR, NOT: Constant time
- Shift operations: Constant time

**Counting Set Bits:**
- Naive (check all 32 bits): O(32) = O(1)
- Brian Kernighan: O(k) where k = number of set bits
- Best case: O(1), Worst case: O(log n)

**Subset Generation:**
- Generate all subsets: O(2^n × n)
- Need to examine all 2^n subsets, each takes O(n) to build

**Finding Patterns:**
- Single element (XOR): O(n) to iterate, O(1) per operation
- Missing number: O(n) time

**Space Complexity:**

Most bit manipulation algorithms use **O(1) space** - this is their main advantage!

**Exceptions:**
- Subset generation: O(2^n × n) to store all subsets
- Bit masking for DP: O(2^n) for state space
- Lookup tables: O(2^k) for k-bit chunks`,
      quiz: [
        {
          id: 'q1',
          question:
            'Analyze the time and space complexity of bit manipulation operations. Why are they considered fast?',
          sampleAnswer:
            'Individual bit operations (&, |, ^, ~, <<, >>) are O(1) time and O(1) space - single CPU instructions. Check/set/clear/toggle bit: O(1). Count set bits with n & (n-1): O(k) where k is set bits (better than O(log n) for sparse). Subset generation: O(2^n) to iterate all, O(2^n × n) to store. Bit masking DP: O(2^n × ...) for state space (exponential but compact). Lookup tables: O(1) query with O(2^k) space for k-bit chunks. Why fast? 1) Hardware support - CPU does in one cycle. 2) No memory access - registers only. 3) Predictable - no branches. For example, check even with n & 1 vs n % 2: both O(1) but bitwise is single AND instruction while modulo can be multiple instructions.',
          keyPoints: [
            'Bit operations: O(1) time, single CPU instruction',
            'n & (n-1): O(k) where k = set bits',
            'Subset generation: O(2^n) exponential',
            'Fast: hardware support, no memory, no branches',
            'vs Arithmetic: fewer CPU cycles',
          ],
        },
        {
          id: 'q2',
          question:
            'Compare space complexity of bit manipulation vs alternative approaches for flag storage.',
          sampleAnswer:
            'Bit manipulation stores 32 flags in single 32-bit int (4 bytes). Alternative: 32 boolean variables (32 bytes). Space saving: 8x for 32 flags. For n flags: bit manipulation ⌈n/32⌉ ints, booleans n bytes. Example: 1000 flags: bits need 32 ints (128 bytes), booleans need 1000 bytes. Trade-off: bit manipulation saves space but less readable. Best for: embedded systems (limited RAM), large datasets (millions of flags), performance-critical code. Not worth for: few flags (< 32), readability matters, premature optimization. Example: Linux file permissions (rwxrwxrwx) stored in 9 bits vs 9 booleans. Game entity flags: 32 states in one int vs 32 variables. Modern systems have lots of memory, prefer readability unless proven bottleneck.',
          keyPoints: [
            'Bit: 32 flags in 4 bytes (32-bit int)',
            'Boolean array: n flags in n bytes',
            'Space saving: 8x for 32 flags',
            'Use when: embedded, large datasets, proven bottleneck',
            'vs Readability: optimize only when necessary',
          ],
        },
        {
          id: 'q3',
          question:
            'Explain complexity of bitmask DP. When is exponential state space acceptable?',
          sampleAnswer:
            'Bitmask DP uses integer to represent state (each bit = element included/excluded). State space: 2^n for n elements. For example, TSP with n cities: O(n^2 × 2^n) time, O(n × 2^n) space. Exponential but: 1) Compact - 2^20 ≈ 1M states fits in memory. 2) Fast bit ops for state transitions. 3) Often best known algorithm (TSP, job assignment). Acceptable when n is small (n ≤ 20-22). For n=20: 2^20 × 20 = 20M operations - feasible. For n=30: 2^30 × 30 = 30B operations - too slow. Alternative: brute force is O(n!) which is worse (20! = 2.4×10^18). Bitmask DP trades exponential space for feasible solution. Used: TSP, subset sum, assignment problems. Key: recognize when n is small enough.',
          keyPoints: [
            'Bitmask DP: O(2^n) state space',
            'Compact: 2^20 ≈ 1M states fits memory',
            'Acceptable for n ≤ 20-22',
            'Often better than brute force O(n!)',
            'Uses: TSP, job assignment, subset problems',
          ],
        },
      ],
    },
    {
      id: 'interview-strategy',
      title: 'Interview Strategy',
      content: `**Recognizing Bit Manipulation Problems:**

**Keywords to watch for:**
- "Find the single/unique element"
- "Without using extra space"
- "Appears twice/even times except one"
- "Power of 2"
- "Count set bits / Hamming weight"
- "Missing number"
- "Subset generation"
- "Toggle/flip/set/clear bits"

**Problem-Solving Framework:**

**1. Identify the Pattern:**
- Pairs that cancel? → Think XOR
- Need to check specific bits? → Think masking
- Power of 2 check? → Think n & (n-1)
- Count operations? → Think Brian Kernighan

**2. Consider XOR First:**
- XOR is usually the answer when dealing with pairs
- Properties: a^a=0, a^0=a, commutative, associative

**3. Draw Binary Representations:**
- Write out small examples in binary
- Look for patterns in the bits
- Visualize what operations do

**4. Common Interview Questions:**
- Single Number → XOR all elements
- Missing Number → XOR indices and values
- Power of Two → n & (n-1) == 0
- Count Bits → Brian Kernighan
- Reverse Bits → Process each bit
- Hamming Distance → XOR then count set bits

**Communication Tips:**
- Explain why bit manipulation is optimal (O(1) space)
- Mention alternative approaches (sets, sorting)
- Discuss trade-offs
- Walk through binary examples
- Explain the math/logic behind the operation

**Common Pitfalls:**
- Forgetting to handle negative numbers
- Off-by-one errors with bit positions
- Integer overflow (in languages like Java/C++)
- Mixing up AND vs OR
- Not considering edge cases (0, negative numbers)`,
      quiz: [
        {
          id: 'q1',
          question:
            'How do you recognize a bit manipulation problem? What keywords and patterns signal these techniques?',
          sampleAnswer:
            'Keywords: "binary representation", "XOR", "bit", "power of 2", "single number", "subset", "mask", "toggle", "flip", "set/clear flag". Patterns: 1) Find unique element (XOR). 2) Check/set/clear flags (bit operations). 3) Subset generation (2^n enumeration). 4) Fast multiply/divide by 2 (shifts). 5) Space optimization (pack booleans). 6) Count set bits, reverse bits. For example, "find single number where all others appear twice" → XOR all. "Generate all subsets" → iterate 0 to 2^n-1. "Check if power of 2" → n & (n-1) == 0. "Optimize flag storage" → bitmask. "Count 1-bits" → Brian Kernighan. Signals: mention of binary, efficient flag storage, XOR properties, subset problems. These problems often have elegant bit solutions vs complex alternatives.',
          keyPoints: [
            'Keywords: XOR, binary, bit, power of 2, subset',
            'Patterns: unique element, flags, subsets, fast ops',
            'Examples: single number, power of 2, subset generation',
            'Signals: binary operations, flag storage',
            'Elegant bit solutions vs alternatives',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through your interview approach for bit manipulation problems from recognition to implementation.',
          sampleAnswer:
            'First, recognize bit manipulation opportunity from keywords (XOR, binary, single number). Second, identify pattern: unique element? Flags? Subset? Power of 2? Third, choose technique: XOR for cancellation, masks for flags, n & (n-1) for power of 2, iteration for subsets. Fourth, consider edge cases: 0, negative numbers, overflow, 32 vs 64 bit. Fifth, code clearly with comments explaining bit logic. Sixth, test with examples: trace bits through operations. Finally, analyze complexity and discuss alternatives. For example, "find single number": recognize XOR pattern, explain a^a=0 property, code single line result = functools.reduce(operator.xor, nums), test [4,1,2,1,2] → 4, note O(n) time O(1) space vs hash O(n) space. Show understanding of both bit tricks and fundamentals.',
          keyPoints: [
            'Recognize: keywords, patterns',
            'Identify: XOR, masks, subsets, power of 2',
            'Consider: edge cases, overflow, negatives',
            'Code clearly with comments on bit logic',
            'Test: trace bits, analyze complexity',
            'Discuss alternatives',
          ],
        },
        {
          id: 'q3',
          question:
            'What are common mistakes in bit manipulation and how do you avoid them?',
          sampleAnswer:
            'First: operator precedence (x & 1 == 0 parsed wrong). Second: forgetting edge cases (0, negative, INT_MAX). Third: mixing signed/unsigned (overflow, sign extension). Fourth: wrong shift direction (left vs right). Fifth: off-by-one in bit positions (0-indexed). Sixth: not handling negative numbers correctly (arithmetic vs logical shift). Seventh: integer overflow from left shift. My strategy: 1) Always use parentheses: (x & 1) == 0. 2) Test edge cases: 0, -1, INT_MAX, INT_MIN. 3) Be explicit about signed/unsigned. 4) Comment which direction and why. 5) Use masks like (1 << i) with bounds check. 6) Trace example manually before coding. 7) Remember Python has arbitrary precision (no overflow) but other languages do not. Most mistakes from precedence and edge cases.',
          keyPoints: [
            'Precedence: use parentheses (x & 1) == 0',
            'Edge cases: 0, negative, max/min values',
            'Signed/unsigned: overflow, sign extension',
            'Direction: left vs right shift',
            'Test: trace manually, edge cases',
            'Language: Python arbitrary vs C overflow',
          ],
        },
      ],
    },
  ],
  keyTakeaways: [
    'XOR is your best friend: a ^ a = 0, a ^ 0 = a, perfect for finding unique elements',
    'n & (n-1) removes the rightmost set bit - used for power of 2 check and counting bits',
    'Left shift (<<) multiplies by 2^n, right shift (>>) divides by 2^n',
    'Bit manipulation provides O(1) space solutions when others need O(n)',
    'Brian Kernighan algorithm efficiently counts set bits in O(k) time',
    'Use bit masking to represent sets and combinations efficiently',
    'Always draw binary representations to visualize the problem',
    'Bit manipulation is common in interviews - master XOR patterns especially',
  ],
  relatedProblems: ['single-number', 'number-of-1-bits', 'missing-number'],
};
