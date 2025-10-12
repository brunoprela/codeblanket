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
