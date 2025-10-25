/**
 * Advanced Techniques Section
 */

export const advancedtechniquesSection = {
  id: 'advanced-techniques',
  title: 'Advanced Techniques',
  content: `**Brian Kernighan's Algorithm**
Efficiently count set bits by repeatedly removing the rightmost set bit:
\`\`\`python
def count_set_bits (n):
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
def generate_subsets (nums):
    n = len (nums)
    subsets = []
    
    # Iterate through all 2^n possibilities
    for mask in range(1 << n):
        subset = []
        for i in range (n):
            # Check if bit i is set in mask
            if mask & (1 << i):
                subset.append (nums[i])
        subsets.append (subset)
    
    return subsets
\`\`\`

**Fast Exponentiation**
Use bit representation of exponent:
\`\`\`python
def fast_power (base, exp):
    result = 1
    while exp > 0:
        if exp & 1:  # If bit is set
            result *= base
        base *= base
        exp >>= 1  # Divide exponent by 2
    return result
\`\`\``,
};
