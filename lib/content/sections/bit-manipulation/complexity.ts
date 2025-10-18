/**
 * Time and Space Complexity Section
 */

export const complexitySection = {
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
};
