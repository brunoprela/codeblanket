/**
 * Interview Strategy Section
 */

export const interviewstrategySection = {
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
};
