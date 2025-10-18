/**
 * Interview Strategy Section
 */

export const interviewstrategySection = {
  id: 'interview-strategy',
  title: 'Interview Strategy',
  content: `**Recognizing Math & Geometry Problems:**

**Keywords to watch for:**
- "Rotate" → Matrix rotation
- "Spiral" → Layer-by-layer traversal
- "Prime" → Number theory
- "Factorial", "Combinations" → Combinatorics
- "Distance", "Area", "Angle" → Geometry
- "Cycle", "Repeating" → Floyd's cycle detection
- "Power", "Exponent" → Fast exponentiation

**Problem-Solving Framework:**

**1. Look for Mathematical Properties:**
- Can you use a formula instead of simulation?
- Are there symmetries you can exploit?
- Can you reduce the problem using math?

**2. Consider Edge Cases:**
- Zero, negative numbers
- Very large numbers (overflow)
- Empty inputs
- Single element

**3. Optimize with Math:**
- Don't check all divisors, only up to sqrt(n)
- Use fast exponentiation for powers
- Apply modular arithmetic for large numbers

**4. Draw Diagrams:**
- Visualize matrix transformations
- Sketch geometric problems
- Mark boundaries and patterns

**5. Break Down Complex Problems:**
- Matrix rotation = transpose + reverse
- Spiral = process layer by layer
- Complex geometry = break into simpler shapes

**Communication Tips:**
- Explain the mathematical insight
- Walk through small examples
- Discuss alternative approaches
- Mention time/space optimizations
- Show awareness of edge cases

**Common Pitfalls:**
- Integer overflow (especially in combinations/factorials)
- Floating point precision errors
- Off-by-one errors in matrix indices
- Not handling negative numbers properly
- Forgetting to consider in-place constraints`,
};
