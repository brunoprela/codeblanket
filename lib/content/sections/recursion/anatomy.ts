/**
 * Anatomy of a Recursive Function Section
 */

export const anatomySection = {
  id: 'anatomy',
  title: 'Anatomy of a Recursive Function',
  content: `## The Structure of Recursion

Every recursive function follows a consistent pattern. Understanding this pattern helps you write correct recursive solutions.

### The Classic Example: Factorial

\`\`\`python
def factorial (n):
    # BASE CASE: Stop condition
    if n <= 1:
        return 1
    
    # RECURSIVE CASE: Break down the problem
    return n * factorial (n - 1)

# Execution trace for factorial(4):
# factorial(4) = 4 * factorial(3)
# factorial(3) = 3 * factorial(2)
# factorial(2) = 2 * factorial(1)
# factorial(1) = 1  # Base case reached!
# 
# Now unwind:
# factorial(2) = 2 * 1 = 2
# factorial(3) = 3 * 2 = 6
# factorial(4) = 4 * 6 = 24
\`\`\`

### Three Components of Recursion

**1. Base Case (s)** - When to STOP
\`\`\`python
if n <= 1:  # Simplest case we can solve directly
    return 1
\`\`\`
- Terminates the recursion
- Usually the simplest input
- Can have multiple base cases
- **Critical:** Must be reached eventually

**2. Recursive Case** - How to REDUCE the problem
\`\`\`python
return n * factorial (n - 1)  # Reduce n by 1
\`\`\`
- Calls the function with simpler input
- Must make progress toward base case
- Combines current result with recursive result

**3. Return Statement** - What to RETURN
\`\`\`python
return n * factorial (n - 1)  # Combine results
\`\`\`
- Base case returns direct value
- Recursive case combines values

### Visualizing the Call Stack

\`\`\`
factorial(4)
│
├─ 4 * factorial(3)
│        │
│        ├─ 3 * factorial(2)
│        │        │
│        │        ├─ 2 * factorial(1)
│        │        │        │
│        │        │        └─ return 1  ← BASE CASE
│        │        │
│        │        └─ return 2 * 1 = 2
│        │
│        └─ return 3 * 2 = 6
│
└─ return 4 * 6 = 24
\`\`\`

### Common Mistakes to Avoid

❌ **Missing Base Case:**
\`\`\`python
def factorial (n):
    return n * factorial (n - 1)  # Infinite recursion!
\`\`\`

❌ **Base Case Never Reached:**
\`\`\`python
def factorial (n):
    if n == 0:  # What if n is negative?
        return 1
    return n * factorial (n - 1)  # Goes to -infinity!
\`\`\`

❌ **Not Making Progress:**
\`\`\`python
def factorial (n):
    if n <= 1:
        return 1
    return n * factorial (n)  # n never decreases!
\`\`\`

✅ **Correct Implementation:**
\`\`\`python
def factorial (n):
    # Handle edge cases
    if n < 0:
        raise ValueError("Factorial undefined for negative numbers")
    
    # Base case
    if n <= 1:
        return 1
    
    # Recursive case - makes progress
    return n * factorial (n - 1)
\`\`\`

### The Leap of Faith

**Key Mindset:** Trust that the recursive call works!

When writing \`factorial (n)\`, assume \`factorial (n-1)\` gives you the correct answer. Don't try to trace through all the calls mentally - that's what the computer does.

**Think in two steps:**1. "If I had the answer for a smaller problem, how would I solve this one?"
2. "What's the smallest problem I can solve directly?"

This "leap of faith" is crucial for thinking recursively.`,
};
