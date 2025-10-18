/**
 * Common Stack Patterns Section
 */

export const patternsSection = {
  id: 'patterns',
  title: 'Common Stack Patterns',
  content: `**Pattern 1: Matching Pairs (Parentheses Validation)**

**Problem:** Validate balanced parentheses: \`(){}[]\`

**Visualization:**
\`\`\`
Input: "({[]})"
Step 1: '(' → push '('        Stack: ['(']
Step 2: '{' → push '{'        Stack: ['(', '{']
Step 3: '[' → push '['        Stack: ['(', '{', '[']
Step 4: ']' → pop '[', match ✓ Stack: ['(', '{']
Step 5: '}' → pop '{', match ✓ Stack: ['(']
Step 6: ')' → pop '(', match ✓ Stack: []
Result: Valid (stack empty)
\`\`\`

**Key Insight:** Opening brackets push onto stack, closing brackets must match the top.

---

**Pattern 2: Monotonic Stack**

A **monotonic stack** maintains elements in increasing or decreasing order. When a new element violates the order, pop elements until the order is restored.

**Use Cases:**
- Next Greater Element
- Stock Span Problem
- Largest Rectangle in Histogram

**Monotonic Increasing Stack Example:**
\`\`\`
Input: [3, 1, 4, 1, 5]
Goal: Find next greater element for each

Processing:
Index 0 (3): Stack empty → push. Stack: [(0,3)]
Index 1 (1): 1 < 3, keep stack → push. Stack: [(0,3), (1,1)]
Index 2 (4): 4 > 1, pop (1,1) → next[1] = 4
             4 > 3, pop (0,3) → next[0] = 4
             Push (2,4). Stack: [(2,4)]
Index 3 (1): 1 < 4 → push. Stack: [(2,4), (3,1)]
Index 4 (5): 5 > 1, pop (3,1) → next[3] = 5
             5 > 4, pop (2,4) → next[2] = 5
             Push (4,5). Stack: [(4,5)]

Result: next = [4, 4, 5, 5, -1]
\`\`\`

---

**Pattern 3: Stack with Min/Max Tracking**

Maintain both the stack values and the running min/max:
\`\`\`
push(5):  main=[5]     min=[5]
push(3):  main=[5,3]   min=[5,3]
push(7):  main=[5,3,7] min=[5,3,3]
getMin(): return 3 (top of min stack)
pop():    main=[5,3]   min=[5,3]
getMin(): return 3
\`\`\`

---

**Pattern 4: Expression Evaluation**

Use two stacks: one for operands, one for operators.

**Infix to Postfix:**
\`\`\`
Infix: (3 + 4) * 5
     ( → push to operator stack
     3 → push to operand stack
     + → push to operator stack
     4 → push to operand stack
     ) → pop until '(', evaluate: 3 + 4 = 7
     * → push to operator stack
     5 → push to operand stack
     End → pop all: 7 * 5 = 35
\`\`\``,
};
