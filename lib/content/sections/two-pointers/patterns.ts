/**
 * The Three Main Patterns Section
 */

export const patternsSection = {
  id: 'patterns',
  title: 'The Three Main Patterns',
  content: `Understanding these three patterns will help you recognize when to use two pointers:

**Pattern 1: Opposite Direction (Converging Pointers)**
**Setup:** Start at both ends, move toward center

**Visual:**
\`\`\`
[1, 2, 3, 4, 5, 6, 7, 8]
 L →           ← R
\`\`\`

**When to Use:**
- Working with sorted arrays
- Finding pairs that sum to a target
- Palindrome checking
- Reversing arrays/strings
- Container with most water

**Classic Problem:** Two Sum in Sorted Array
- If sum too small → move left pointer right (increase sum)
- If sum too large → move right pointer left (decrease sum)
- If sum equals target → found it!

**Pattern 2: Same Direction (Fast & Slow Pointers)**
**Setup:** Both start at beginning, move at different speeds

**Visual:**
\`\`\`
[1, 1, 2, 2, 3, 3, 4]
 S
 F →
\`\`\`

**When to Use:**
- Removing duplicates in-place
- Partitioning arrays
- Cycle detection in linked lists
- Finding middle element

**Classic Problem:** Remove Duplicates
- Slow pointer marks position for next unique element
- Fast pointer scans ahead to find different element
- Copy unique elements from fast to slow position

**Pattern 3: Sliding Window**
**Setup:** Two pointers define a window that slides

**Visual:**
\`\`\`
[a, b, c, d, e, f, g]
    L → R
       Window
\`\`\`

**When to Use:**
- Fixed or variable size subarray problems
- Substring problems with constraints
- Maximum/minimum window problems
- Running calculations over ranges

**Classic Problem:** Maximum Sum Subarray of Size K
- Expand window by moving right pointer
- Shrink window by moving left pointer
- Maintain window properties as you slide`,
};
