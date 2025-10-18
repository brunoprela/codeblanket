/**
 * The Algorithm Step-by-Step Section
 */

export const algorithmSection = {
  id: 'algorithm',
  title: 'The Algorithm Step-by-Step',
  content: `**Algorithm Overview:**

1. **Initialize Pointers:**
   - Set \`left = 0\` (start of array)
   - Set \`right = n - 1\` (end of array)

2. **While \`left <= right\`:**
   - Calculate middle: \`mid = left + (right - left) // 2\`
   - Compare \`array[mid]\` with target:
     - **If equal:** Return \`mid\` (found!)
     - **If array[mid] < target:** Search right half (\`left = mid + 1\`)
     - **If array[mid] > target:** Search left half (\`right = mid - 1\`)

3. **If loop ends:** Return -1 (not found)

**Visual Example:**
Searching for 7 in [1, 3, 5, 7, 9, 11, 13, 15, 17]

\`\`\`
Iteration 1:
[1, 3, 5, 7, 9, 11, 13, 15, 17]
 L           M              R
Compare: 9 > 7, search left half

Iteration 2:
[1, 3, 5, 7]
 L     M   R
Compare: 5 < 7, search right half

Iteration 3:
[7]
 L/M/R
Compare: 7 == 7, FOUND at index 3!
\`\`\`

**Why This Works:**
Each comparison eliminates half the search space. After k comparisons, we've eliminated 2^k elements. This is why it's so fast!`,
};
