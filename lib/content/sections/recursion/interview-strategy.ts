/**
 * Recursion in Interviews Section
 */

export const interviewstrategySection = {
  id: 'interview-strategy',
  title: 'Recursion in Interviews',
  content: `## Mastering Recursive Interview Problems

Recursion is a common interview topic. Here's how to approach it confidently.

---

## Recognizing Recursive Problems

**🚨 Recursion Red Flags:**

1. **"Find all..."** or **"Generate all..."**
   - All permutations, combinations, subsets
   - Usually requires backtracking

2. **Tree or graph problems**
   - Tree traversal, path finding
   - Graph DFS/BFS

3. **Divide and conquer**
   - Binary search, merge sort, quick sort
   - Problem naturally divides in half

4. **"Optimal substructure"**
   - Solution built from subproblem solutions
   - Often DP candidate

5. **Nested structures**
   - Nested lists, nested JSON
   - File systems, organizational hierarchies

6. **Mathematical recurrences**
   - Fibonacci, factorial, combinations
   - Defined by recurrence relation

---

## Step-by-Step Approach

### 1. Clarify the Problem
\`\`\`
Questions to ask:
- What are the base cases? (empty, size 1)
- What's the range of inputs? (stack depth concerns)
- Should I modify input or create new structures?
- Are there performance requirements?
\`\`\`

### 2. Define Base Case(s)
\`\`\`python
# Start here - what's the simplest case?
def solve(input):
    # Base case(s) - when do we stop?
    if len(input) == 0:
        return ...
    if len(input) == 1:
        return ...
\`\`\`

### 3. Define Recursive Case
\`\`\`python
# How do we break down the problem?
def solve(input):
    # ... base cases ...
    
    # Recursive case - make problem smaller
    smaller_result = solve(smaller_input)
    
    # Combine with current element
    return combine(current, smaller_result)
\`\`\`

### 4. Trust the Recursion
- Don't try to trace all calls mentally
- Assume recursive call works correctly
- Focus on current level only

### 5. Test with Small Examples
\`\`\`python
# Test progression:
solve([])       # Empty - base case
solve([1])      # Single element - base case
solve([1, 2])   # Two elements - first real case
solve([1,2,3])  # Three elements - verify pattern
\`\`\`

---

## Common Interview Patterns

### Pattern 1: Process and Recurse
**Template:**
\`\`\`python
def process_list(arr, index=0):
    # Base case: processed all elements
    if index >= len(arr):
        return base_value
    
    # Process current element
    current = process(arr[index])
    
    # Recurse on rest
    rest = process_list(arr, index + 1)
    
    # Combine
    return combine(current, rest)
\`\`\`

**Example: Sum of list**
\`\`\`python
def sum_list(arr, index=0):
    if index >= len(arr):
        return 0
    return arr[index] + sum_list(arr, index + 1)
\`\`\`

### Pattern 2: Divide and Conquer
**Template:**
\`\`\`python
def divide_conquer(arr, left, right):
    # Base case: single element or empty
    if left >= right:
        return base_value
    
    # Divide
    mid = (left + right) // 2
    
    # Conquer
    left_result = divide_conquer(arr, left, mid)
    right_result = divide_conquer(arr, mid + 1, right)
    
    # Combine
    return merge(left_result, right_result)
\`\`\`

**Example: Binary search**
\`\`\`python
def binary_search(arr, target, left, right):
    if left > right:
        return -1
    
    mid = (left + right) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search(arr, target, mid + 1, right)
    else:
        return binary_search(arr, target, left, mid - 1)
\`\`\`

### Pattern 3: Backtracking
**Template:**
\`\`\`python
def backtrack(choices, current_path, results):
    # Base case: valid solution found
    if is_solution(current_path):
        results.append(current_path[:])  # Make copy!
        return
    
    # Try each choice
    for choice in choices:
        if is_valid(choice, current_path):
            # Make choice
            current_path.append(choice)
            
            # Recurse
            backtrack(choices, current_path, results)
            
            # Undo choice (backtrack)
            current_path.pop()
\`\`\`

**Example: Generate all subsets**
\`\`\`python
def subsets(nums):
    def backtrack(index, current):
        # Base case: processed all elements
        if index == len(nums):
            result.append(current[:])
            return
        
        # Choice 1: include nums[index]
        current.append(nums[index])
        backtrack(index + 1, current)
        current.pop()
        
        # Choice 2: don't include nums[index]
        backtrack(index + 1, current)
    
    result = []
    backtrack(0, [])
    return result
\`\`\`

### Pattern 4: Tree Recursion
**Template:**
\`\`\`python
def traverse_tree(node):
    # Base case: empty node
    if node is None:
        return base_value
    
    # Process current node
    current = process(node.val)
    
    # Recurse on children
    left_result = traverse_tree(node.left)
    right_result = traverse_tree(node.right)
    
    # Combine
    return combine(current, left_result, right_result)
\`\`\`

**Example: Tree height**
\`\`\`python
def max_depth(root):
    if root is None:
        return 0
    
    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)
    
    return 1 + max(left_depth, right_depth)
\`\`\`

---

## Communication Tips

**During the interview, say:**

1. **Identify recursion:**
   > "This looks like a recursive problem because..."

2. **Define base case:**
   > "The base case is when... and we return..."

3. **Explain recursive case:**
   > "For the recursive case, we break it down by..."

4. **Show trust:**
   > "Assuming the recursive call correctly solves the smaller problem..."

5. **Discuss complexity:**
   > "This makes N recursive calls with depth D, so time is O(...)"

---

## Complexity Analysis

**Time Complexity:**
\`\`\`
Questions to ask:
- How many recursive calls per invocation?
- What's the depth of recursion?
- How much work per call?

Common patterns:
- Linear recursion (1 call): O(n)
- Binary recursion (2 calls): O(2^n) or O(n log n)
- Divide & conquer: Often O(n log n)
- With memoization: Often O(n)
\`\`\`

**Space Complexity:**
\`\`\`
- Call stack depth: O(depth)
- Additional structures: O(...)

Example:
def factorial(n):  # Space: O(n) for call stack
    if n <= 1:
        return 1
    return n * factorial(n - 1)
\`\`\`

---

## Common Interview Questions

**Easy:**
1. Calculate factorial
2. Fibonacci number
3. Sum of array
4. Reverse string
5. Power function
6. Count down from N

**Medium:**
7. Binary search (recursive)
8. Merge sort
9. Quick sort
10. Tree traversals (preorder, inorder, postorder)
11. Validate BST
12. Generate parentheses
13. Subsets / Power set
14. Permutations
15. Combinations

**Hard:**
16. N-Queens
17. Word Search
18. Serialize/Deserialize Tree
19. Longest Common Subsequence
20. Edit Distance

---

## Red Flags to Avoid

❌ **Don't:**
- Forget base case
- Modify input when you shouldn't
- Use mutable default arguments
- Trace through all calls mentally
- Ignore time/space complexity

✅ **Do:**
- Write base case first
- Test with smallest inputs
- Consider iterative alternative
- Mention memoization if applicable
- Discuss trade-offs

---

## Practice Strategy

**Week 1: Fundamentals**
- Factorial, Fibonacci, sum array
- Get comfortable with base + recursive case

**Week 2: Linear Recursion**
- String reverse, palindrome check
- List operations

**Week 3: Binary Recursion**
- Binary search, merge sort
- Tree problems

**Week 4: Backtracking**
- Subsets, permutations
- Combination problems

**Week 5: Advanced**
- N-Queens, Sudoku solver
- Graph DFS/BFS

---

## Final Interview Checklist

Before submitting solution:

- [ ] Base case is correct and will be reached
- [ ] Making progress toward base case
- [ ] Returning values (not just computing)
- [ ] No mutable default arguments
- [ ] Tested with edge cases (empty, single element)
- [ ] Discussed time/space complexity
- [ ] Considered memoization if needed
- [ ] Considered iterative alternative

**Remember:** Recursion is elegant but requires practice. Master the patterns, trust the process, and you'll ace recursive interview questions!`,
};
