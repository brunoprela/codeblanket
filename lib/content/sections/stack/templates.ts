/**
 * Code Templates Section
 */

export const templatesSection = {
  id: 'templates',
  title: 'Code Templates',
  content: `**Template 1: Basic Stack Operations**
\`\`\`python
class Stack:
    def __init__(self):
        self.items = []
    
    def push (self, item):
        self.items.append (item)
    
    def pop (self):
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("pop from empty stack")
    
    def peek (self):
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("peek from empty stack")
    
    def is_empty (self):
        return len (self.items) == 0
    
    def size (self):
        return len (self.items)
\`\`\`

**Template 2: Monotonic Stack (Next Greater Element)**
\`\`\`python
def monotonic_stack_pattern (nums: List[int]) -> List[int]:
    """
    Generic monotonic stack template.
    For 'next greater': use while nums[stack[-1]] < nums[i]
    For 'next smaller': use while nums[stack[-1]] > nums[i]
    """
    n = len (nums)
    result = [-1] * n
    stack = []
    
    for i in range (n):
        # Adjust condition based on problem:
        # < for next greater, > for next smaller
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append (i)
    
    return result
\`\`\`

**Template 3: Stack with Min/Max Tracking**
\`\`\`python
class MinStack:
    """
    Stack that supports O(1) getMin operation.
    """
    def __init__(self):
        self.stack = []
        self.min_stack = []
    
    def push (self, val: int):
        self.stack.append (val)
        # Push current min (or val if stack was empty)
        if not self.min_stack:
            self.min_stack.append (val)
        else:
            self.min_stack.append (min (val, self.min_stack[-1]))
    
    def pop (self):
        self.stack.pop()
        self.min_stack.pop()
    
    def top (self):
        return self.stack[-1]
    
    def getMin (self):
        return self.min_stack[-1]
\`\`\`

**Template 4: Balanced Parentheses**
\`\`\`python
def is_valid_parentheses (s: str) -> bool:
    """
    Check if brackets/parentheses are balanced.
    """
    stack = []
    pairs = {'(': ')', '{': '}', '[': ']'}
    
    for char in s:
        if char in pairs:  # Opening bracket
            stack.append (char)
        else:  # Closing bracket
            if not stack or pairs[stack.pop()] != char:
                return False
    
    return len (stack) == 0
\`\`\``,
};
