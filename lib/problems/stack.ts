import { Problem } from '../types';

export const stackProblems: Problem[] = [
  {
    id: 'valid-parentheses',
    title: 'Valid Parentheses',
    difficulty: 'Easy',
    description: `Given a string \`s\` containing just the characters \`'('\`, \`')'\`, \`'{'\`, \`'}'\`, \`'['\` and \`']'\`, determine if the input string is valid.

An input string is valid if:
1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.
3. Every close bracket has a corresponding open bracket of the same type.

**LeetCode:** [20. Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)
**YouTube:** [NeetCode - Valid Parentheses](https://www.youtube.com/watch?v=WTzjTskDFMg)

**Approach:**
Use a stack to track opening brackets. When encountering a closing bracket, check if it matches the most recent opening bracket (top of stack). If all brackets match and the stack is empty at the end, the string is valid.`,
    examples: [
      {
        input: 's = "()"',
        output: 'true',
        explanation: 'The parentheses are balanced.',
      },
      {
        input: 's = "()[]{}"',
        output: 'true',
        explanation: 'All brackets are properly matched.',
      },
      {
        input: 's = "(]"',
        output: 'false',
        explanation: 'Opening parenthesis is closed by a closing bracket.',
      },
      {
        input: 's = "([)]"',
        output: 'false',
        explanation: 'Brackets are not closed in the correct order.',
      },
    ],
    constraints: [
      '1 <= s.length <= 10^4',
      's consists of parentheses only: ()[]{}',
    ],
    hints: [
      'Use a stack to keep track of opening brackets',
      'When you encounter a closing bracket, check if it matches the top of the stack',
      'The string is valid only if the stack is empty after processing all characters',
      'Use a dictionary to map opening brackets to their closing counterparts',
    ],
    starterCode: `def is_valid(s: str) -> bool:
    """
    Determine if the parentheses/brackets are valid.
    
    Args:
        s: String containing only '(', ')', '{', '}', '[', ']'
        
    Returns:
        True if valid, False otherwise
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['()'],
        expected: true,
      },
      {
        input: ['()[]{}'],
        expected: true,
      },
      {
        input: ['(]'],
        expected: false,
      },
      {
        input: ['([)]'],
        expected: false,
      },
      {
        input: ['{[]}'],
        expected: true,
      },
      {
        input: ['(('],
        expected: false,
      },
    ],
    solution: `def is_valid(s: str) -> bool:
    stack = []
    pairs = {'(': ')', '{': '}', '[': ']'}
    
    for char in s:
        if char in pairs:  # Opening bracket
            stack.append(char)
        else:  # Closing bracket
            if not stack or pairs[stack.pop()] != char:
                return False
    
    return len(stack) == 0

# Alternative solution with explicit mapping of closing to opening
def is_valid_alt(s: str) -> bool:
    stack = []
    closing_to_opening = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in closing_to_opening:  # Closing bracket
            if not stack or stack.pop() != closing_to_opening[char]:
                return False
        else:  # Opening bracket
            stack.append(char)
    
    return len(stack) == 0`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 1,
    topic: 'Stack',
    leetcodeUrl: 'https://leetcode.com/problems/valid-parentheses/',
    youtubeUrl: 'https://www.youtube.com/watch?v=WTzjTskDFMg',
  },
  {
    id: 'min-stack',
    title: 'Min Stack',
    difficulty: 'Medium',
    description: `Design a stack that supports push, pop, top, and retrieving the minimum element in **constant time**.

Implement the \`MinStack\` class:
- \`MinStack()\` initializes the stack object.
- \`push(val)\` pushes the element \`val\` onto the stack.
- \`pop()\` removes the element on the top of the stack.
- \`top()\` gets the top element of the stack.
- \`getMin()\` retrieves the minimum element in the stack.

You must implement a solution with **O(1)** time complexity for each function.

**LeetCode:** [155. Min Stack](https://leetcode.com/problems/min-stack/)
**YouTube:** [NeetCode - Min Stack](https://www.youtube.com/watch?v=qkLl7nAwDPo)

**Approach:**
Use two parallel stacks: one for actual values, and one to track the running minimum at each level. When pushing, also push the current minimum. When popping, pop from both stacks to maintain synchronization.`,
    examples: [
      {
        input:
          '["MinStack","push","push","push","getMin","pop","top","getMin"]\n[[],[-2],[0],[-3],[],[],[],[]]',
        output: '[null,null,null,null,-3,null,0,-2]',
        explanation: `MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin(); // return -3
minStack.pop();
minStack.top();    // return 0
minStack.getMin(); // return -2`,
      },
    ],
    constraints: [
      '-2^31 <= val <= 2^31 - 1',
      'Methods pop, top and getMin will always be called on non-empty stacks',
      'At most 3 * 10^4 calls will be made to push, pop, top, and getMin',
    ],
    hints: [
      'Use two stacks: one for values, one for tracking minimums',
      'When pushing a value, also push the current minimum to the min stack',
      'When popping, pop from both stacks to keep them synchronized',
      'The top of the min stack is always the current minimum',
    ],
    starterCode: `class MinStack:
    """
    Stack that supports O(1) getMin operation.
    """
    
    def __init__(self):
        """Initialize your data structure here."""
        # Write your code here
        pass

    def push(self, val: int) -> None:
        """Push element val onto stack."""
        # Write your code here
        pass

    def pop(self) -> None:
        """Remove the element on the top of the stack."""
        # Write your code here
        pass

    def top(self) -> int:
        """Get the top element."""
        # Write your code here
        pass

    def getMin(self) -> int:
        """Retrieve the minimum element in the stack."""
        # Write your code here
        pass
`,
    testCases: [
      {
        input: [
          [
            'MinStack',
            'push',
            'push',
            'push',
            'getMin',
            'pop',
            'top',
            'getMin',
          ],
          [[], [-2], [0], [-3], [], [], [], []],
        ],
        expected: [null, null, null, null, -3, null, 0, -2],
      },
      {
        input: [
          ['MinStack', 'push', 'push', 'getMin', 'getMin', 'push', 'getMin'],
          [[], [2], [0], [], [], [3], []],
        ],
        expected: [null, null, null, 0, 0, null, 0],
      },
    ],
    solution: `class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []
    
    def push(self, val: int) -> None:
        self.stack.append(val)
        # Push current minimum (or val if stack was empty)
        if not self.min_stack:
            self.min_stack.append(val)
        else:
            self.min_stack.append(min(val, self.min_stack[-1]))
    
    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()
    
    def top(self) -> int:
        return self.stack[-1]
    
    def getMin(self) -> int:
        return self.min_stack[-1]


# Alternative: Single stack storing tuples (value, current_min)
class MinStackAlt:
    def __init__(self):
        self.stack = []  # Each element is (value, min_so_far)
    
    def push(self, val: int) -> None:
        if not self.stack:
            self.stack.append((val, val))
        else:
            current_min = min(val, self.stack[-1][1])
            self.stack.append((val, current_min))
    
    def pop(self) -> None:
        self.stack.pop()
    
    def top(self) -> int:
        return self.stack[-1][0]
    
    def getMin(self) -> int:
        return self.stack[-1][1]`,
    timeComplexity: 'O(1) for all operations',
    spaceComplexity: 'O(n)',
    order: 2,
    topic: 'Stack',
    leetcodeUrl: 'https://leetcode.com/problems/min-stack/',
    youtubeUrl: 'https://www.youtube.com/watch?v=qkLl7nAwDPo',
  },
  {
    id: 'largest-rectangle',
    title: 'Largest Rectangle in Histogram',
    difficulty: 'Hard',
    description: `Given an array of integers \`heights\` representing the histogram's bar height where the width of each bar is \`1\`, return **the area of the largest rectangle** in the histogram.

**LeetCode:** [84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)
**YouTube:** [NeetCode - Largest Rectangle in Histogram](https://www.youtube.com/watch?v=zx5Sw9130L0)

**Approach:**
Use a monotonic increasing stack to track bar indices. When a shorter bar is encountered, calculate the area of rectangles that can be formed with the taller bars as the height. The width extends to the previous shorter bar (on the stack) and the current bar.

**Key Insight:**
For each bar, we want to find:
1. **Left boundary**: The first bar to the left that is shorter (or the start)
2. **Right boundary**: The first bar to the right that is shorter (or the end)
3. **Area**: \`height[i] * (right - left - 1)\`

The monotonic stack efficiently tracks these boundaries in one pass.`,
    examples: [
      {
        input: 'heights = [2,1,5,6,2,3]',
        output: '10',
        explanation:
          'The largest rectangle has height 5 and width 2 (bars at indices 2 and 3), giving area = 10.',
      },
      {
        input: 'heights = [2,4]',
        output: '4',
        explanation: 'The largest rectangle has height 4 and width 1.',
      },
      {
        input: 'heights = [1,1]',
        output: '2',
        explanation:
          'The largest rectangle has height 1 and width 2, giving area = 2.',
      },
    ],
    constraints: ['1 <= heights.length <= 10^5', '0 <= heights[i] <= 10^4'],
    hints: [
      'Use a stack to keep track of bar indices in increasing order of heights',
      "When you encounter a bar shorter than the top of the stack, it means previous bars can't extend further right",
      'Calculate the area for each popped bar: height * width, where width is determined by the current position and the new stack top',
      'Add sentinel values (0 height) at the beginning and end to simplify edge cases',
    ],
    starterCode: `from typing import List

def largest_rectangle_area(heights: List[int]) -> int:
    """
    Find the area of the largest rectangle in the histogram.
    
    Args:
        heights: Array of bar heights
        
    Returns:
        Maximum rectangle area
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[2, 1, 5, 6, 2, 3]],
        expected: 10,
      },
      {
        input: [[2, 4]],
        expected: 4,
      },
      {
        input: [[1, 1]],
        expected: 2,
      },
      {
        input: [[2, 1, 2]],
        expected: 3,
      },
      {
        input: [[1]],
        expected: 1,
      },
      {
        input: [[4, 2, 0, 3, 2, 5]],
        expected: 6,
      },
    ],
    solution: `from typing import List

def largest_rectangle_area(heights: List[int]) -> int:
    """
    Monotonic stack solution with sentinel values.
    """
    # Add sentinel values to simplify edge cases
    heights = [0] + heights + [0]
    stack = []  # Stack stores indices
    max_area = 0
    
    for i in range(len(heights)):
        # When current bar is shorter, calculate area for taller bars
        while stack and heights[stack[-1]] > heights[i]:
            h_index = stack.pop()
            height = heights[h_index]
            
            # Width is between the new stack top and current position
            width = i - stack[-1] - 1
            max_area = max(max_area, height * width)
        
        stack.append(i)
    
    return max_area


# Alternative: Without sentinel values (more explicit)
def largest_rectangle_area_alt(heights: List[int]) -> int:
    stack = []
    max_area = 0
    
    for i in range(len(heights)):
        while stack and heights[stack[-1]] > heights[i]:
            h_index = stack.pop()
            height = heights[h_index]
            
            # Width calculation depends on whether stack is empty
            if not stack:
                width = i
            else:
                width = i - stack[-1] - 1
            
            max_area = max(max_area, height * width)
        
        stack.append(i)
    
    # Process remaining bars in stack
    while stack:
        h_index = stack.pop()
        height = heights[h_index]
        
        if not stack:
            width = len(heights)
        else:
            width = len(heights) - stack[-1] - 1
        
        max_area = max(max_area, height * width)
    
    return max_area


# Brute force solution (for comparison) - O(n^2)
def largest_rectangle_area_brute(heights: List[int]) -> int:
    max_area = 0
    
    for i in range(len(heights)):
        min_height = heights[i]
        for j in range(i, len(heights)):
            min_height = min(min_height, heights[j])
            width = j - i + 1
            max_area = max(max_area, min_height * width)
    
    return max_area`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 3,
    topic: 'Stack',
    leetcodeUrl:
      'https://leetcode.com/problems/largest-rectangle-in-histogram/',
    youtubeUrl: 'https://www.youtube.com/watch?v=zx5Sw9130L0',
  },
];
