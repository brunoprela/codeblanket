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
      'When you encounter a bar shorter than the top of the stack, it means previous bars cannot extend further right',
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
  // EASY - Baseball Game
  {
    id: 'baseball-game',
    title: 'Baseball Game',
    difficulty: 'Easy',
    topic: 'Stack',
    order: 4,
    description: `You are keeping the scores for a baseball game with strange rules. At the beginning of the game, you start with an empty record.

You are given a list of strings \`operations\`, where \`operations[i]\` is the \`ith\` operation you must apply to the record and is one of the following:

- An integer \`x\`: Record a new score of \`x\`.
- \`"+"\`: Record a new score that is the sum of the previous two scores.
- \`"D"\`: Record a new score that is the double of the previous score.
- \`"C"\`: Invalidate the previous score, removing it from the record.

Return the sum of all the scores on the record after applying all the operations.`,
    examples: [
      {
        input: 'ops = ["5","2","C","D","+"]',
        output: '30',
        explanation:
          '"5" - Add 5 to record: [5]. "2" - Add 2: [5, 2]. "C" - Remove 2: [5]. "D" - Add 10 (double 5): [5, 10]. "+" - Add 15 (sum of 5 and 10): [5, 10, 15]. Total sum = 30.',
      },
    ],
    constraints: [
      '1 <= operations.length <= 1000',
      'operations[i] is "C", "D", "+", or a string representing an integer',
    ],
    hints: [
      'Use a stack to keep track of valid scores',
      'Process each operation according to the rules',
      'Return the sum of all elements in the stack',
    ],
    starterCode: `from typing import List

def cal_points(operations: List[str]) -> int:
    """
    Calculate final score after all operations.
    
    Args:
        operations: List of score operations
        
    Returns:
        Sum of all valid scores
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [['5', '2', 'C', 'D', '+']],
        expected: 30,
      },
      {
        input: [['5', '-2', '4', 'C', 'D', '9', '+', '+']],
        expected: 27,
      },
      {
        input: [['1']],
        expected: 1,
      },
    ],
    solution: `from typing import List

def cal_points(operations: List[str]) -> int:
    """
    Stack to track valid scores.
    Time: O(n), Space: O(n)
    """
    stack = []
    
    for op in operations:
        if op == '+':
            # Sum of last two scores
            stack.append(stack[-1] + stack[-2])
        elif op == 'D':
            # Double the last score
            stack.append(2 * stack[-1])
        elif op == 'C':
            # Remove last score
            stack.pop()
        else:
            # Add integer score
            stack.append(int(op))
    
    return sum(stack)
`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    leetcodeUrl: 'https://leetcode.com/problems/baseball-game/',
    youtubeUrl: 'https://www.youtube.com/watch?v=Id_tqGdsZQI',
  },
  // EASY - Remove All Adjacent Duplicates In String
  {
    id: 'remove-adjacent-duplicates',
    title: 'Remove All Adjacent Duplicates In String',
    difficulty: 'Easy',
    topic: 'Stack',
    order: 5,
    description: `You are given a string \`s\` consisting of lowercase English letters. A **duplicate removal** consists of choosing two **adjacent** and **equal** letters and removing them.

We repeatedly make **duplicate removals** on \`s\` until we no longer can.

Return the final string after all such duplicate removals have been made. It can be proven that the answer is **unique**.`,
    examples: [
      {
        input: 's = "abbaca"',
        output: '"ca"',
        explanation:
          'Remove "bb" to get "aaca", then remove "aa" to get "ca".',
      },
      {
        input: 's = "azxxzy"',
        output: '"ay"',
      },
    ],
    constraints: ['1 <= s.length <= 10^5', 's consists of lowercase English letters'],
    hints: [
      'Use a stack to process characters',
      'If top of stack equals current char, pop it',
      'Otherwise push current char',
    ],
    starterCode: `def remove_duplicates(s: str) -> str:
    """
    Remove all adjacent duplicate characters.
    
    Args:
        s: Input string
        
    Returns:
        String after removing all adjacent duplicates
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['abbaca'],
        expected: 'ca',
      },
      {
        input: ['azxxzy'],
        expected: 'ay',
      },
      {
        input: ['aa'],
        expected: '',
      },
    ],
    solution: `def remove_duplicates(s: str) -> str:
    """
    Stack to track non-duplicate characters.
    Time: O(n), Space: O(n)
    """
    stack = []
    
    for char in s:
        if stack and stack[-1] == char:
            # Remove adjacent duplicate
            stack.pop()
        else:
            # Add character
            stack.append(char)
    
    return ''.join(stack)
`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    leetcodeUrl:
      'https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/',
    youtubeUrl: 'https://www.youtube.com/watch?v=w6LcypDgC4w',
  },
  // EASY - Backspace String Compare
  {
    id: 'backspace-string-compare',
    title: 'Backspace String Compare',
    difficulty: 'Easy',
    topic: 'Stack',
    order: 6,
    description: `Given two strings \`s\` and \`t\`, return \`true\` if they are equal when both are typed into empty text editors. \`'#'\` means a backspace character.

Note that after backspacing an empty text, the text will continue empty.`,
    examples: [
      {
        input: 's = "ab#c", t = "ad#c"',
        output: 'true',
        explanation: 'Both s and t become "ac".',
      },
      {
        input: 's = "ab##", t = "c#d#"',
        output: 'true',
        explanation: 'Both s and t become "".',
      },
      {
        input: 's = "a#c", t = "b"',
        output: 'false',
        explanation: 's becomes "c" while t becomes "b".',
      },
    ],
    constraints: [
      '1 <= s.length, t.length <= 200',
      's and t only contain lowercase letters and "#" characters',
    ],
    hints: [
      'Use a stack to process each string',
      'When you see "#", pop from stack if not empty',
      'Otherwise, push the character',
      'Compare the final stacks',
    ],
    starterCode: `def backspace_compare(s: str, t: str) -> bool:
    """
    Compare two strings with backspace characters.
    
    Args:
        s: First string
        t: Second string
        
    Returns:
        True if they are equal after processing backspaces
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['ab#c', 'ad#c'],
        expected: true,
      },
      {
        input: ['ab##', 'c#d#'],
        expected: true,
      },
      {
        input: ['a#c', 'b'],
        expected: false,
      },
    ],
    solution: `def backspace_compare(s: str, t: str) -> bool:
    """
    Build final strings using stacks and compare.
    Time: O(n + m), Space: O(n + m)
    """
    def build_string(string: str) -> str:
        """Helper to process backspaces"""
        stack = []
        for char in string:
            if char == '#':
                if stack:
                    stack.pop()
            else:
                stack.append(char)
        return ''.join(stack)
    
    return build_string(s) == build_string(t)

# Alternative: O(1) space using two pointers from end
def backspace_compare_optimal(s: str, t: str) -> bool:
    """
    Process from end to avoid extra space.
    Time: O(n + m), Space: O(1)
    """
    def next_valid_char_index(string: str, index: int) -> int:
        """Find next valid character index"""
        backspace_count = 0
        while index >= 0:
            if string[index] == '#':
                backspace_count += 1
            elif backspace_count > 0:
                backspace_count -= 1
            else:
                break
            index -= 1
        return index
    
    i, j = len(s) - 1, len(t) - 1
    
    while i >= 0 or j >= 0:
        i = next_valid_char_index(s, i)
        j = next_valid_char_index(t, j)
        
        if i >= 0 and j >= 0:
            if s[i] != t[j]:
                return False
        elif i >= 0 or j >= 0:
            return False
        
        i -= 1
        j -= 1
    
    return True
`,
    timeComplexity: 'O(n + m)',
    spaceComplexity: 'O(n + m) for stack, O(1) for two-pointer',
    leetcodeUrl: 'https://leetcode.com/problems/backspace-string-compare/',
    youtubeUrl: 'https://www.youtube.com/watch?v=4YgzB_8dE8Y',
  },
  // EASY - Next Greater Element I
  {
    id: 'next-greater-element-i',
    title: 'Next Greater Element I',
    difficulty: 'Easy',
    topic: 'Stack',
    order: 7,
    description: `The **next greater element** of some element \`x\` in an array is the **first greater** element that is **to the right** of \`x\` in the same array.

You are given two **distinct 0-indexed** integer arrays \`nums1\` and \`nums2\`, where \`nums1\` is a subset of \`nums2\`.

For each \`0 <= i < nums1.length\`, find the index \`j\` such that \`nums1[i] == nums2[j]\` and determine the **next greater element** of \`nums2[j]\` in \`nums2\`. If there is no next greater element, then the answer for this query is \`-1\`.

Return an array \`ans\` of length \`nums1.length\` such that \`ans[i]\` is the **next greater element** as described above.`,
    examples: [
      {
        input: 'nums1 = [4,1,2], nums2 = [1,3,4,2]',
        output: '[-1,3,-1]',
        explanation:
          'Next greater of 4 is -1 (no greater). Next greater of 1 is 3. Next greater of 2 is -1.',
      },
      {
        input: 'nums1 = [2,4], nums2 = [1,2,3,4]',
        output: '[3,-1]',
      },
    ],
    constraints: [
      '1 <= nums1.length <= nums2.length <= 1000',
      '0 <= nums1[i], nums2[i] <= 10^4',
      'All integers in nums1 and nums2 are unique',
      'All the integers of nums1 also appear in nums2',
    ],
    hints: [
      'Use a monotonic stack to find next greater elements',
      'Build a hash map of num -> next greater for all nums2 elements',
      'Then lookup each nums1 element in the map',
    ],
    starterCode: `from typing import List

def next_greater_element(nums1: List[int], nums2: List[int]) -> List[int]:
    """
    Find next greater element for each element in nums1.
    
    Args:
        nums1: Query array (subset of nums2)
        nums2: Full array
        
    Returns:
        Array of next greater elements
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[4, 1, 2], [1, 3, 4, 2]],
        expected: [-1, 3, -1],
      },
      {
        input: [[2, 4], [1, 2, 3, 4]],
        expected: [3, -1],
      },
    ],
    solution: `from typing import List

def next_greater_element(nums1: List[int], nums2: List[int]) -> List[int]:
    """
    Monotonic stack + hash map.
    Time: O(n + m), Space: O(n)
    """
    # Build map of num -> next greater for nums2
    next_greater = {}
    stack = []
    
    for num in nums2:
        # Pop smaller elements and record their next greater
        while stack and stack[-1] < num:
            next_greater[stack.pop()] = num
        stack.append(num)
    
    # Remaining elements have no next greater
    for num in stack:
        next_greater[num] = -1
    
    # Build result for nums1
    return [next_greater[num] for num in nums1]
`,
    timeComplexity: 'O(n + m)',
    spaceComplexity: 'O(n)',
    leetcodeUrl: 'https://leetcode.com/problems/next-greater-element-i/',
    youtubeUrl: 'https://www.youtube.com/watch?v=68a1Dc_qVq4',
  },
  // EASY - Implement Queue using Stacks
  {
    id: 'implement-queue-using-stacks',
    title: 'Implement Queue using Stacks',
    difficulty: 'Easy',
    topic: 'Stack',
    order: 8,
    description: `Implement a first in first out (FIFO) queue using only two stacks. The implemented queue should support all the functions of a normal queue (\`push\`, \`peek\`, \`pop\`, and \`empty\`).

Implement the \`MyQueue\` class:

- \`void push(int x)\` Pushes element x to the back of the queue.
- \`int pop()\` Removes the element from the front of the queue and returns it.
- \`int peek()\` Returns the element at the front of the queue.
- \`boolean empty()\` Returns \`true\` if the queue is empty, \`false\` otherwise.

**Notes:**
- You must use **only** standard operations of a stack.
- All the calls to \`pop\` and \`peek\` are valid.`,
    examples: [
      {
        input:
          '["MyQueue", "push", "push", "peek", "pop", "empty"]\n[[], [1], [2], [], [], []]',
        output: '[null, null, null, 1, 1, false]',
        explanation:
          'MyQueue myQueue = new MyQueue();\nmyQueue.push(1);\nmyQueue.push(2);\nmyQueue.peek(); // return 1\nmyQueue.pop(); // return 1\nmyQueue.empty(); // return false',
      },
    ],
    constraints: ['1 <= x <= 9', 'At most 100 calls will be made to push, pop, peek, and empty'],
    hints: [
      'Use two stacks: one for push, one for pop',
      'Transfer from push stack to pop stack when needed',
      'Amortize the transfer cost',
    ],
    starterCode: `class MyQueue:
    """
    Queue implementation using two stacks.
    """
    
    def __init__(self):
        """Initialize the queue."""
        pass
    
    def push(self, x: int) -> None:
        """Push element to back of queue."""
        pass
    
    def pop(self) -> int:
        """Remove and return front element."""
        pass
    
    def peek(self) -> int:
        """Return front element."""
        pass
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        pass
`,
    testCases: [
      {
        input: [['push', 'push', 'peek', 'pop', 'empty'], [[1], [2], [], [], []]],
        expected: [null, null, 1, 1, false],
      },
    ],
    solution: `class MyQueue:
    """
    Queue using two stacks.
    Push: O(1), Pop: O(1) amortized, Peek: O(1) amortized
    """
    
    def __init__(self):
        self.stack_in = []   # For push operations
        self.stack_out = []  # For pop/peek operations
    
    def push(self, x: int) -> None:
        """Push to input stack. O(1)"""
        self.stack_in.append(x)
    
    def pop(self) -> int:
        """Transfer if needed, then pop. O(1) amortized"""
        self._transfer_if_needed()
        return self.stack_out.pop()
    
    def peek(self) -> int:
        """Transfer if needed, then peek. O(1) amortized"""
        self._transfer_if_needed()
        return self.stack_out[-1]
    
    def empty(self) -> bool:
        """Check if both stacks are empty. O(1)"""
        return not self.stack_in and not self.stack_out
    
    def _transfer_if_needed(self) -> None:
        """Transfer from in to out if out is empty"""
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())
`,
    timeComplexity: 'O(1) amortized for all operations',
    spaceComplexity: 'O(n)',
    leetcodeUrl: 'https://leetcode.com/problems/implement-queue-using-stacks/',
    youtubeUrl: 'https://www.youtube.com/watch?v=eanwa3ht3YQ',
  },
];
