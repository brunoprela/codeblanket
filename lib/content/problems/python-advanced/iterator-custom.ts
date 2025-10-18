/**
 * Custom Range Iterator
 * Problem ID: iterator-custom
 * Order: 10
 */

import { Problem } from '../../../types';

export const iterator_customProblem: Problem = {
  id: 'iterator-custom',
  title: 'Custom Range Iterator',
  difficulty: 'Medium',
  description: `Implement a custom iterator class similar to Python's range().

The iterator should:
- Support start, stop, and step parameters
- Implement __iter__ and __next__
- Raise StopIteration when done
- Support both forward and backward iteration

**Pattern:**
python
for i in CustomRange(0, 10, 2):
    print(i)  # 0, 2, 4, 6, 8
`,
  examples: [
    {
      input: 'CustomRange(0, 10, 2)',
      output: '[0, 2, 4, 6, 8]',
    },
    {
      input: 'CustomRange(10, 0, -2)',
      output: '[10, 8, 6, 4, 2]',
    },
  ],
  constraints: [
    'Must implement iterator protocol',
    'Support positive and negative step',
    'Handle edge cases',
  ],
  hints: [
    '__iter__ should return self',
    'Track current value',
    'Check stopping condition in __next__',
  ],
  starterCode: `class CustomRange:
    """
    Custom range iterator.
    """
    
    def __init__(self, start, stop, step=1):
        # TODO: Store start, stop, step and initialize current
        self.start = start
        self.stop = stop
        self.step = step
        self.current = start
    
    def __iter__(self):
        # TODO: Return self and reset current
        pass
    
    def __next__(self):
        # TODO: Check if done, return current and increment
        pass


# Test
result = list(CustomRange(0, 10, 2))
print(result)


# Test helper function (for automated testing)
def test_custom_range(start, stop, step):
    """Test function for CustomRange - implement the class methods above first!"""
    try:
        return list(CustomRange(start, stop, step))
    except:
        return None  # Return None if methods not yet implemented
`,
  testCases: [
    {
      input: [0, 10, 2],
      expected: [0, 2, 4, 6, 8],
      functionName: 'test_custom_range',
    },
    {
      input: [10, 0, -2],
      expected: [10, 8, 6, 4, 2],
      functionName: 'test_custom_range',
    },
  ],
  solution: `class CustomRange:
    def __init__(self, start, stop, step=1):
        self.current = start
        self.stop = stop
        self.step = step
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if (self.step > 0 and self.current >= self.stop) or \\
           (self.step < 0 and self.current <= self.stop):
            raise StopIteration
        
        value = self.current
        self.current += self.step
        return value


# Test helper function (for automated testing)
def test_custom_range(start, stop, step):
    """Test function for CustomRange."""
    return list(CustomRange(start, stop, step))`,
  timeComplexity: 'O(1) per iteration',
  spaceComplexity: 'O(1)',
  order: 10,
  topic: 'Python Advanced',
};
