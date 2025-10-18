/**
 * Custom Iterator Class
 * Problem ID: intermediate-custom-iterator-class
 * Order: 33
 */

import { Problem } from '../../../types';

export const intermediate_custom_iterator_classProblem: Problem = {
  id: 'intermediate-custom-iterator-class',
  title: 'Custom Iterator Class',
  difficulty: 'Medium',
  description: `Create a class that implements the iterator protocol.

Iterator protocol requires:
- __iter__() returns self
- __next__() returns next value or raises StopIteration

**Use Case:** Custom sequences, data streaming

This tests:
- Iterator protocol
- __iter__ and __next__
- StopIteration`,
  examples: [
    {
      input: 'Counter(0, 5)',
      output: 'Yields 0, 1, 2, 3, 4',
    },
  ],
  constraints: [
    'Implement __iter__ and __next__',
    'Raise StopIteration when done',
  ],
  hints: [
    '__iter__ returns self',
    '__next__ returns next value',
    'Raise StopIteration at end',
  ],
  starterCode: `class Counter:
    """
    Iterator that counts from start to end.
    
    Examples:
        >>> for i in Counter(0, 3):
        ...     print(i)
        0
        1
        2
    """
    def __init__(self, start, end):
        self.current = start
        self.end = end
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        value = self.current
        self.current += 1
        return value


def test_iterator():
    """Test custom iterator"""
    counter = Counter(0, 5)
    result = list(counter)
    return sum(result)
`,
  testCases: [
    {
      input: [],
      expected: 10,
      functionName: 'test_iterator',
    },
  ],
  solution: `class Counter:
    def __init__(self, start, end):
        self.current = start
        self.end = end
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        value = self.current
        self.current += 1
        return value


def test_iterator():
    counter = Counter(0, 5)
    result = list(counter)
    return sum(result)`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  order: 33,
  topic: 'Python Intermediate',
};
