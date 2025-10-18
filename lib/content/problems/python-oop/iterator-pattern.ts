/**
 * Iterator Pattern
 * Problem ID: oop-iterator-pattern
 * Order: 42
 */

import { Problem } from '../../../types';

export const iterator_patternProblem: Problem = {
  id: 'oop-iterator-pattern',
  title: 'Iterator Pattern',
  difficulty: 'Medium',
  description: `Implement iterator pattern for custom collection.

**Pattern:**
- Traverse collection without exposing structure
- __iter__ and __next__
- Raise StopIteration when done

This tests:
- Iterator protocol
- Custom iteration
- Collection traversal`,
  examples: [
    {
      input: 'for item in collection',
      output: 'Custom iteration logic',
    },
  ],
  constraints: ['Implement __iter__ and __next__', 'Hide internal structure'],
  hints: [
    '__iter__ returns iterator',
    '__next__ returns next item',
    'Raise StopIteration at end',
  ],
  starterCode: `class ReverseIterator:
    """Iterator that goes backward"""
    def __init__(self, data):
        self.data = data
        self.index = len(data)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index -= 1
        return self.data[self.index]


class ReverseCollection:
    """Collection with reverse iteration"""
    def __init__(self, data):
        self.data = data
    
    def __iter__(self):
        """Return iterator"""
        return ReverseIterator(self.data)


def test_iterator():
    """Test iterator pattern"""
    collection = ReverseCollection([1, 2, 3, 4, 5])
    
    # Iterate in reverse
    result = list(collection)
    
    # Should be [5, 4, 3, 2, 1]
    return sum(result)
`,
  testCases: [
    {
      input: [],
      expected: 15,
      functionName: 'test_iterator',
    },
  ],
  solution: `class ReverseIterator:
    def __init__(self, data):
        self.data = data
        self.index = len(data)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index -= 1
        return self.data[self.index]


class ReverseCollection:
    def __init__(self, data):
        self.data = data
    
    def __iter__(self):
        return ReverseIterator(self.data)


def test_iterator():
    collection = ReverseCollection([1, 2, 3, 4, 5])
    result = list(collection)
    return sum(result)`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1) for iterator',
  order: 42,
  topic: 'Python Object-Oriented Programming',
};
