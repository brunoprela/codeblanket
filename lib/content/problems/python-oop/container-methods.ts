/**
 * Container Methods (__len__, __getitem__, etc.)
 * Problem ID: oop-container-methods
 * Order: 20
 */

import { Problem } from '../../../types';

export const container_methodsProblem: Problem = {
  id: 'oop-container-methods',
  title: 'Container Methods (__len__, __getitem__, etc.)',
  difficulty: 'Medium',
  description: `Implement container protocol to act like list/dict.

**Container protocol:**
- __len__ for len()
- __getitem__ for []
- __setitem__ for [] =
- __delitem__ for del []
- __contains__ for in
- __iter__ for iteration

This tests:
- Container protocol
- Custom collections
- Indexing/iteration`,
  examples: [
    {
      input: 'obj[0], len(obj), item in obj',
      output: 'Custom container behavior',
    },
  ],
  constraints: ['Implement container methods', 'Act like built-in container'],
  hints: [
    '__getitem__ for indexing',
    '__len__ for length',
    '__iter__ for for-loops',
  ],
  starterCode: `class CustomList:
    """Custom list-like container"""
    def __init__(self):
        self._items = []
    
    def __len__(self):
        """Return length"""
        return len(self._items)
    
    def __getitem__(self, index):
        """Get item by index"""
        return self._items[index]
    
    def __setitem__(self, index, value):
        """Set item by index"""
        self._items[index] = value
    
    def __contains__(self, item):
        """Check if item in container"""
        return item in self._items
    
    def __iter__(self):
        """Return iterator"""
        return iter(self._items)
    
    def append(self, item):
        """Add item"""
        self._items.append(item)


def test_container():
    """Test container methods"""
    container = CustomList()
    
    # Add items
    container.append(10)
    container.append(20)
    container.append(30)
    
    # Use len()
    length = len(container)
    
    # Use indexing
    first = container[0]
    
    # Use in
    has_20 = 20 in container
    
    # Use iteration
    total = sum(container)
    
    return total
`,
  testCases: [
    {
      input: [],
      expected: 60,
      functionName: 'test_container',
    },
  ],
  solution: `class CustomList:
    def __init__(self):
        self._items = []
    
    def __len__(self):
        return len(self._items)
    
    def __getitem__(self, index):
        return self._items[index]
    
    def __setitem__(self, index, value):
        self._items[index] = value
    
    def __contains__(self, item):
        return item in self._items
    
    def __iter__(self):
        return iter(self._items)
    
    def append(self, item):
        self._items.append(item)


def test_container():
    container = CustomList()
    container.append(10)
    container.append(20)
    container.append(30)
    
    length = len(container)
    first = container[0]
    has_20 = 20 in container
    total = sum(container)
    
    return total`,
  timeComplexity: 'O(1) for most operations',
  spaceComplexity: 'O(n)',
  order: 20,
  topic: 'Python Object-Oriented Programming',
};
