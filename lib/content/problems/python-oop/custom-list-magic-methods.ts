/**
 * Custom List with Magic Methods
 * Problem ID: custom-list-magic-methods
 * Order: 15
 */

import { Problem } from '../../../types';

export const custom_list_magic_methodsProblem: Problem = {
  id: 'custom-list-magic-methods',
  title: 'Custom List with Magic Methods',
  difficulty: 'Medium',
  category: 'python-oop',
  description: `Create a \`MyList\` class that behaves like Python's built-in list using magic methods.

Implement:
- \`__init__(items=[])\`: Initialize with items
- \`__len__()\`: Support \`len()\`
- \`__getitem__(index)\`: Support indexing \`mylist[i]\`
- \`__setitem__(index, value)\`: Support assignment \`mylist[i] = val\`
- \`__contains__(item)\`: Support \`in\` operator
- \`__iter__()\`: Support iteration
- \`__str__()\`: Return string like "[1, 2, 3]"
- \`append(item)\`: Add item to end

**Examples:**
\`\`\`python
mylist = MyList([1, 2, 3])
print(len(mylist))      # 3
print(mylist[0])        # 1
print(2 in mylist)      # True
mylist.append(4)
print(str(mylist))      # "[1, 2, 3, 4]"

for item in mylist:
    print(item)         # 1, 2, 3, 4
\`\`\``,
  starterCode: `class MyList:
    def __init__(self, items=None):
        """Initialize with items."""
        pass
    
    def __len__(self):
        """Return length."""
        pass
    
    def __getitem__(self, index):
        """Get item by index."""
        pass
    
    def __setitem__(self, index, value):
        """Set item by index."""
        pass
    
    def __contains__(self, item):
        """Check if item exists."""
        pass
    
    def __iter__(self):
        """Make iterable."""
        pass
    
    def __str__(self):
        """String representation."""
        pass
    
    def append(self, item):
        """Add item to end."""
        pass`,
  testCases: [
    {
      input: [['MyList', [1, 2, 3]], ['len']],
      expected: 3,
    },
    {
      input: [
        ['MyList', [1, 2, 3]],
        ['getitem', 1],
      ],
      expected: 2,
    },
    {
      input: [
        ['MyList', [1, 2, 3]],
        ['contains', 2],
      ],
      expected: true,
    },
  ],
  hints: [
    'Store items in internal list: self._items = items or []',
    'Delegate most operations to self._items',
    '__iter__ should return iter(self._items)',
    '__str__ can use str(self._items)',
  ],
  solution: `class MyList:
    def __init__(self, items=None):
        """Initialize with items."""
        self._items = items if items is not None else []
    
    def __len__(self):
        """Return length."""
        return len(self._items)
    
    def __getitem__(self, index):
        """Get item by index."""
        return self._items[index]
    
    def __setitem__(self, index, value):
        """Set item by index."""
        self._items[index] = value
    
    def __contains__(self, item):
        """Check if item exists."""
        return item in self._items
    
    def __iter__(self):
        """Make iterable."""
        return iter(self._items)
    
    def __str__(self):
        """String representation."""
        return str(self._items)
    
    def append(self, item):
        """Add item to end."""
        self._items.append(item)


# Test
mylist = MyList([1, 2, 3])
print(len(mylist))      # 3
print(mylist[0])        # 1
print(2 in mylist)      # True
mylist.append(4)
print(mylist)           # [1, 2, 3, 4]

for item in mylist:
    print(item)`,
  timeComplexity: 'O(1) for most operations, O(n) for __contains__',
  spaceComplexity: 'O(n)',
  order: 15,
  topic: 'Python Object-Oriented Programming',
};
