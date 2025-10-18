/**
 * Type Hints with Generics
 * Problem ID: oop-type-hints-generics
 * Order: 45
 */

import { Problem } from '../../../types';

export const type_hints_genericsProblem: Problem = {
  id: 'oop-type-hints-generics',
  title: 'Type Hints with Generics',
  difficulty: 'Medium',
  description: `Use generic type hints for flexible type checking.

**Generics:**
- TypeVar for type variables
- Generic[T] base class
- Flexible type-safe code
- Used in collections, containers

This tests:
- Generic types
- Type variables
- Type safety`,
  examples: [
    {
      input: 'class Stack(Generic[T])',
      output: 'Works with any type',
    },
  ],
  constraints: ['Use TypeVar and Generic', 'Type-safe container'],
  hints: [
    'from typing import TypeVar, Generic',
    'TypeVar("T")',
    'class Stack(Generic[T])',
  ],
  starterCode: `from typing import TypeVar, Generic, List

T = TypeVar('T')


class Stack(Generic[T]):
    """Generic stack"""
    def __init__(self):
        self._items: List[T] = []
    
    def push(self, item: T) -> None:
        """Push item"""
        self._items.append(item)
    
    def pop(self) -> T:
        """Pop item"""
        return self._items.pop()
    
    def is_empty(self) -> bool:
        """Check if empty"""
        return len(self._items) == 0
    
    def size(self) -> int:
        """Get size"""
        return len(self._items)


def test_generics():
    """Test generic types"""
    # Int stack
    int_stack: Stack[int] = Stack()
    int_stack.push(1)
    int_stack.push(2)
    int_stack.push(3)
    
    # String stack
    str_stack: Stack[str] = Stack()
    str_stack.push("hello")
    str_stack.push("world")
    
    return int_stack.size() + str_stack.size()
`,
  testCases: [
    {
      input: [],
      expected: 5,
      functionName: 'test_generics',
    },
  ],
  solution: `from typing import TypeVar, Generic, List

T = TypeVar('T')


class Stack(Generic[T]):
    def __init__(self):
        self._items: List[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> T:
        return self._items.pop()
    
    def is_empty(self) -> bool:
        return len(self._items) == 0
    
    def size(self) -> int:
        return len(self._items)


def test_generics():
    int_stack: Stack[int] = Stack()
    int_stack.push(1)
    int_stack.push(2)
    int_stack.push(3)
    
    str_stack: Stack[str] = Stack()
    str_stack.push("hello")
    str_stack.push("world")
    
    return int_stack.size() + str_stack.size()`,
  timeComplexity: 'O(1) for operations',
  spaceComplexity: 'O(n)',
  order: 45,
  topic: 'Python Object-Oriented Programming',
};
