/**
 * Advanced Type Annotations
 * Problem ID: advanced-typing-annotations
 * Order: 31
 */

import { Problem } from '../../../types';

export const typing_annotationsProblem: Problem = {
  id: 'advanced-typing-annotations',
  title: 'Advanced Type Annotations',
  difficulty: 'Medium',
  description: `Use advanced typing features for better type safety and documentation.

Implement with type hints:
- Generic functions
- TypeVar for constraints
- Union and Optional types
- Callable types
- Literal types

**Benefit:** Better IDE support, documentation, and runtime type checking with tools.`,
  examples: [
    {
      input: 'def get_first(items: List[T]) -> Optional[T]',
      output: 'Generic function with type variable',
    },
  ],
  constraints: [
    'Use typing module',
    'Add comprehensive type hints',
    'Support generics where appropriate',
  ],
  hints: [
    'TypeVar("T") for generics',
    'Optional[X] = Union[X, None]',
    'Use Callable[[Args], Return]',
  ],
  starterCode: `from typing import TypeVar, List, Optional, Callable, Union, Literal, Dict, Any

T = TypeVar('T')
Number = TypeVar('Number', int, float)

def get_first(items: List[T]) -> Optional[T]:
    """Get first element or None if empty.
    
    Args:
        items: List of items
        
    Returns:
        First item or None
    """
    pass


def apply_twice(func: Callable[[T], T], value: T) -> T:
    """Apply function twice to value.
    
    Args:
        func: Function to apply
        value: Input value
        
    Returns:
        Result after applying func twice
    """
    pass


def safe_divide(a: Number, b: Number) -> Union[Number, Literal["error"]]:
    """Divide a by b, return "error" on division by zero.
    
    Args:
        a: Numerator
        b: Denominator
        
    Returns:
        Result or "error"
    """
    pass


# Test with type checking
result = get_first([1,2,3])
`,
  testCases: [
    {
      input: [],
      expected: 1,
    },
  ],
  solution: `from typing import TypeVar, List, Optional, Callable, Union, Literal, Dict, Any

T = TypeVar('T')
Number = TypeVar('Number', int, float)

def get_first(items: List[T]) -> Optional[T]:
    return items[0] if items else None


def apply_twice(func: Callable[[T], T], value: T) -> T:
    return func(func(value))


def safe_divide(a: Number, b: Number) -> Union[Number, Literal["error"]]:
    if b == 0:
        return "error"
    return a / b


# Test with type checking
result = get_first([1,2,3])`,
  timeComplexity: 'O(1) for all functions',
  spaceComplexity: 'O(1)',
  order: 31,
  topic: 'Python Advanced',
};
