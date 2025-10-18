/**
 * advanced-enum-auto
 * Order: 47
 */

import { Problem } from '../../../types';

export const enum_autoProblem: Problem = {
  id: 'advanced-enum-auto',
  title: 'Enum with Auto Values',
  difficulty: 'Easy',
  description: `Create an Enum with automatically assigned values.

Enum features:
- Named constants
- Auto-generated values
- Iteration support
- Type safety

**Use Case:** Status codes, colors, states

This tests:
- Enum class
- auto() function
- Enum iteration`,
  examples: [
    {
      input: 'Color enum',
      output: 'RED, GREEN, BLUE with auto values',
    },
  ],
  constraints: ['Use Enum class', 'Use auto() for values'],
  hints: [
    'Import Enum and auto',
    'Values auto-increment',
    'Access by name or value',
  ],
  starterCode: `from enum import Enum, auto

class Status(Enum):
    """Status enumeration"""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()


def test_enum():
    """Test enum"""
    # Get value
    pending_value = Status.PENDING.value
    
    # Count members
    count = len(Status)
    
    # Get by value
    status = Status(2)
    
    return status.value
`,
  testCases: [
    {
      input: [],
      expected: 2,
      functionName: 'test_enum',
    },
  ],
  solution: `from enum import Enum, auto

class Status(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()


def test_enum():
    pending_value = Status.PENDING.value
    count = len(Status)
    status = Status(2)
    
    return status.value`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 47,
  topic: 'Python Advanced',
};
