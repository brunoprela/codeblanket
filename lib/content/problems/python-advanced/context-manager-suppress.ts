/**
 * Exception Suppressing Context Manager
 * Problem ID: context-manager-suppress
 * Order: 18
 */

import { Problem } from '../../../types';

export const context_manager_suppressProblem: Problem = {
  id: 'context-manager-suppress',
  title: 'Exception Suppressing Context Manager',
  difficulty: 'Medium',
  description: `Create a context manager that suppresses specific exceptions.

The context manager should:
- Accept exception types to suppress
- Suppress only those exceptions
- Let other exceptions propagate
- Log suppressed exceptions

**Use Case:** Gracefully handling expected errors.`,
  examples: [
    {
      input: 'with suppress(ValueError): int("abc")',
      output: 'ValueError suppressed',
    },
  ],
  constraints: [
    'Accept multiple exception types',
    'Only suppress specified types',
    'Return True to suppress in __exit__',
  ],
  hints: [
    'Store exception types in __init__',
    'Check exc_type in __exit__',
    'Use isinstance for checking',
  ],
  starterCode: `class suppress:
    """
    Context manager that suppresses exceptions.
    """
    
    def __init__(self, *exceptions):
        # Your code here
        pass
    
    def __enter__(self):
        # Your code here
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Your code here
        pass


# Test
with suppress(ValueError, TypeError):
    int("not a number")  # Suppressed
    print("This runs")

with suppress(ValueError):
    1 / 0  # Not suppressed (ZeroDivisionError)


def test_suppress(suppress_types, error_type):
    """Test function for suppress context manager."""
    # Map string names to exception classes
    exception_map = {
        'ValueError': ValueError,
        'TypeError': TypeError,
        'ZeroDivisionError': ZeroDivisionError
    }
    
    # Get exception classes from strings
    exceptions_to_suppress = tuple(exception_map[name] for name in suppress_types)
    error_to_raise = exception_map[error_type]
    
    try:
        with suppress(*exceptions_to_suppress):
            raise error_to_raise("test error")
        return 'suppressed'
    except:
        return 'not suppressed'
`,
  testCases: [
    {
      input: [['ValueError'], 'ValueError'],
      expected: 'suppressed',
      functionName: 'test_suppress',
    },
    {
      input: [['ValueError'], 'TypeError'],
      expected: 'not suppressed',
      functionName: 'test_suppress',
    },
  ],
  solution: `class suppress:
    def __init__(self, *exceptions):
        self.exceptions = exceptions
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return False
        
        # Suppress if exception type matches
        if issubclass(exc_type, self.exceptions):
            print(f"Suppressed {exc_type.__name__}")
            return True  # Suppress exception
        
        return False  # Let exception propagate


def test_suppress(suppress_types, error_type):
    """Test function for suppress context manager."""
    # Map string names to exception classes
    exception_map = {
        'ValueError': ValueError,
        'TypeError': TypeError,
        'ZeroDivisionError': ZeroDivisionError
    }
    
    # Get exception classes from strings
    exceptions_to_suppress = tuple(exception_map[name] for name in suppress_types)
    error_to_raise = exception_map[error_type]
    
    try:
        with suppress(*exceptions_to_suppress):
            raise error_to_raise("test error")
        return 'suppressed'
    except:
        return 'not suppressed'`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 18,
  topic: 'Python Advanced',
};
